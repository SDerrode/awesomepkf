#!/usr/bin/env python
"""What ignoring back-action costs, and when it becomes testable -- one model, one grid.

Truth: the scalar Gaussian pairwise Markov chain of classical_vs_pairwise_exact.py,
A = [[0.6, A^xy], [0.3, 0.4]], R = [[0.10, 0.05], [0.05, 0.10]], A^xy in [0, 0.5].
Every number is exact (no Monte Carlo): the N -> infinity (stationary, non-causal Wiener)
smoothed-state MSE and NEES of each model's smoother applied to TRUE data
(`asym_metrics`), the MSE penalty being relative to the pairwise smoother, whose error is
the Bayesian bound. Classical competitors (all in C = {A^xy = 0}):

  ablated      the true model with A^xy set to 0 (every other block kept);
  projected    the KL projection of the true transition onto C (R, incl. R^xy, and A^yy
               free) = large-sample complete-data ML (x observed in training);
  textbook     the KL projection onto y_n = H x_n + v_n;
  frozen-best  A^xx, A^yx, R kept at their true values and A^yy chosen to MINIMISE the
               state MSE (the steelman of backaction_tradeoff.py);
  capacity     a member of C whose smoother EQUALS the pairwise one (closed form,
               `classC_match_member`), with |A^yy| < 1 when such a member exists.

Testability: power at alpha = 0.05 of the back-action LRT with A^xx, A^yx, Q frozen
(the identifiable setting of the paper), P(chi2_1(lambda) > c), lambda = 2 N KL_rate,
KL_rate = min over A^yy of the Itakura--Saito divergence rate between the true y-spectrum
and the null's (Whittle), N = 400.

Run (from the paper directory; needs classical_vs_pairwise_exact.py next to it):
  PYTHONPATH=/path/to/awesomePKF python -B experiments/unified_backaction_cost.py
Output: experiments/unified_backaction_cost_results.json, figures/unified_backaction_cost.pdf
(--from-json redraws the figure from the saved results)
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import chi2, ncx2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import classical_vs_pairwise_exact as cv  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT_JSON = HERE / "unified_backaction_cost_results.json"
OUT_FIG = HERE.parent / "figures" / "unified_backaction_cost"
R_T = cv.R_TRUE
N_TEST, ALPHA = 400, 0.05
GRID = np.round(np.linspace(0.0, 0.5, 21), 4)          # A^xy


def A_of(axy, ayy=cv.AYY):
    return np.array([[cv.AXX, axy], [cv.AYX, ayy]])


def y_spec(A, R, w):
    ew = np.exp(-1j * w)
    out = np.empty(len(w))
    for k in range(len(w)):
        M = np.linalg.inv(np.eye(2) - A * ew[k])
        out[k] = np.real((M @ R @ M.conj().T)[1, 1])
    return out


def kl_frozen(axy, nw=4001):
    """Itakura--Saito KL rate of the true y-law to the best frozen null (A^xy = 0)."""
    if axy == 0.0:
        return 0.0, cv.AYY
    w = np.linspace(-np.pi, np.pi, nw)
    f = y_spec(A_of(axy), R_T, w)

    def kl(b):
        x = f / y_spec(A_of(0.0, b), R_T, w)
        return np.trapezoid(x - 1 - np.log(x), w) / (4 * np.pi)
    r = minimize_scalar(kl, bounds=(-0.95, 0.95), method="bounded",
                        options={"xatol": 1e-10})
    return float(r.fun), float(r.x)


def capacity_member(At, Rt):
    """Smallest-|A^yy| exact-matching member of C over a grid of A^xx values."""
    best = None
    for a in np.linspace(0.05, 0.99, 95):
        m = cv.classC_match_member(At, Rt, float(a))
        if m is None or not m[2]:
            continue
        Am, Rm, _, stable = m
        if best is None or abs(Am[1, 1]) < abs(best[0][1, 1]):
            best = (Am, Rm, stable)
    return best


def main():
    rows = []
    for axy in GRID:
        At = A_of(float(axy))
        base = cv.asym_metrics(At, R_T, At, R_T, 1)
        comps = {"ablated": cv.ablate(At, R_T, 1),
                 "projected": cv.project_classC_closed(At, R_T, 1, 1),
                 "textbook": cv.project_textbook_closed(At, R_T, 1, 1)[:2]}

        def mse_frozen(b):
            return cv.asym_metrics(At, R_T, A_of(0.0, b), R_T, 1)["mse"]
        fb = minimize_scalar(mse_frozen, bounds=(-0.95, 0.95), method="bounded",
                             options={"xatol": 1e-9})
        comps["frozen_best"] = (A_of(0.0, float(fb.x)), R_T)
        row = {"Axy": float(axy), "pairwise_mse": base["mse"], "pairwise_nees": base["nees"]}
        for name, (Am, Rm) in comps.items():
            m = cv.asym_metrics(At, R_T, Am, Rm, 1)
            row[name] = {"penalty_pct": 100.0 * (m["mse"] / base["mse"] - 1.0),
                         "nees": m["nees"]}
        row["frozen_best"]["Ayy"] = float(fb.x)
        cap = capacity_member(At, R_T) if axy > 0 else (At, R_T, True)
        if cap is None:
            row["capacity"] = None
        else:
            m = cv.asym_metrics(At, R_T, cap[0], cap[1], 1)
            row["capacity"] = {"penalty_pct": 100.0 * (m["mse"] / base["mse"] - 1.0),
                               "Ayy": float(cap[0][1, 1]), "stable": bool(cap[2])}
        kl, b0 = kl_frozen(float(axy))
        lam = 2 * N_TEST * kl
        row["test"] = {"kl_rate": kl, "lambda": lam, "Ayy_null": b0,
                       "power": float(ncx2.sf(chi2.ppf(1 - ALPHA, 1), 1, lam)) if lam > 0
                       else ALPHA}
        rows.append(row)
        print(f"A^xy={axy:4.2f}  abl {row['ablated']['penalty_pct']:6.1f}%  "
              f"proj {row['projected']['penalty_pct']:5.1f}%  tb {row['textbook']['penalty_pct']:5.1f}%  "
              f"frozen {row['frozen_best']['penalty_pct']:5.2f}%  "
              f"cap {row['capacity']['penalty_pct'] if row['capacity'] else float('nan'):.1e}%"
              f"{'' if not row['capacity'] or row['capacity']['stable'] else ' (unstable A^yy)'}  "
              f"power {row['test']['power']:.3f}", flush=True)
    OUT_JSON.write_text(json.dumps({"meta": {"N_test": N_TEST, "alpha": ALPHA,
                                             "R": R_T.tolist(), "Axx": cv.AXX, "Ayx": cv.AYX,
                                             "Ayy": cv.AYY},
                                    "rows": rows}, indent=1))
    print(f"saved {OUT_JSON}")
    make_figure(rows)


def make_figure(rows):
    """(a) MSE penalty of the classical smoothers (log scale; the capacity member, whose
    penalty is exactly 0, is stated in the legend title, and the textbook fit, whose
    penalty reflects the observation memory it cannot represent rather than back-action,
    is reported in the text only); (b) LRT power."""
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    mpl.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "legend.fontsize": 7, "lines.linewidth": 1.3,
                         "savefig.facecolor": "white", "pdf.fonttype": 42})
    x = np.array([r["Axy"] for r in rows])
    pos = x > 0
    col = {"ablated": "#D55E00", "projected": "#0072B2", "frozen_best": "#009E73"}
    lab = {"ablated": r"ablated ($A^{xy}$ set to 0)",
           "projected": "likelihood fit (KL projection)",
           "frozen_best": "best for the state, blocks frozen"}
    mk = {"ablated": "o", "projected": "s", "frozen_best": "^"}
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.5))
    for k in ("ablated", "projected", "frozen_best"):
        y = np.array([r[k]["penalty_pct"] for r in rows])
        ax[0].semilogy(x[pos], y[pos], "-", marker=mk[k], ms=3, color=col[k], label=lab[k])
    ax[0].set_xlabel(r"back-action $A^{xy}$")
    ax[0].set_ylabel("state-MSE penalty over pairwise (%)")
    ax[0].set_title("(a) cost of a classical smoother", fontsize=8)
    ax[0].legend(loc="lower right", frameon=False,
                 title="best in the class: 0 (exact match)", title_fontsize=7)
    ax[0].set_xlim(-0.01, 0.51)
    ax[0].grid(alpha=0.3, which="both")
    ax[1].plot(x, [r["test"]["power"] for r in rows], "-o", ms=3, color="#CC79A7")
    ax[1].axhline(ALPHA, ls=":", color="0.5", lw=0.8)
    ax[1].set_xlabel(r"back-action $A^{xy}$")
    ax[1].set_ylabel(rf"LRT power ($N={N_TEST}$, $\alpha={ALPHA}$)")
    ax[1].set_title("(b) testability", fontsize=8)
    ax[1].set_xlim(-0.01, 0.51)
    ax[1].set_ylim(-0.03, 1.05)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    OUT_FIG.parent.mkdir(exist_ok=True)
    fig.savefig(str(OUT_FIG) + ".pdf")
    fig.savefig(str(OUT_FIG) + ".png", dpi=200)
    print(f"saved {OUT_FIG}.pdf/.png")


if __name__ == "__main__":
    if "--from-json" in sys.argv:          # redraw only
        make_figure(json.loads(OUT_JSON.read_text())["rows"])
    else:
        main()
