#!/usr/bin/env python
"""Out-of-class oscillator against the paper's classical class C = {A^xy = 0} (R^xy free).

Same data as backaction_oscillator.py (third-order damped oscillator, position observed,
velocity latent, AR(1) forcing; plus the in-class control). Compared models, all fitted by
exact y-likelihood on the first 70 % of each record and scored by one-step held-out
prediction error on the rest:

  pairwise   p = q = 1, A^xy free, R full          (6 parameters)
  C, p = 1   p = q = 1, A^xy = 0,  R full          (5)  -- the paper's classical class
  C, p = 2   p = 2, q = 1, A^xy = 0, R full (3x3)  (13) -- one more latent state

The p = 2 model's likelihood is the exact stationary Gaussian likelihood of y, computed by
a Cholesky factorization of the Toeplitz autocovariance matrix (equivalent to the Kalman
filter, much faster in Python); its held-out predictions are the innovations of the same
factorization on the test segment, from the stationary prior, as for the p = 1 models.

Reports: complex fitted poles (count over realizations), held-out gain over C(p=1) with
paired Diebold--Mariano tests, and AIC/BIC on the training likelihood, counting the
identifiable dimension of each y-law (4, 4, 6) rather than the raw parameters above, with
the number of realizations in which each model is preferred.

Run (from the paper directory): python -B experiments/backaction_oscillator_classC.py [M]
  [--oscillator-only: keep the saved in-class control] [--replot: redraw the figure]
Output: experiments/backaction_oscillator_classC_results.json,
        figures/backaction_poles_classC.pdf (fitted poles of the three models)
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_discrete_lyapunov, solve_triangular, toeplitz
from scipy.optimize import minimize
from scipy.stats import t as student_t

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backaction_oscillator as bo  # noqa: E402

OUT_JSON = Path(__file__).resolve().parent / "backaction_oscillator_classC_results.json"


# ---- C with p = 2 latent states: z = (x1, x2, y), A^xy = 0 -------------------------------
def unpack_p2(th):
    Axx = th[0:4].reshape(2, 2)
    Ayx = th[4:6]
    ayy = th[6]
    A = np.zeros((3, 3))
    A[:2, :2] = Axx
    A[2, :2] = Ayx
    A[2, 2] = ayy
    L = np.zeros((3, 3))
    L[np.tril_indices(3)] = th[7:13]
    L[np.diag_indices(3)] = np.exp(L[np.diag_indices(3)])
    return A, L @ L.T


def acov_y(A, R, n):
    if np.max(np.abs(np.linalg.eigvals(A))) >= 0.999:
        return None
    S = solve_discrete_lyapunov(A, R)
    v = S[:, 2].copy()
    g = np.empty(n)
    for h in range(n):
        g[h] = v[2]
        v = A @ v
    return g


def loglik_toeplitz(y, g):
    try:
        c, low = cho_factor(toeplitz(g))
    except np.linalg.LinAlgError:
        return -np.inf
    return float(-0.5 * (2 * np.log(np.diag(c)).sum() + y @ cho_solve((c, low), y)
                         + len(y) * np.log(2 * np.pi)))


def innovations(y, g):
    C = np.linalg.cholesky(toeplitz(g))
    return np.diag(C) * solve_triangular(C, y, lower=True)


def fit_p2(Ytr, starts=4):
    n = len(Ytr)

    def nll(th):
        A, R = unpack_p2(th)
        g = acov_y(A, R, n)
        if g is None:
            return 1e10
        ll = loglik_toeplitz(Ytr, g)
        return -ll if np.isfinite(ll) else 1e10
    best = None
    for s in range(starts):
        rng = np.random.default_rng(100 + s)
        th0 = np.r_[0.5, 0.2, -0.2, 0.5, 0.3, 0.1, 0.3, np.zeros(6)] + 0.2 * rng.standard_normal(13)
        th0[[7, 9, 12]] = np.log(0.3) + 0.2 * rng.standard_normal(3)   # diagonal of L
        r = minimize(nll, th0, method="L-BFGS-B", options={"maxiter": 3000})
        if best is None or r.fun < best.fun:
            best = r
    A, R = unpack_p2(best.x)
    return A, R, -best.fun


def heldout_p2(Yte, A, R):
    e = innovations(Yte, acov_y(A, R, len(Yte)))
    return float(np.mean(e[1:] ** 2))


def dm(a, b):
    """Paired Diebold--Mariano on per-realization MSE: mean(a - b) / s.e., two-sided p."""
    d = np.asarray(a) - np.asarray(b)
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    return float(t), float(2 * student_t.sf(abs(t), len(d) - 1))


def run(sim_fn, label, M, N=600, split=0.7):
    ntr = int(N * split)
    rec = {"mse": {"pairwise": [], "C1": [], "C2": []}, "ll": {"pairwise": [], "C1": [], "C2": []},
           "complex": {"pairwise": 0, "C1": 0, "C2": 0},
           "poles": {"pairwise": [], "C1": [], "C2": []}}
    # identifiable dimensions: the y-law of each model (Prop. 1), not its raw parameter count;
    # pairwise and C(p=1) are ARMA(2,1) laws (4), C(p=2) an ARMA(3,2) law (6)
    npar = {"pairwise": 4, "C1": 4, "C2": 6}
    for m in range(M):
        Y = sim_fn(N, np.random.default_rng(m))          # same records as backaction_oscillator.py
        Ytr, Yte = Y[:ntr], Y[ntr:]
        for name, flags in (("pairwise", (True, True)), ("C1", (False, True))):
            A, Q = bo.fit(Ytr, *flags)
            rec["mse"][name].append(bo.heldout_mse(Yte, A, Q)[0])
            rec["ll"][name].append(bo.yfilter(Ytr, A, Q)[0])
            rec["complex"][name] += int(np.max(np.abs(np.linalg.eigvals(A).imag)) > 1e-6)
            rec["poles"][name] += [[float(z.real), float(z.imag)] for z in np.linalg.eigvals(A)]
        A2, R2, ll2 = fit_p2(Ytr)
        rec["poles"]["C2"] += [[float(z.real), float(z.imag)] for z in np.linalg.eigvals(A2)]
        rec["mse"]["C2"].append(heldout_p2(Yte, A2, R2))
        rec["ll"]["C2"].append(ll2)
        rec["complex"]["C2"] += int(np.max(np.abs(np.linalg.eigvals(A2).imag)) > 1e-6)
        print(f"  {label} {m + 1}/{M}", flush=True)
    mse = {k: np.array(v) for k, v in rec["mse"].items()}
    out = {"label": label, "M": M, "N": N, "complex": rec["complex"], "poles": rec["poles"]}
    for k in ("pairwise", "C2"):
        g = 100 * (mse["C1"] - mse[k]) / mse["C1"]
        out[f"gain_{k}_over_C1_pct"] = [float(g.mean()), float(g.std(ddof=1) / np.sqrt(M))]
        out[f"dm_{k}_vs_C1"] = dm(mse["C1"], mse[k])
    g = 100 * (mse["pairwise"] - mse["C2"]) / mse["pairwise"]
    out["gain_C2_over_pairwise_pct"] = [float(g.mean()), float(g.std(ddof=1) / np.sqrt(M))]
    out["dm_C2_vs_pairwise"] = dm(mse["pairwise"], mse["C2"])
    ntr_ = int(N * split)
    crit = {}
    for k in ("pairwise", "C1", "C2"):
        ll = np.array(rec["ll"][k])
        crit[k] = (-2 * ll + 2 * npar[k], -2 * ll + np.log(ntr_) * npar[k])
        out[f"ll_train_{k}"] = ll.tolist()
    for j, name in ((0, "aic"), (1, "bic")):
        best = np.argmin(np.vstack([crit[k][j] for k in ("pairwise", "C1", "C2")]), axis=0)
        out[f"{name}_preferred_counts"] = {k: int(np.sum(best == i))
                                           for i, k in enumerate(("pairwise", "C1", "C2"))}
    for k in ("pairwise", "C1", "C2"):
        ll = np.array(rec["ll"][k])
        out[f"aic_{k}"] = float(np.mean(-2 * ll + 2 * npar[k]))
        out[f"bic_{k}"] = float(np.mean(-2 * ll + np.log(ntr_) * npar[k]))
        out[f"heldout_mse_{k}"] = float(mse[k].mean())
    print(json.dumps(out, indent=1), flush=True)
    return out


def make_figure(res):
    """Fitted transition poles on the oscillator: pairwise, C (p=1), C (p=2), true."""
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    mpl.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "legend.fontsize": 6.5, "pdf.fonttype": 42,
                         "savefig.facecolor": "white"})
    P = res["oscillator"]["poles"]
    _, _, true = bo.true_system(np.array([0.74 + 0.32j, 0.74 - 0.32j]))
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(th), np.sin(th), color="0.8", lw=0.8, zorder=1)
    for k, mk, col, lab, sz, zo in (("C2", "^", "#009E73", r"classical $\mathcal{C}$, $p=2$", 14, 2),
                                    ("pairwise", "o", "#0072B2", r"pairwise, $p=1$", 14, 3),
                                    ("C1", "s", "#D55E00", r"classical $\mathcal{C}$, $p=1$", 22, 4)):
        z = np.array(P[k])
        ax.scatter(z[:, 0], z[:, 1], s=sz, marker=mk, color=col, alpha=0.6, label=lab, zorder=zo,
                   edgecolors="white" if k == "C1" else "none", linewidths=0.4)
    ax.scatter(true.real, true.imag, s=70, marker="x", color="k", lw=1.8, label="true", zorder=4)
    ax.axhline(0, color="0.7", lw=0.5)
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    ax.set_aspect("equal", "box")
    ax.set_xlim(0.2, 1.05)
    ax.set_ylim(-0.55, 0.55)
    ax.legend(loc="upper left", framealpha=0.85, borderpad=0.3, labelspacing=0.25,
              handletextpad=0.3, markerscale=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = Path(__file__).resolve().parents[1] / "figures" / "backaction_poles_classC"
    fig.savefig(str(out) + ".pdf")
    fig.savefig(str(out) + ".png", dpi=200)
    print(f"saved {out}.pdf/.png")


def main():
    M = next((int(a) for a in sys.argv[1:] if a.isdigit()), 40)
    if "--replot" in sys.argv:
        make_figure(json.loads(OUT_JSON.read_text()))
        return
    old = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {}
    Fo, Lo, _ = bo.true_system(np.array([0.74 + 0.32j, 0.74 - 0.32j]))
    res = {"oscillator": run(lambda N, rng: bo.simulate(Fo, Lo, N, rng), "oscillator", M)}
    if "--oscillator-only" in sys.argv and "in_class" in old:
        res["in_class"] = old["in_class"]          # keep the in-class control already computed
    else:
        A_in = np.array([[0.6, 0.0], [0.3, 0.5]])
        Q_in = np.array([[0.10, 0.0], [0.0, 0.10]])
        res["in_class"] = run(lambda N, rng: bo.simulate_couple2(A_in, Q_in, N, rng), "in-class", M)
    OUT_JSON.write_text(json.dumps(res, indent=1))
    print(f"saved {OUT_JSON}")
    make_figure(res)


if __name__ == "__main__":
    main()
