#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reproduce the back-action experiment from Section 4 of the paper
"Non-linear Gaussian pairwise Kalman filters".

Sweeps the coupling coefficient b (strength of y → x back-action) and compares:
  - EPKF on the true pairwise model (correct model)
  - UPKF on the true pairwise model (correct model)
  - EKF_naive: standard EKF on the misspecified Markov model (ignores back-action)

Generates:
  - papier_NonLinearPKF/figures/backaction_mse_nees_vs_b.png

Usage (from repo root):
    python3 -m prg.run_paper_section4_backaction
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(REPO_ROOT, "papier_NonLinearPKF", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

from prg.models.nonLinear.model_x1_y1_pairwise_param import Model_x1_y1_pairwise_param
from prg.models.nonLinear.model_x1_y1_markov_naive  import Model_x1_y1_markov_naive
from prg.classes.ParamNonLinear import ParamNonLinear
from prg.classes.NonLinear_EPKF import NonLinear_EPKF
from prg.classes.NonLinear_UPKF import NonLinear_UPKF
from prg.utils.utils import compute_errors

# ── Experiment parameters ──────────────────────────────────────────────────────
N        = 1000
N_SEEDS  = 30
SIGMA_SET = "wan2000"
Q_X      = 0.04
Q_Y      = 0.025
B_VALUES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0])


def _make_param(model):
    p = model.get_params().copy()
    dx, dy = p.pop("dim_x"), p.pop("dim_y")
    return ParamNonLinear(0, dx, dy, **p)


def _run(filt, N, data_gen=None):
    xt_l, xh_l, pp_l, ii_l, ss_l = [], [], [], [], []
    kwargs = {"N": N}
    if data_gen is not None:
        kwargs["data_generator"] = data_gen
    for _ in filt.process_filter(**kwargs):
        step = filt.history.last()
        if step["xkp1"] is not None:
            xt_l.append(step["xkp1"])
        xh_l.append(step["Xkp1_update"])
        pp_l.append(step["PXXkp1_update"])
        if step["ikp1"] is not None:
            ii_l.append(step["ikp1"].ravel())
        if step.get("Skp1") is not None:
            ss_l.append(step["Skp1"])
    i_arr = np.array(ii_l) if ii_l else None
    return xt_l, xh_l, pp_l, i_arr, ss_l or None


def run_one(b, seed):
    # ── True pairwise model ───────────────────────────────────────────────────
    mod_pw    = Model_x1_y1_pairwise_param(b=b, q_x=Q_X, q_y=Q_Y)
    param_pw  = _make_param(mod_pw)

    # ── Naive Markov model (misspecified: ignores b*tanh(y) back-action) ──────
    mod_naive   = Model_x1_y1_markov_naive(a=0.5, d=2.0, q_x=Q_X, q_y=Q_Y)
    param_naive = _make_param(mod_naive)

    # ── EPKF on true pairwise model (generates trajectory) ───────────────────
    epkf = NonLinear_EPKF(param=param_pw, sKey=seed, verbose=0)
    xt_e, xh_e, pp_e, ii_e, ss_e = _run(epkf, N)
    err_e = compute_errors(epkf, xt_e, xh_e, pp_e, ii_e, ss_e)

    shared = [(s["k"], s["xkp1"], s["ykp1"]) for s in epkf.history._history]

    # ── UPKF on same trajectory ───────────────────────────────────────────────
    upkf = NonLinear_UPKF(param=param_pw, sigmaSet=SIGMA_SET, sKey=seed, verbose=0)
    xt_u, xh_u, pp_u, ii_u, ss_u = _run(upkf, N, iter(shared))
    err_u = compute_errors(upkf, xt_u, xh_u, pp_u, ii_u, ss_u)

    # ── Naive EKF on same trajectory (misspecified model) ────────────────────
    ekf_naive = NonLinear_EPKF(param=param_naive, sKey=seed, verbose=0)
    xt_n, xh_n, pp_n, ii_n, ss_n = _run(ekf_naive, N, iter(shared))
    err_n = compute_errors(ekf_naive, xt_n, xh_n, pp_n, ii_n, ss_n)

    return (
        err_e["mse_total"], err_u["mse_total"], err_n["mse_total"],
        err_e["nees_mean"], err_u["nees_mean"], err_n["nees_mean"],
    )


def main():
    print("\n" + "=" * 65)
    print("  Section 4 — Back-action: EPKF/UPKF vs naive EKF")
    print("=" * 65)

    results = {k: {"mse": [], "nees": []} for k in ("epkf", "upkf", "naive")}

    for b in B_VALUES:
        print(f"\n  b = {b:.1f}")
        mse_e_l, mse_u_l, mse_n_l = [], [], []
        nees_e_l, nees_u_l, nees_n_l = [], [], []

        for seed in range(N_SEEDS):
            try:
                me, mu, mn, ne, nu, nn = run_one(b, seed)
                mse_e_l.append(me); mse_u_l.append(mu); mse_n_l.append(mn)
                nees_e_l.append(ne); nees_u_l.append(nu); nees_n_l.append(nn)
            except Exception as exc:
                print(f"    seed {seed}: FAILED ({exc})")

        def _stat(lst):
            a = np.array([v for v in lst if v is not None and np.isfinite(v)])
            return (a.mean(), a.std()) if len(a) else (np.nan, np.nan)

        me, se = _stat(mse_e_l);  mu, su = _stat(mse_u_l);  mn, sn = _stat(mse_n_l)
        ne, se2 = _stat(nees_e_l); nu, su2 = _stat(nees_u_l); nn_, sn2 = _stat(nees_n_l)

        results["epkf"]["mse"].append((me, se))
        results["upkf"]["mse"].append((mu, su))
        results["naive"]["mse"].append((mn, sn))
        results["epkf"]["nees"].append((ne, se2))
        results["upkf"]["nees"].append((nu, su2))
        results["naive"]["nees"].append((nn_, sn2))

        print(f"    MSE  — EPKF: {me:.4f}±{se:.4f}  UPKF: {mu:.4f}±{su:.4f}"
              f"  Naive: {mn:.4f}±{sn:.4f}")
        print(f"    NEES — EPKF: {ne:.3f}±{se2:.3f}   UPKF: {nu:.3f}±{su2:.3f}"
              f"   Naive: {nn_:.3f}±{sn2:.3f}")

    # ── Figure ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    colors = {"epkf": "tab:blue", "upkf": "tab:orange", "naive": "tab:red"}
    labels = {"epkf": "EPKF (pairwise)", "upkf": "UPKF (pairwise)", "naive": "EKF (Markov, no back-action)"}

    for metric, ax, ylabel, title in [
        ("mse",  axes[0], "MSE",       "MSE vs back-action coupling b"),
        ("nees", axes[1], "NEES mean", "NEES vs back-action coupling b"),
    ]:
        for key in ("epkf", "upkf", "naive"):
            vals  = np.array(results[key][metric])
            means = vals[:, 0]
            stds  = vals[:, 1]
            ax.semilogy(B_VALUES, means, "o-", color=colors[key],
                        label=labels[key], linewidth=1.5)
        if metric == "nees":
            ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, label="ideal (1.0)")
        ax.set_xlabel(r"Coupling coefficient $b$", fontsize=11)
        ax.set_ylabel(ylabel + " (log scale)", fontsize=11)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, which="both")

    fig.suptitle(
        "Back-action experiment: pairwise filters vs misspecified Markov EKF\n"
        f"({N_SEEDS} seeds × N={N} steps, $q_x$={Q_X}, $q_y$={Q_Y})",
        fontsize=10,
    )
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "backaction_mse_nees_vs_b.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved → {os.path.relpath(out, REPO_ROOT)}")
    print("Done.")


if __name__ == "__main__":
    main()
