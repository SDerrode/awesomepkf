"""
Reproduce the multiplicative noise experiment from Section 4 of the paper
"Non-linear Gaussian pairwise Kalman filters".

Sweeps the observation noise standard deviation sigma_y over 8 values
and compares EPKF, UPKF, UKF-aug across MSE and NEES.

Generates:
  - papier_NonLinearPKF/figures/multip_mse_nees_vs_sigma.png

Usage (from repo root):
    python3 -m prg.run_paper_section4_multip
"""

from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT   = Path(__file__).resolve().parent.parent
FIGURES_DIR = REPO_ROOT / "papier_NonLinearPKF" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

from prg.classes.nonlinear_epkf import NonLinear_EPKF
from prg.classes.nonlinear_ukf import NonLinear_UKF
from prg.classes.nonlinear_upkf import NonLinear_UPKF
from prg.classes.param_nonlinear import ParamNonLinear
from prg.models.nonLinear.model_x1_y1_multiplicative import Model_x1_y1_multiplicative
from prg.models.nonLinear.model_x1_y1_multiplicative_augmented import Model_x1_y1_multiplicative_augmented
from prg.utils.metrics import compute_errors

N         = 1000
N_SEEDS   = 30
SIGMA_SET = "wan2000"
SIGMA_Y_VALUES = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0])


def _build_params(q_y):
    mod_pw  = Model_x1_y1_multiplicative(q_y=q_y)
    mod_aug = Model_x1_y1_multiplicative_augmented(q_y=q_y)
    p_pw  = mod_pw.get_params().copy();  px, py = p_pw.pop("dim_x"), p_pw.pop("dim_y")
    p_aug = mod_aug.get_params().copy(); ax, ay = p_aug.pop("dim_x"), p_aug.pop("dim_y")
    param_pw  = ParamNonLinear(0, px,  py,  **p_pw)
    param_aug = ParamNonLinear(0, ax,  ay,  **p_aug)
    return mod_pw, param_pw, mod_aug, param_aug


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


def run_one(q_y, seed):
    mod_pw, param_pw, _mod_aug, param_aug = _build_params(q_y)
    dim_x_pw = mod_pw.dim_x  # = 1

    # ── EPKF (simulate trajectory) ───────────────────────────────────────────
    epkf = NonLinear_EPKF(param=param_pw, sKey=seed, verbose=0)
    xt_e, xh_e, pp_e, ii_e, ss_e = _run(epkf, N)
    err_e = compute_errors(epkf, xt_e, xh_e, pp_e, ii_e, ss_e)

    shared = [(s["k"], s["xkp1"], s["ykp1"]) for s in epkf.history._history]

    # ── UPKF ─────────────────────────────────────────────────────────────────
    upkf = NonLinear_UPKF(param=param_pw, sigmaSet=SIGMA_SET, sKey=seed, verbose=0)
    xt_u, xh_u, pp_u, ii_u, ss_u = _run(upkf, N, iter(shared))
    err_u = compute_errors(upkf, xt_u, xh_u, pp_u, ii_u, ss_u)

    # ── UKF-aug ──────────────────────────────────────────────────────────────
    ukf = NonLinear_UKF(param=param_aug, sigmaSet=SIGMA_SET, sKey=seed, verbose=0)
    xt_a, xh_a, pp_a, _ii_a, _ss_a = _run(ukf, N, iter(shared))
    # extract only x component (augmented state = [x, y])
    xh_x = [xh[:dim_x_pw] for xh in xh_a]
    pp_x = [p[:dim_x_pw, :dim_x_pw] for p in pp_a]
    err_a = compute_errors(ukf, xt_a, xh_x, pp_x)

    return (err_e["mse_total"], err_u["mse_total"], err_a["mse_total"],
            err_e["nees_mean"], err_u["nees_mean"], err_a["nees_mean"])


def main():
    print("\n" + "=" * 65)
    print("  Section 4.2 — Multiplicative noise: UPKF vs UKF-aug")
    print("=" * 65)

    results = {k: {"mse": [], "nees": []} for k in ("epkf", "upkf", "ukf_aug")}

    for q_y in SIGMA_Y_VALUES:
        sigma_y = np.sqrt(q_y)
        print(f"\n  sigma_y = {sigma_y:.3f}  (q_y = {q_y:.4f})")
        mse_e_list, mse_u_list, mse_a_list = [], [], []
        nees_e_list, nees_u_list, nees_a_list = [], [], []

        for seed in range(N_SEEDS):
            try:
                mse_e, mse_u, mse_a, nees_e, nees_u, nees_a = run_one(q_y, seed)
                mse_e_list.append(mse_e); mse_u_list.append(mse_u); mse_a_list.append(mse_a)
                nees_e_list.append(nees_e); nees_u_list.append(nees_u); nees_a_list.append(nees_a)
            except Exception as exc:
                print(f"    seed {seed}: FAILED ({exc})")

        def _stat(lst):
            a = np.array([v for v in lst if v is not None and np.isfinite(v)])
            return a.mean(), a.std()

        me, se = _stat(mse_e_list);  mu, su = _stat(mse_u_list);  ma, sa = _stat(mse_a_list)
        ne, se2 = _stat(nees_e_list); nu, su2 = _stat(nees_u_list); na, sa2 = _stat(nees_a_list)

        results["epkf"]["mse"].append((me, se))
        results["upkf"]["mse"].append((mu, su))
        results["ukf_aug"]["mse"].append((ma, sa))
        results["epkf"]["nees"].append((ne, se2))
        results["upkf"]["nees"].append((nu, su2))
        results["ukf_aug"]["nees"].append((na, sa2))

        print(f"    MSE  — EPKF: {me:.4f}±{se:.4f}  UPKF: {mu:.4f}±{su:.4f}  UKF-aug: {ma:.4f}±{sa:.4f}")
        print(f"    NEES — EPKF: {ne:.3f}±{se2:.3f}   UPKF: {nu:.3f}±{su2:.3f}   UKF-aug: {na:.3f}±{sa2:.3f}")

    # ── Print LaTeX table ──────────────────────────────────────────────────────
    sigma_y_vals = np.sqrt(SIGMA_Y_VALUES)
    print("\n" + "─" * 80)
    print("LaTeX table (MSE):")
    print(r"\begin{tabular}{cccc}")
    print(r"\toprule")
    print(r"$\sigma_{v^y}$ & EPKF & UPKF & UKF-aug \\")
    print(r"\midrule")
    for i, sv in enumerate(sigma_y_vals):
        me, _ = results["epkf"]["mse"][i]
        mu, _ = results["upkf"]["mse"][i]
        ma, _ = results["ukf_aug"]["mse"][i]
        print(f"{sv:.3f} & {me:.4f} & {mu:.4f} & {ma:.4f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")

    print("\nLaTeX table (NEES):")
    print(r"\begin{tabular}{cccc}")
    print(r"\toprule")
    print(r"$\sigma_{v^y}$ & EPKF & UPKF & UKF-aug \\")
    print(r"\midrule")
    for i, sv in enumerate(sigma_y_vals):
        ne, _ = results["epkf"]["nees"][i]
        nu, _ = results["upkf"]["nees"][i]
        na, _ = results["ukf_aug"]["nees"][i]
        print(f"{sv:.3f} & {ne:.3f} & {nu:.3f} & {na:.3f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")

    # ── Figures ───────────────────────────────────────────────────────────────
    sigma_y_vals = np.sqrt(SIGMA_Y_VALUES)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    colors = {"epkf": "tab:blue", "upkf": "tab:orange", "ukf_aug": "tab:green"}
    labels = {"epkf": "EPKF", "upkf": "UPKF", "ukf_aug": "UKF-aug"}

    for metric, ax, ylabel, title in [
        ("mse",  axes[0], "MSE",       "MSE vs observation noise"),
        ("nees", axes[1], "NEES mean", "NEES vs observation noise"),
    ]:
        for key in ("epkf", "upkf", "ukf_aug"):
            vals = np.array(results[key][metric])
            means, stds = vals[:, 0], vals[:, 1]
            ax.plot(sigma_y_vals, means, "o-", color=colors[key],
                    label=labels[key], linewidth=1.5)
            ax.fill_between(sigma_y_vals, means - stds, means + stds,
                            alpha=0.15, color=colors[key])
        if metric == "nees":
            ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, label="ideal (1.0)")
        ax.set_xlabel(r"$\sigma_{v^y}$", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Multiplicative observation noise: EPKF / UPKF / UKF-aug\n"
        f"({N_SEEDS} seeds × N={N} steps)",
        fontsize=10,
    )
    fig.tight_layout()
    out = FIGURES_DIR / "multip_mse_nees_vs_sigma.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved → {out.relative_to(REPO_ROOT)}")
    print("Done.")


if __name__ == "__main__":
    main()
