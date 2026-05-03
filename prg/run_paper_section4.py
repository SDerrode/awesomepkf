#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reproduce all experiments from Section 4 of the paper
"Non-linear Gaussian pairwise Kalman filters".

Generates:
  - papier_NonLinearPKF/figures/epkf_observations_x1_y1_Retroactions.png
  - papier_NonLinearPKF/figures/epkf_x1_y1_Retroactions.png
  - papier_NonLinearPKF/figures/upkf_x1_y1_Retroactions.png
  - papier_NonLinearPKF/figures/ppf_x1_y1_Retroactions.png

And prints:
  - Table 1 : MSE / MAE / NEES / NIS for EPKF, UPKF, PPF
  - Table 2 : MSE / MAE for EKF-aug, UKF-aug
  - The actual Q matrix used in the model

Usage (from repo root):
    python3 -m prg.run_paper_section4
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR  = os.path.join(REPO_ROOT, "papier_NonLinearPKF", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Experiment parameters ──────────────────────────────────────────────────────
N           = 1000   # number of time steps (as stated in the paper)
SKEY        = 42     # shared RNG seed → same simulated trajectory for all filters
N_PARTICLES = 500    # PPF particle count (Section 4)
SIGMA_SET   = "wan2000"
WINDOW      = {"xmin": 0, "xmax": N}
DPI         = 150

# ── Imports ────────────────────────────────────────────────────────────────────
from prg.models.nonLinear import ModelFactoryNonLinear
from prg.models.linear    import ModelFactoryLinear
from prg.classes.ParamNonLinear import ParamNonLinear
from prg.classes.ParamLinear    import ParamLinear
from prg.classes.NonLinear_EPKF import NonLinear_EPKF
from prg.classes.NonLinear_UPKF import NonLinear_UPKF
from prg.classes.NonLinear_PPF  import NonLinear_PPF
from prg.utils.utils import compute_errors


# ==============================================================================
# Helpers
# ==============================================================================

def _build_param(model_name: str):
    """Instantiate model + ParamNonLinear (or ParamLinear for augmented)."""
    if model_name in ModelFactoryNonLinear.list_models():
        model = ModelFactoryNonLinear.create(model_name)
        p = model.get_params().copy()
        dim_x, dim_y = p.pop("dim_x"), p.pop("dim_y")
        param = ParamNonLinear(0, dim_x, dim_y, **p)
    elif model_name in ModelFactoryLinear.list_models():
        model = ModelFactoryLinear.create(model_name)
        p = model.get_params().copy()
        dim_x, dim_y = p.pop("dim_x"), p.pop("dim_y")
        param = ParamLinear(0, dim_x, dim_y, **p)
    else:
        raise ValueError(f"Unknown model: {model_name!r}")
    return model, param


def _run_filter(filt, N):
    """Run a filter for N steps; return (x_true, x_hat, P_xx, innov, S)."""
    x_true_list, x_hat_list, P_list, i_list, S_list = [], [], [], [], []
    for k, xt, yk, xp, xu in filt.process_filter(N=N):
        step = filt.history.last()
        if xt is not None:
            x_true_list.append(xt)
        x_hat_list.append(xu)
        P_list.append(step["PXXkp1_update"])
        if step["ikp1"] is not None:
            i_list.append(step["ikp1"].ravel())
        if step["Skp1"] is not None:
            S_list.append(step["Skp1"])
    i_arr = np.array(i_list) if i_list else None   # (N, dim_y)
    return x_true_list, x_hat_list, P_list, i_arr, S_list if S_list else None


def _compute_metrics(filt, x_true_list, x_hat_list, P_list, i_arr, S_list):
    err = compute_errors(filt, x_true_list, x_hat_list, P_list, i_arr, S_list)
    return err


def _plot_filter(history, title, params, labels, covars, out_path):
    """Produce a single-panel time-series figure and save it."""
    df = history.as_dataframe()

    fig, ax = plt.subplots(figsize=(8, 3))
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for param_key, label, covar_key, col in zip(params, labels, covars, colors):
        series = np.array([v.ravel()[0] for v in df[param_key]])
        ax.plot(series, label=label, color=col, linewidth=0.8)
        if covar_key is not None and covar_key in df.columns:
            sigma = np.array([np.sqrt(max(float(v.ravel()[0]), 0))
                              for v in df[covar_key]])
            ax.fill_between(range(len(series)),
                            series - 2 * sigma, series + 2 * sigma,
                            alpha=0.2, color=col)
    ax.set_xlabel("Time step")
    ax.legend(fontsize=7)
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {os.path.relpath(out_path, REPO_ROOT)}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    print("\n" + "=" * 65)
    print("  Section 4 — Pairwise model : model_x1_y1_pairwise")
    print("=" * 65)

    model_name     = "model_x1_y1_pairwise"
    model_name_aug = "model_x1_y1_augmented"

    # ── Build models ───────────────────────────────────────────────────────────
    model_pw, param_pw   = _build_param(model_name)
    model_aug, param_aug = _build_param(model_name_aug)

    # Show Q
    print(f"\nQ (pairwise model) =\n{np.round(param_pw.mQ, 4)}")
    print(f"\nLatex Q:\n{model_pw.latex_model()}\n")

    # ── Simulate ONCE with EPKF to get the shared trajectory ──────────────────
    print("Running EPKF …")
    epkf = NonLinear_EPKF(param=param_pw, sKey=SKEY, verbose=0)
    _t0 = time.perf_counter()
    x_true_e, x_hat_e, P_e, i_e, S_e = _run_filter(epkf, N)
    t_epkf = time.perf_counter() - _t0
    err_e = _compute_metrics(epkf, x_true_e, x_hat_e, P_e, i_e, S_e)

    # Save the shared simulated data as a list of (k, x, y) for UPKF and PPF
    shared_data = [
        (step["k"], step["xkp1"], step["ykp1"])
        for step in epkf.history._history
    ]

    def _shared_generator():
        for item in shared_data:
            yield item

    # ── UPKF on same data ──────────────────────────────────────────────────────
    print("Running UPKF …")
    upkf = NonLinear_UPKF(param=param_pw, sigmaSet=SIGMA_SET, sKey=SKEY, verbose=0)
    x_true_u, x_hat_u, P_u, i_u, S_u = _run_filter(
        upkf, N
    )
    # Re-run on shared trajectory
    upkf2 = NonLinear_UPKF(param=param_pw, sigmaSet=SIGMA_SET, sKey=SKEY, verbose=0)
    xt2, xh2, pp2, ii2, ss2 = [], [], [], [], []
    _t0 = time.perf_counter()
    for k, xt, yk, xp, xu in upkf2.process_filter(N=N, data_generator=_shared_generator()):
        step = upkf2.history.last()
        if xt is not None:
            xt2.append(xt)
        xh2.append(xu)
        pp2.append(step["PXXkp1_update"])
        if step["ikp1"] is not None:
            ii2.append(step["ikp1"].ravel())
        if step["Skp1"] is not None:
            ss2.append(step["Skp1"])
    t_upkf = time.perf_counter() - _t0
    i_arr2 = np.array(ii2) if ii2 else None
    err_u = _compute_metrics(upkf2, xt2, xh2, pp2, i_arr2, ss2 if ss2 else None)

    # ── PPF on same data ───────────────────────────────────────────────────────
    print("Running PPF  …")
    ppf = NonLinear_PPF(param=param_pw, n_particles=N_PARTICLES, sKey=SKEY, verbose=0)
    xt3, xh3, pp3 = [], [], []
    _t0 = time.perf_counter()
    for k, xt, yk, xp, xu in ppf.process_filter(N=N, data_generator=_shared_generator()):
        step = ppf.history.last()
        if xt is not None:
            xt3.append(xt)
        xh3.append(xu)
        pp3.append(step["PXXkp1_update"])
    t_ppf = time.perf_counter() - _t0
    # NIS is not meaningful for PPF (no analytic S matrix) → omit
    err_p = _compute_metrics(ppf, xt3, xh3, pp3, None, None)

    # ── Table 1 ────────────────────────────────────────────────────────────────
    print("\n" + "─" * 55)
    print("  Table 1 — EPKF / UPKF / PPF")
    print("─" * 55)
    header = f"{'':20s}  {'EPKF':>8}  {'UPKF':>8}  {'PPF':>8}"
    print(header)
    for key, label in [
        ("mse_total",  "MSE "),
        ("mae_total",  "MAE "),
        ("nees_mean",  "NEES mean"),
        ("nis_mean",   "NIS mean"),
    ]:
        ve = err_e[key]; vu = err_u[key]; vp = err_p[key]
        def _fmt(v):
            return f"{v:.4f}" if isinstance(v, float) else str(v)
        print(f"  {label:18s}  {_fmt(ve):>8}  {_fmt(vu):>8}  {_fmt(vp):>8}")

    # ── Timing ────────────────────────────────────────────────────────────────
    print(f"\n  CPU time (N={N} steps):")
    print(f"    EPKF : {t_epkf*1e3:.1f} ms  ({t_epkf/N*1e3:.3f} ms/step)")
    print(f"    UPKF : {t_upkf*1e3:.1f} ms  ({t_upkf/N*1e3:.3f} ms/step)")
    print(f"    PPF  : {t_ppf*1e3:.1f} ms  ({t_ppf/N*1e3:.3f} ms/step)  [{N_PARTICLES} particles]")

    # ── LaTeX snippet for Table 1 ──────────────────────────────────────────────
    print("\nLaTeX Table 1:")
    for key, label in [
        ("mse_total",  r"\textbf{MSE}"),
        ("mae_total",  r"\textbf{MAE}"),
        ("nees_mean",  r"\textbf{NEES mean}"),
        ("nis_mean",   r"\textbf{NIS mean}"),
    ]:
        ve = err_e[key]; vu = err_u[key]; vp = err_p[key]
        def _fmtl(v):
            return f"{v:.4f}" if isinstance(v, float) else "na"
        print(f"  {label:30s} & {_fmtl(ve):8} & {_fmtl(vu):8} & {_fmtl(vp):8} \\\\")

    # ── Augmented EKF / UKF (Table 2) ─────────────────────────────────────────
    print("\n" + "─" * 55)
    print("  Table 2 — Augmented EKF / UKF")
    print("─" * 55)

    try:
        from prg.classes.NonLinear_EPKF import NonLinear_EPKF as _EPKF
        from prg.classes.NonLinear_UKF  import NonLinear_UKF  as _UKF

        dim_x_pw = param_pw.dim_x  # = 1; augmented state is [x, y]

        ekf_aug = _EPKF(param=param_aug, sKey=SKEY, verbose=0)
        xta, xha, ppa = [], [], []
        for k, xt, yk, xp, xu in ekf_aug.process_filter(N=N, data_generator=_shared_generator()):
            step = ekf_aug.history.last()
            if xt is not None:
                xta.append(xt)
            xha.append(xu)
            ppa.append(step["PXXkp1_update"])
        # Extract X component only (augmented state = [x, y]; x_true has shape (dim_x_pw,1))
        xha_x = [xh[:dim_x_pw] for xh in xha]
        ppa_x = [p[:dim_x_pw, :dim_x_pw] for p in ppa]
        err_ekf_aug = compute_errors(ekf_aug, xta, xha_x, ppa_x)

        ukf_aug = _UKF(param=param_aug, sigmaSet=SIGMA_SET, sKey=SKEY, verbose=0)
        xtu, xhu, ppu = [], [], []
        for k, xt, yk, xp, xu in ukf_aug.process_filter(N=N, data_generator=_shared_generator()):
            step = ukf_aug.history.last()
            if xt is not None:
                xtu.append(xt)
            xhu.append(xu)
            ppu.append(step["PXXkp1_update"])
        xhu_x = [xh[:dim_x_pw] for xh in xhu]
        ppu_x = [p[:dim_x_pw, :dim_x_pw] for p in ppu]
        err_ukf_aug = compute_errors(ukf_aug, xtu, xhu_x, ppu_x)

        header2 = f"{'':20s}  {'EKF-aug':>8}  {'UKF-aug':>8}"
        print(header2)
        for key, label in [("mse_total", "MSE "), ("mae_total", "MAE ")]:
            ve2 = err_ekf_aug[key]; vu2 = err_ukf_aug[key]
            print(f"  {label:18s}  {ve2:.4f}    {vu2:.4f}")

        print("\nLaTeX Table 2:")
        for key, label in [
            ("mse_total", r"\textbf{MSE}"),
            ("mae_total", r"\textbf{MAE}"),
        ]:
            ve2 = err_ekf_aug[key]; vu2 = err_ukf_aug[key]
            print(f"  {label:30s} & {ve2:.4f}   & {vu2:.4f}   \\\\")

    except Exception as exc:
        print(f"  [WARN] Augmented model failed: {exc}")

    # ── Figures ────────────────────────────────────────────────────────────────
    print("\nGenerating figures …")

    _plot_filter(
        epkf.history,
        "Simulated observations",
        ["ykp1"], ["Observation y"], [None],
        os.path.join(FIGURES_DIR, "epkf_observations_x1_y1_Retroactions.png"),
    )

    _plot_filter(
        epkf.history,
        "EPKF filtering",
        ["xkp1", "Xkp1_update"],
        ["x true", "x̂ EPKF"],
        [None, "PXXkp1_update"],
        os.path.join(FIGURES_DIR, "epkf_x1_y1_Retroactions.png"),
    )

    _plot_filter(
        upkf2.history,
        "UPKF filtering",
        ["xkp1", "Xkp1_update"],
        ["x true", "x̂ UPKF"],
        [None, "PXXkp1_update"],
        os.path.join(FIGURES_DIR, "upkf_x1_y1_Retroactions.png"),
    )

    _plot_filter(
        ppf.history,
        "PPF filtering",
        ["xkp1", "Xkp1_update"],
        ["x true", "x̂ PPF"],
        [None, "PXXkp1_update"],
        os.path.join(FIGURES_DIR, "ppf_x1_y1_Retroactions.png"),
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
