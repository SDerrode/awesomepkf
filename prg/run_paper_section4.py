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

import time
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT    = Path(__file__).resolve().parent.parent
FIGURES_DIR  = REPO_ROOT / "papier_NonLinearPKF" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Experiment parameters ──────────────────────────────────────────────────────
N           = 1000   # number of time steps (as stated in the paper)
SKEY        = 42     # shared RNG seed → same simulated trajectory for all filters
N_PARTICLES = 500    # PPF particle count (Section 4)
SIGMA_SET   = "wan2000"
WINDOW      = {"xmin": 0, "xmax": N}
DPI         = 150

# ── Imports ────────────────────────────────────────────────────────────────────
from prg.classes.nonlinear_epkf import NonLinear_EPKF
from prg.classes.nonlinear_ppf import NonLinear_PPF
from prg.classes.nonlinear_ukf import NonLinear_UKF
from prg.classes.nonlinear_upkf import NonLinear_UPKF
from prg.classes.param_linear import ParamLinear
from prg.classes.param_nonlinear import ParamNonLinear
from prg.models.linear import ModelFactoryLinear
from prg.models.nonlinear import ModelFactoryNonLinear
from prg.utils.metrics import compute_errors

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
    for _k, xt, _yk, _xp, xu in filt.process_filter(N=N):
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
    return compute_errors(filt, x_true_list, x_hat_list, P_list, i_arr, S_list)


def _plot_filter(history, title, params, labels, covars, out_path):
    """Produce a single-panel time-series figure and save it."""
    df = history.as_dataframe()

    fig, ax = plt.subplots(figsize=(8, 3))
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for param_key, label, covar_key, col in zip(params, labels, covars, colors, strict=False):
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
    print(f"  Saved → {Path(out_path).relative_to(REPO_ROOT)}")


# ==============================================================================
# Main
# ==============================================================================

def _run_epkf_first(param_pw):
    """Run EPKF and capture the simulated trajectory to share with UPKF/PPF."""
    print("Running EPKF …")
    epkf = NonLinear_EPKF(param=param_pw, sKey=SKEY, verbose=0)
    t0 = time.perf_counter()
    x_true_e, x_hat_e, P_e, i_e, S_e = _run_filter(epkf, N)
    elapsed = time.perf_counter() - t0
    err_e = _compute_metrics(epkf, x_true_e, x_hat_e, P_e, i_e, S_e)

    shared_data = [
        (step["k"], step["xkp1"], step["ykp1"]) for step in epkf.history._history
    ]
    return epkf, err_e, elapsed, shared_data


def _run_upkf_on_shared(param_pw, shared_data):
    """Re-run UPKF on the trajectory simulated by EPKF."""
    print("Running UPKF …")
    upkf = NonLinear_UPKF(param=param_pw, sigmaSet=SIGMA_SET, sKey=SKEY, verbose=0)

    xt, xh, pp, ii, ss = [], [], [], [], []
    t0 = time.perf_counter()
    for _k, xt_k, _yk, _xp, xu in upkf.process_filter(
        N=N, data_generator=iter(shared_data)
    ):
        step = upkf.history.last()
        if xt_k is not None:
            xt.append(xt_k)
        xh.append(xu)
        pp.append(step["PXXkp1_update"])
        if step["ikp1"] is not None:
            ii.append(step["ikp1"].ravel())
        if step["Skp1"] is not None:
            ss.append(step["Skp1"])
    elapsed = time.perf_counter() - t0
    i_arr = np.array(ii) if ii else None
    err = _compute_metrics(upkf, xt, xh, pp, i_arr, ss if ss else None)
    return upkf, err, elapsed


def _run_ppf_on_shared(param_pw, shared_data):
    """Run PPF on the shared EPKF trajectory."""
    print("Running PPF  …")
    ppf = NonLinear_PPF(param=param_pw, n_particles=N_PARTICLES, sKey=SKEY, verbose=0)
    xt, xh, pp = [], [], []
    t0 = time.perf_counter()
    for _k, xt_k, _yk, _xp, xu in ppf.process_filter(
        N=N, data_generator=iter(shared_data)
    ):
        step = ppf.history.last()
        if xt_k is not None:
            xt.append(xt_k)
        xh.append(xu)
        pp.append(step["PXXkp1_update"])
    elapsed = time.perf_counter() - t0
    # NIS is not meaningful for PPF (no analytic S matrix) → omit.
    err = _compute_metrics(ppf, xt, xh, pp, None, None)
    return ppf, err, elapsed


def _print_table1(err_e, err_u, err_p, t_epkf, t_upkf, t_ppf):
    """Print Table 1 (MSE / MAE / NEES / NIS) and timing block."""
    print("\n" + "─" * 55)
    print("  Table 1 — EPKF / UPKF / PPF")
    print("─" * 55)
    print(f"{'':20s}  {'EPKF':>8}  {'UPKF':>8}  {'PPF':>8}")

    def _fmt(v):
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    for key, label in [
        ("mse_total",  "MSE "),
        ("mae_total",  "MAE "),
        ("nees_mean",  "NEES mean"),
        ("nis_mean",   "NIS mean"),
    ]:
        ve, vu, vp = err_e[key], err_u[key], err_p[key]
        print(f"  {label:18s}  {_fmt(ve):>8}  {_fmt(vu):>8}  {_fmt(vp):>8}")

    print(f"\n  CPU time (N={N} steps):")
    print(f"    EPKF : {t_epkf*1e3:.1f} ms  ({t_epkf/N*1e3:.3f} ms/step)")
    print(f"    UPKF : {t_upkf*1e3:.1f} ms  ({t_upkf/N*1e3:.3f} ms/step)")
    print(f"    PPF  : {t_ppf*1e3:.1f} ms  ({t_ppf/N*1e3:.3f} ms/step)  [{N_PARTICLES} particles]")

    print("\nLaTeX Table 1:")

    def _fmtl(v):
        return f"{v:.4f}" if isinstance(v, float) else "na"

    for key, label in [
        ("mse_total",  r"\textbf{MSE}"),
        ("mae_total",  r"\textbf{MAE}"),
        ("nees_mean",  r"\textbf{NEES mean}"),
        ("nis_mean",   r"\textbf{NIS mean}"),
    ]:
        ve, vu, vp = err_e[key], err_u[key], err_p[key]
        print(f"  {label:30s} & {_fmtl(ve):8} & {_fmtl(vu):8} & {_fmtl(vp):8} \\\\")


def _run_aug_filter(filter_cls, param_aug, dim_x_pw, shared_data, **kwargs):
    """Generic helper for the EKF-aug / UKF-aug runs (Table 2)."""
    flt = filter_cls(param=param_aug, sKey=SKEY, verbose=0, **kwargs)
    xt, xh, pp = [], [], []
    for _k, xt_k, _yk, _xp, xu in flt.process_filter(
        N=N, data_generator=iter(shared_data)
    ):
        step = flt.history.last()
        if xt_k is not None:
            xt.append(xt_k)
        xh.append(xu)
        pp.append(step["PXXkp1_update"])
    # Augmented state is [x, y] — only the x component matters for MSE/MAE.
    xh_x = [v[:dim_x_pw] for v in xh]
    pp_x = [p[:dim_x_pw, :dim_x_pw] for p in pp]
    return compute_errors(flt, xt, xh_x, pp_x)


def _print_table2(param_pw, param_aug, shared_data):
    """Run EKF-aug / UKF-aug and print Table 2."""
    print("\n" + "─" * 55)
    print("  Table 2 — Augmented EKF / UKF")
    print("─" * 55)

    try:
        dim_x_pw = param_pw.dim_x  # = 1; augmented state is [x, y]
        err_ekf_aug = _run_aug_filter(
            NonLinear_EPKF, param_aug, dim_x_pw, shared_data
        )
        err_ukf_aug = _run_aug_filter(
            NonLinear_UKF, param_aug, dim_x_pw, shared_data, sigmaSet=SIGMA_SET
        )

        print(f"{'':20s}  {'EKF-aug':>8}  {'UKF-aug':>8}")
        for key, label in [("mse_total", "MSE "), ("mae_total", "MAE ")]:
            print(f"  {label:18s}  {err_ekf_aug[key]:.4f}    {err_ukf_aug[key]:.4f}")

        print("\nLaTeX Table 2:")
        for key, label in [
            ("mse_total", r"\textbf{MSE}"),
            ("mae_total", r"\textbf{MAE}"),
        ]:
            print(
                f"  {label:30s} & {err_ekf_aug[key]:.4f}   "
                f"& {err_ukf_aug[key]:.4f}   \\\\"
            )
    except Exception as exc:
        print(f"  [WARN] Augmented model failed: {exc}")


def _generate_figures(epkf, upkf, ppf):
    """Produce the 4 PNGs that go into the paper."""
    print("\nGenerating figures …")
    _plot_filter(
        epkf.history,
        "Simulated observations",
        ["ykp1"], ["Observation y"], [None],
        str(FIGURES_DIR / "epkf_observations_x1_y1_Retroactions.png"),
    )
    _plot_filter(
        epkf.history,
        "EPKF filtering",
        ["xkp1", "Xkp1_update"],
        ["x true", "x̂ EPKF"],
        [None, "PXXkp1_update"],
        str(FIGURES_DIR / "epkf_x1_y1_Retroactions.png"),
    )
    _plot_filter(
        upkf.history,
        "UPKF filtering",
        ["xkp1", "Xkp1_update"],
        ["x true", "x̂ UPKF"],
        [None, "PXXkp1_update"],
        str(FIGURES_DIR / "upkf_x1_y1_Retroactions.png"),
    )
    _plot_filter(
        ppf.history,
        "PPF filtering",
        ["xkp1", "Xkp1_update"],
        ["x true", "x̂ PPF"],
        [None, "PXXkp1_update"],
        str(FIGURES_DIR / "ppf_x1_y1_Retroactions.png"),
    )


def main():
    print("\n" + "=" * 65)
    print("  Section 4 — Pairwise model : model_x1_y1_pairwise")
    print("=" * 65)

    model_pw, param_pw = _build_param("model_x1_y1_pairwise")
    _model_aug, param_aug = _build_param("model_x1_y1_augmented")

    print(f"\nQ (pairwise model) =\n{np.round(param_pw.mQ, 4)}")
    print(f"\nLatex Q:\n{model_pw.latex_model()}\n")

    epkf, err_e, t_epkf, shared_data = _run_epkf_first(param_pw)
    upkf, err_u, t_upkf = _run_upkf_on_shared(param_pw, shared_data)
    ppf,  err_p, t_ppf  = _run_ppf_on_shared(param_pw, shared_data)

    _print_table1(err_e, err_u, err_p, t_epkf, t_upkf, t_ppf)
    _print_table2(param_pw, param_aug, shared_data)
    _generate_figures(epkf, upkf, ppf)

    print("\nDone.")


if __name__ == "__main__":
    main()
