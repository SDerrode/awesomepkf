"""
Reproduce all experiments from Section 5 of the paper
"Non-linear Gaussian pairwise Kalman filters".

Pipeline:
  1. Download ENSO data (Niño 3.4 SST + SOI) from NOAA — once only.
  2. Train a NNModel on 1951-2005.
  3. Apply EPKF, UPKF, PPF on 2006-2025 test period.
  4. Save figures and print metrics tables.

Generates:
  - papier_NonLinearPKF/figures/nn_gx_gy_enso.png
  - papier_NonLinearPKF/figures/epkf_enso.png
  - papier_NonLinearPKF/figures/upkf_enso.png
  - papier_NonLinearPKF/figures/ppf_enso.png

Usage (from repo root):
    python3 -m prg.run_paper_section5
"""

import re
import urllib.request
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parent.parent
DATA_DIR    = REPO_ROOT / "data" / "datafile" / "realdata" / "enso"
FIGURES_DIR = REPO_ROOT / "papier_NonLinearPKF" / "figures"
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

ENSO_CSV  = DATA_DIR / "enso_nino34_soi.csv"
TRAIN_CSV = DATA_DIR / "enso_train.csv"
TEST_CSV  = DATA_DIR / "enso_test.csv"

# ── Experiment parameters ─────────────────────────────────────────────────────
TRAIN_END   = 2005     # last training year (inclusive)
TEST_START  = 2006     # first test year (inclusive)
NN_HIDDEN   = (64, 64)
NN_EPOCHS   = 1500
NN_LR       = 1e-3
NN_SEED     = 42
SKEY        = 42
N_PARTICLES = 1000
SIGMA_SET   = "wan2000"
DPI         = 150

# ── NOAA data sources ─────────────────────────────────────────────────────────
# ERSSTv5 Niño indices (monthly, back to 1950): same format as sstoi.indices
NINO_URL = "https://www.cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.91-20.ascii"
# Standardised SOI (monthly, back to 1951)
SOI_URL  = "https://www.cpc.ncep.noaa.gov/data/indices/soi"

# ── Module imports ────────────────────────────────────────────────────────────
from prg.classes.nonlinear_epkf import NonLinear_EPKF
from prg.classes.nonlinear_ppf import NonLinear_PPF
from prg.classes.nonlinear_upkf import NonLinear_UPKF
from prg.classes.param_nonlinear import ParamNonLinear
from prg.utils.nn_model import NNModel
from prg.utils.utils import compute_errors

# ==============================================================================
# 1. Data download and preparation
# ==============================================================================

def _parse_nino34(text):
    """Parse ERSSTv5 nino index text → dict {(year, month): nino34_sst}."""
    result = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or not line[0].isdigit():
            continue
        parts = line.split()
        if len(parts) < 9:
            continue
        try:
            yr, mo = int(parts[0]), int(parts[1])
            nino34 = float(parts[8])   # column 8: NINO3.4 raw SST (not anomaly)
            if nino34 < -90:           # missing value flag
                continue
            result[(yr, mo)] = nino34
        except (ValueError, IndexError):
            continue
    return result


def _parse_soi(text):
    """Parse SOI text (first section = standardised SOI) → dict {(year, month): soi}.

    The file has two sections; only the first is used.  Negative values can
    be concatenated without a space (e.g. ``1.2-999.9``), so regex is used.
    """
    result = {}
    section = 0
    MISSING_FLAG = -900.0

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("YEAR"):
            section += 1
            continue
        if section != 1:          # skip everything outside first section
            continue
        # extract all numbers, handling concatenated negatives
        nums = re.findall(r'[-+]?\d+\.?\d*', stripped)
        if not nums:
            continue
        try:
            yr = int(nums[0])
        except ValueError:
            continue
        for mo_idx, val_str in enumerate(nums[1:13], start=1):
            try:
                val = float(val_str)
            except ValueError:
                continue
            if val < MISSING_FLAG:   # -999.9 missing flag
                continue
            result[(yr, mo_idx)] = val
    return result


def download_enso_data():
    """Download Niño 3.4 + SOI from NOAA and cache locally (one-time)."""
    if ENSO_CSV.exists():
        print(f"  Data cache found: {ENSO_CSV.relative_to(REPO_ROOT)}")
        return

    print("  Downloading Niño 3.4 SST (ERSSTv5) from NOAA …")
    with urllib.request.urlopen(NINO_URL, timeout=30) as r:
        nino_text = r.read().decode("utf-8", errors="replace")

    print("  Downloading SOI from NOAA …")
    with urllib.request.urlopen(SOI_URL, timeout=30) as r:
        soi_text = r.read().decode("utf-8", errors="replace")

    nino34 = _parse_nino34(nino_text)
    soi    = _parse_soi(soi_text)

    common_keys = sorted(set(nino34) & set(soi))
    rows = [(yr, mo, nino34[(yr, mo)], soi[(yr, mo)]) for yr, mo in common_keys]

    with ENSO_CSV.open("w") as f:
        f.write("year,month,X0,Y0\n")
        for yr, mo, x, y in rows:
            f.write(f"{yr},{mo},{x:.4f},{y:.4f}\n")
    print(f"  Saved {len(rows)} months "
          f"({rows[0][0]}-{rows[-1][0]}) → {ENSO_CSV.relative_to(REPO_ROOT)}")


def load_and_split():
    """Load full ENSO CSV, split into train/test, save NNModel-ready CSVs."""
    all_rows = []
    with ENSO_CSV.open() as f:
        next(f)   # skip header
        for line in f:
            yr, mo, x, y = line.strip().split(",")
            all_rows.append((int(yr), int(mo), float(x), float(y)))

    train_rows = [(x, y) for yr, mo, x, y in all_rows if yr <= TRAIN_END]
    test_rows  = [(x, y) for yr, mo, x, y in all_rows if yr >= TEST_START]
    test_dates = [(yr, mo) for yr, mo, x, y in all_rows if yr >= TEST_START]

    train = np.array(train_rows)
    test  = np.array(test_rows)

    for path, arr in [(TRAIN_CSV, train), (TEST_CSV, test)]:
        np.savetxt(path, arr, delimiter=",", header="X0,Y0", comments="")

    print(f"  Train: {len(train)} months ({all_rows[0][0]}–{TRAIN_END})  "
          f"|  Test: {len(test)} months ({TEST_START}–{test_dates[-1][0]})")
    return train, test, test_dates


# ==============================================================================
# 2. 3D figure of learned dynamics
# ==============================================================================

def plot_nn_functions(model, train_data, out_path, n_grid=50):
    """Two side-by-side 3D surfaces: g_x(x,y) and g_y(x,y)."""
    x_min, x_max = train_data[:, 0].min(), train_data[:, 0].max()
    y_min, y_max = train_data[:, 1].min(), train_data[:, 1].max()
    mx = 0.1 * (x_max - x_min)
    my = 0.1 * (y_max - y_min)
    xg = np.linspace(x_min - mx, x_max + mx, n_grid)
    yg = np.linspace(y_min - my, y_max + my, n_grid)
    XG, YG = np.meshgrid(xg, yg)
    Z_grid = np.stack([XG.ravel(), YG.ravel()], axis=1)
    G_grid = model._forward_np(Z_grid)
    GX = G_grid[:, 0].reshape(n_grid, n_grid)
    GY = G_grid[:, 1].reshape(n_grid, n_grid)

    fig = plt.figure(figsize=(12, 5))
    panels = [
        (GX, r"$g_x(x,y)$ — Niño 3.4 dynamics", r"$x_{k+1}$ (Niño 3.4 SST)", "viridis"),
        (GY, r"$g_y(x,y)$ — SOI dynamics",       r"$y_{k+1}$ (SOI)",           "plasma"),
    ]
    for idx, (G, title, zlabel, cmap) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, 2, idx, projection="3d")
        ax.plot_surface(XG, YG, G, cmap=cmap, alpha=0.85)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(r"$x_k$ (Niño 3.4)")
        ax.set_ylabel(r"$y_k$ (SOI)")
        ax.set_zlabel(zlabel)
        ax.view_init(30, -60)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {Path(out_path).relative_to(REPO_ROOT)}")


# ==============================================================================
# 3. Filtering helpers
# ==============================================================================

def _make_gen(test_data):
    """Return a generator factory that yields (k, x_true, y_obs) per step."""
    def _gen():
        for k, (x, y) in enumerate(test_data):
            yield k, np.array([[x]]), np.array([[y]])
    return _gen


def _run_real_filter(filt, test_data):
    """Run a filter on real test data; return (x_true, x_hat, P, innov, S)."""
    gen = _make_gen(test_data)
    x_true_list, x_hat_list, P_list, i_list, S_list = [], [], [], [], []
    for _k, xt, _yk, _xp, xu in filt.process_filter(
            N=len(test_data), data_generator=gen()):
        step = filt.history.last()
        if xt is not None:
            x_true_list.append(xt)
        x_hat_list.append(xu)
        P_list.append(step["PXXkp1_update"])
        if step["ikp1"] is not None:
            i_list.append(step["ikp1"].ravel())
        if step["Skp1"] is not None:
            S_list.append(step["Skp1"])
    i_arr = np.array(i_list) if i_list else None
    return x_true_list, x_hat_list, P_list, i_arr, S_list if S_list else None


def _plot_real_filter(x_true_list, x_hat_list, P_list, test_dates, title, out_path):
    """Time-series plot: true Niño 3.4, filter estimate, ±2σ band."""
    n = len(x_hat_list)
    xt_arr  = np.array([v.ravel()[0] for v in x_true_list[:n]])
    xh_arr  = np.array([v.ravel()[0] for v in x_hat_list])
    sig_arr = np.array([np.sqrt(max(float(p.ravel()[0]), 0)) for p in P_list])
    dates_n = test_dates[:n]

    tick_pos = [i for i, (yr, mo) in enumerate(dates_n) if mo == 1 and yr % 5 == 0]
    tick_lbl = [str(dates_n[i][0]) for i in tick_pos]

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(xt_arr, label="Niño 3.4 (true)", color="tab:blue",   linewidth=0.8)
    ax.plot(xh_arr, label=r"$\hat{x}$",       color="tab:orange", linewidth=0.8)
    ax.fill_between(range(n),
                    xh_arr - 2 * sig_arr,
                    xh_arr + 2 * sig_arr,
                    alpha=0.2, color="tab:orange")
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lbl, fontsize=7)
    ax.set_xlabel("Year")
    ax.set_ylabel("Niño 3.4 SST (°C)")
    ax.legend(fontsize=7)
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {Path(out_path).relative_to(REPO_ROOT)}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    print("\n" + "=" * 65)
    print("  Section 5 — Real data experiment: ENSO (Niño 3.4 + SOI)")
    print("=" * 65)

    # ── 1. Data ───────────────────────────────────────────────────────────────
    print("\n[1] ENSO data")
    download_enso_data()
    train_data, test_data, test_dates = load_and_split()

    # ── 2. Train NNModel ──────────────────────────────────────────────────────
    print(f"\n[2] Training NNModel  (epochs={NN_EPOCHS}, hidden={NN_HIDDEN}) …")
    nn_model = NNModel(
        csv_path=TRAIN_CSV,
        dim_x=1, dim_y=1,
        hidden_sizes=NN_HIDDEN,
        epochs=NN_EPOCHS,
        lr=NN_LR,
        seed=NN_SEED,
        verbose=300,
    )
    print(f"  mQ estimated:\n{np.round(nn_model.mQ, 4)}")

    # ── 3. 3D figure of learned dynamics ─────────────────────────────────────
    print("\n[3] Generating 3D figure of learned dynamics …")
    plot_nn_functions(
        nn_model, train_data,
        str(FIGURES_DIR / "nn_gx_gy_enso.png"),
    )

    # ── 4. Build PKF param from NN model ─────────────────────────────────────
    p = nn_model.get_params().copy()
    dim_x, dim_y = p.pop("dim_x"), p.pop("dim_y")
    p["mz0"] = test_data[0].reshape(2, 1)   # initialise at first test point
    p["Pz0"] = nn_model.mQ.copy() * 5        # slightly diffuse
    param = ParamNonLinear(0, dim_x, dim_y, **p)

    # ── 5–7. Run filters ──────────────────────────────────────────────────────
    print("\n[4] Running filters …")

    print("    EPKF …")
    epkf = NonLinear_EPKF(param=param, sKey=SKEY, verbose=0)
    xt_e, xh_e, pp_e, ii_e, ss_e = _run_real_filter(epkf, test_data)
    err_e = compute_errors(epkf, xt_e, xh_e, pp_e, ii_e, ss_e)

    print("    UPKF …")
    upkf = NonLinear_UPKF(param=param, sigmaSet=SIGMA_SET, sKey=SKEY, verbose=0)
    xt_u, xh_u, pp_u, ii_u, ss_u = _run_real_filter(upkf, test_data)
    err_u = compute_errors(upkf, xt_u, xh_u, pp_u, ii_u, ss_u)

    print("    PPF  …")
    ppf = NonLinear_PPF(param=param, n_particles=N_PARTICLES, sKey=SKEY, verbose=0)
    xt_p, xh_p, pp_p = [], [], []
    for _k, xt, _yk, _xp, xu in ppf.process_filter(
            N=len(test_data), data_generator=_make_gen(test_data)()):
        step = ppf.history.last()
        if xt is not None:
            xt_p.append(xt)
        xh_p.append(xu)
        pp_p.append(step["PXXkp1_update"])
    err_p = compute_errors(ppf, xt_p, xh_p, pp_p, None, None)

    # ── 8. Metrics table ──────────────────────────────────────────────────────
    print("\n" + "─" * 55)
    print(f"  Table — EPKF / UPKF / PPF  (test {TEST_START}–{test_dates[-1][0]})")
    print("─" * 55)
    print(f"{'':20s}  {'EPKF':>8}  {'UPKF':>8}  {'PPF':>8}")
    for key, label in [
        ("mse_total", "MSE"),
        ("mae_total", "MAE"),
        ("nees_mean", "NEES mean"),
        ("nis_mean",  "NIS mean"),
    ]:
        ve, vu, vp = err_e[key], err_u[key], err_p[key]
        def _fmt(v):
            return f"{v:.4f}" if isinstance(v, float) else str(v)
        print(f"  {label:18s}  {_fmt(ve):>8}  {_fmt(vu):>8}  {_fmt(vp):>8}")

    print("\nLaTeX table:")
    for key, label in [
        ("mse_total", r"\textbf{MSE}"),
        ("mae_total", r"\textbf{MAE}"),
        ("nees_mean", r"\textbf{NEES mean}"),
        ("nis_mean",  r"\textbf{NIS mean}"),
    ]:
        ve, vu, vp = err_e[key], err_u[key], err_p[key]
        def _fmtl(v):
            return f"{v:.4f}" if isinstance(v, float) else "na"
        print(f"  {label:30s} & {_fmtl(ve):8} & {_fmtl(vu):8} & {_fmtl(vp):8} \\\\")

    # ── 9. Filtering figures ──────────────────────────────────────────────────
    print("\n[5] Generating filtering figures …")
    _plot_real_filter(
        xt_e, xh_e, pp_e, test_dates,
        "EPKF — Niño 3.4 reconstruction (2006–2025)",
        str(FIGURES_DIR / "epkf_enso.png"),
    )
    _plot_real_filter(
        xt_u, xh_u, pp_u, test_dates,
        "UPKF — Niño 3.4 reconstruction (2006–2025)",
        str(FIGURES_DIR / "upkf_enso.png"),
    )
    _plot_real_filter(
        xt_p, xh_p, pp_p, test_dates,
        "PPF — Niño 3.4 reconstruction (2006–2025)",
        str(FIGURES_DIR / "ppf_enso.png"),
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
