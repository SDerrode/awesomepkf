"""
Section 5 (alternative) — Real data experiment: S&P 500 Stochastic Volatility.

Data
----
  S&P 500 (^GSPC) daily OHLCV from Yahoo Finance (2000–2023).

Variables
---------
  x_n  = log(Parkinson_RV_n)      log-variance proxy (latent state)
           where Parkinson_RV_n = [log(H_n/L_n)]² / (4 ln 2)
  ỹ_n  = log(r_n²) − μ_w          de-meaned log-squared close return (observation)
           r_n = log(C_n / C_{n-1}),   μ_w = −(γ + ln 2) ≈ −1.2704

In the pairwise model, the filter observes ỹ_n and estimates x_n.
Both indices are jointly non-stationary in levels (vol clusters); we work
with the raw values and let the NN learn the dynamics.

Periods
-------
  Training : 2000-01-01 → 2015-12-31  (~4000 days)
  Test     : 2016-01-01 → 2023-12-31  (~2000 days)

Pipeline
--------
  1. Download S&P 500 OHLCV from Yahoo Finance (cached).
  2. Compute (x_n, ỹ_n); report contemporaneous correlation (compare with ENSO −0.617).
  3. Train NNModel on training pairs (z_n, z_{n+1}).
  4. Apply EPKF, UPKF, PPF on test period (filter only sees ỹ_n).
  5. Baselines: single-shot x̂_n = ỹ_n, persistence x̂_n = ỹ_{n−1}.
  6. Save figures and print metrics.

Usage
-----
  python3 -m prg.run_paper_section5_sv
"""

import math
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import yfinance as yf

# ── paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parent.parent
DATA_DIR    = REPO_ROOT / "data" / "datafile" / "realdata" / "sv_sp500"
FIGURES_DIR = REPO_ROOT / "papier_NonLinearPKF" / "figures"
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

RAW_CSV   = DATA_DIR / "sp500_ohlcv.csv"
TRAIN_CSV = DATA_DIR / "sv_train.csv"
TEST_CSV  = DATA_DIR / "sv_test.csv"

# ── Experiment parameters ─────────────────────────────────────────────────────
TICKER      = "^GSPC"
FULL_START  = "2000-01-01"
FULL_END    = "2023-12-31"
TRAIN_END   = "2015-12-31"
TEST_START  = "2016-01-01"

NN_HIDDEN   = (64, 64)
NN_EPOCHS   = 1500
NN_LR       = 1e-3
NN_SEED     = 42
SKEY        = 42
N_PARTICLES = 1000
SIGMA_SET   = "wan2000"
DPI         = 150

# log(χ²_1) mean correction: E[log(Z²)] where Z~N(0,1) = −(γ+ln2)
MU_W = -(np.euler_gamma + np.log(2))   # ≈ −1.2704

# ── Module imports ────────────────────────────────────────────────────────────
from prg.classes.nonlinear_epkf import NonLinear_EPKF
from prg.classes.nonlinear_ppf import NonLinear_PPF
from prg.classes.nonlinear_upkf import NonLinear_UPKF
from prg.classes.param_nonlinear import ParamNonLinear
from prg.utils.metrics import compute_errors
from prg.utils.nn_model import NNModel

# ==============================================================================
# 1. Data download and preparation
# ==============================================================================

def download_sp500():
    """Download S&P 500 OHLCV from Yahoo Finance and cache."""
    if RAW_CSV.exists():
        print(f"  Cache found: {RAW_CSV.relative_to(REPO_ROOT)}")
        return

    print(f"  Downloading {TICKER} ({FULL_START} → {FULL_END}) via yfinance …")
    df = yf.download(TICKER, start=FULL_START, end=FULL_END,
                     auto_adjust=True, progress=False)

    # Flatten multi-index columns (yfinance 0.2+)
    if isinstance(df.columns, tuple) or hasattr(df.columns, 'levels'):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    df = df[["Open", "High", "Low", "Close"]].dropna()
    df.to_csv(RAW_CSV)
    print(f"  Saved {len(df)} rows → {RAW_CSV.relative_to(REPO_ROOT)}")


def compute_sv_series(csv_path):
    """
    Load OHLCV CSV; compute (date, x_n, ytilde_n).

    x_n     = log(Parkinson_RV_n)  log-variance (proxy ground truth)
    ytilde_n = log(r_n²) − μ_w     de-meaned log-squared close return

    Returns list of (date_str, x, ytilde) tuples (NaN rows dropped).
    """
    # Deferred: pandas only used by this loader, not by the model code.
    import pandas as pd  # noqa: PLC0415

    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    # Flatten multi-index columns if present
    df.columns = [c.split(',')[0].strip("('\" ") if ',' in str(c) else str(c).strip()
                  for c in df.columns]

    # Ensure required columns exist
    for col in ["High", "Low", "Close"]:
        if col not in df.columns:
            # Try case-insensitive match
            matches = [c for c in df.columns if c.lower() == col.lower()]
            if matches:
                df.rename(columns={matches[0]: col}, inplace=True)
            else:
                raise KeyError(f"Column '{col}' not found. Available: {df.columns.tolist()}")

    df = df.dropna(subset=["High", "Low", "Close"])
    df = df[df["High"] > df["Low"]]   # guard against bad rows

    # Close-to-close log return
    df["log_ret"] = np.log(df["Close"] / df["Close"].shift(1))
    df = df.dropna(subset=["log_ret"])

    # Parkinson range-based variance
    df["park_rv"] = (np.log(df["High"] / df["Low"]))**2 / (4 * math.log(2))
    df = df[df["park_rv"] > 0]

    # State x_n = log(Parkinson_RV)
    df["x"] = np.log(df["park_rv"])

    # Observation ỹ_n = log(r_n²) − μ_w   (guard log(0) with clip)
    r2 = np.maximum(df["log_ret"]**2, 1e-30)
    df["ytilde"] = np.log(r2) - MU_W

    df = df.dropna(subset=["x", "ytilde"])
    df = df[np.isfinite(df["x"]) & np.isfinite(df["ytilde"])]

    return [(str(idx.date()), float(row["x"]), float(row["ytilde"]))
            for idx, row in df.iterrows()]


def load_and_split(rows):
    """Split into train/test; save NNModel-ready CSVs."""
    train = [(x, y) for d, x, y in rows if d <= TRAIN_END]
    test  = [(x, y) for d, x, y in rows if d >= TEST_START]
    dates_test = [d for d, x, y in rows if d >= TEST_START]

    train_arr = np.array(train)
    test_arr  = np.array(test)

    np.savetxt(TRAIN_CSV, train_arr, delimiter=",", header="X0,Y0", comments="")
    np.savetxt(TEST_CSV,  test_arr,  delimiter=",", header="X0,Y0", comments="")

    print(f"  Train: {len(train)} days (up to {TRAIN_END})  "
          f"|  Test: {len(test)} days ({TEST_START} → {dates_test[-1]})")

    # Contemporaneous correlation between x and ytilde
    xa = train_arr[:, 0]
    ya = train_arr[:, 1]
    corr = float(np.corrcoef(xa, ya)[0, 1])
    print(f"  Contemporaneous correlation x ↔ ỹ  (train): {corr:.4f}")
    print("  [ENSO had x ↔ y correlation ≈ −0.617; lower is better here]")

    return train_arr, test_arr, dates_test


# ==============================================================================
# 2. 3-D figure of learned dynamics
# ==============================================================================

def plot_nn_functions(model, train_data, out_path, n_grid=50):
    """Two side-by-side 3-D surfaces: g_x(x, ỹ) and g_ỹ(x, ỹ)."""
    x_min, x_max = train_data[:, 0].min(), train_data[:, 0].max()
    y_min, y_max = train_data[:, 1].min(), train_data[:, 1].max()
    mx = 0.05 * (x_max - x_min)
    my = 0.05 * (y_max - y_min)
    xg = np.linspace(x_min - mx, x_max + mx, n_grid)
    yg = np.linspace(y_min - my, y_max + my, n_grid)
    XG, YG = np.meshgrid(xg, yg)
    Z_grid = np.stack([XG.ravel(), YG.ravel()], axis=1)
    G_grid = model._forward_np(Z_grid)
    GX = G_grid[:, 0].reshape(n_grid, n_grid)
    GY = G_grid[:, 1].reshape(n_grid, n_grid)

    fig = plt.figure(figsize=(12, 5))
    panels = [
        (GX, r"$g_x(x,\tilde{y})$ — log-variance dynamics",
         r"$x_{k+1}$ (log-var)",   "viridis"),
        (GY, r"$g_{\tilde{y}}(x,\tilde{y})$ — log-ret² dynamics",
         r"$\tilde{y}_{k+1}$",      "plasma"),
    ]
    for idx, (G, title, zlabel, cmap) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, 2, idx, projection="3d")
        ax.plot_surface(XG, YG, G, cmap=cmap, alpha=0.85)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(r"$x_k$ (log-var)")
        ax.set_ylabel(r"$\tilde{y}_k$ (log-ret²)")
        ax.set_zlabel(zlabel)
        ax.view_init(30, -60)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {Path(out_path).relative_to(REPO_ROOT)}")


# ==============================================================================
# 3. Filtering helpers (same as section 5 ENSO)
# ==============================================================================

def _make_gen(test_data):
    def _gen():
        for k, (x, y) in enumerate(test_data):
            yield k, np.array([[x]]), np.array([[y]])
    return _gen


def _run_real_filter(filt, test_data):
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


def _plot_real_filter(x_true_list, x_hat_list, P_list, dates, title, out_path,
                      ylabel="Log-variance $x_n$"):
    n = len(x_hat_list)
    xt_arr  = np.array([v.ravel()[0] for v in x_true_list[:n]])
    xh_arr  = np.array([v.ravel()[0] for v in x_hat_list])
    sig_arr = np.array([np.sqrt(max(float(p.ravel()[0]), 0)) for p in P_list])

    # Deferred: only the SV plot needs date objects.
    import datetime  # noqa: PLC0415

    date_objs = [datetime.date.fromisoformat(d) for d in dates[:n]]

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(date_objs, xt_arr, label="Parkinson log-var (proxy truth)",
            color="tab:blue", linewidth=0.7)
    ax.plot(date_objs, xh_arr, label=r"$\hat{x}$",
            color="tab:orange", linewidth=0.7)
    ax.fill_between(date_objs,
                    xh_arr - 2 * sig_arr,
                    xh_arr + 2 * sig_arr,
                    alpha=0.2, color="tab:orange")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=7)
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {Path(out_path).relative_to(REPO_ROOT)}")


# ==============================================================================
# 4. Baselines
# ==============================================================================

def single_shot_baseline(test_data):
    """x̂_n = ỹ_n  (single-shot: use observation directly as estimate)."""
    x_true = [np.array([[row[0]]]) for row in test_data]
    x_hat  = [np.array([[row[1]]]) for row in test_data]
    P_hat  = [np.array([[1.0]])   for _ in test_data]   # placeholder
    return x_true, x_hat, P_hat


def persistence_baseline(test_data):
    """x̂_n = ỹ_{n-1}  (lagged-observation persistence)."""
    x_true, x_hat, P_hat = [], [], []
    for k, (x, y) in enumerate(test_data):
        x_true.append(np.array([[x]]))
        if k == 0:
            x_hat.append(np.array([[y]]))   # initialise at first obs
        else:
            x_hat.append(np.array([[test_data[k-1][1]]]))
        P_hat.append(np.array([[1.0]]))
    return x_true, x_hat, P_hat


def _compute_mse_mae(x_true_list, x_hat_list):
    err = np.array([xt.ravel()[0] - xh.ravel()[0]
                    for xt, xh in zip(x_true_list, x_hat_list, strict=False)])
    return float(np.mean(err**2)), float(np.mean(np.abs(err)))


# ==============================================================================
# Main
# ==============================================================================

def main():
    print("\n" + "=" * 65)
    print("  Section 5 (SV) — S&P 500 stochastic volatility")
    print("=" * 65)

    # ── 1. Data ───────────────────────────────────────────────────────────────
    print("\n[1] S&P 500 data")
    download_sp500()
    rows = compute_sv_series(RAW_CSV)
    train_data, test_data, dates_test = load_and_split(rows)

    # Convert to list of (x, ytilde) tuples for filtering
    test_list  = [(float(r[0]), float(r[1])) for r in test_data]
    _train_list = [(float(r[0]), float(r[1])) for r in train_data]

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
    print(f"  mQ estimated:\n{np.round(nn_model.mQ, 5)}")

    # ── 3. 3-D figure of learned dynamics ─────────────────────────────────────
    print("\n[3] Generating 3-D figure of learned dynamics …")
    plot_nn_functions(
        nn_model, train_data,
        str(FIGURES_DIR / "nn_gx_gy_sv.png"),
    )

    # ── 4. Build PKF param from NN model ─────────────────────────────────────
    p = nn_model.get_params().copy()
    dim_x, dim_y = p.pop("dim_x"), p.pop("dim_y")
    p["mz0"] = test_data[0].reshape(2, 1)   # init at first test point
    p["Pz0"] = nn_model.mQ.copy() * 5        # slightly diffuse

    param = ParamNonLinear(0, dim_x, dim_y, **p)

    # ── 5. Run filters ────────────────────────────────────────────────────────
    print("\n[4] Running filters …")

    print("    EPKF …")
    epkf = NonLinear_EPKF(param=param, sKey=SKEY, verbose=0)
    xt_e, xh_e, pp_e, ii_e, ss_e = _run_real_filter(epkf, test_list)
    err_e = compute_errors(epkf, xt_e, xh_e, pp_e, ii_e, ss_e)

    print("    UPKF …")
    upkf = NonLinear_UPKF(param=param, sigmaSet=SIGMA_SET, sKey=SKEY, verbose=0)
    xt_u, xh_u, pp_u, ii_u, ss_u = _run_real_filter(upkf, test_list)
    err_u = compute_errors(upkf, xt_u, xh_u, pp_u, ii_u, ss_u)

    print("    PPF  …")
    ppf = NonLinear_PPF(param=param, n_particles=N_PARTICLES, sKey=SKEY, verbose=0)
    xt_p, xh_p, pp_p = [], [], []
    for _k, xt, _yk, _xp, xu in ppf.process_filter(
            N=len(test_list), data_generator=_make_gen(test_list)()):
        step = ppf.history.last()
        if xt is not None:
            xt_p.append(xt)
        xh_p.append(xu)
        pp_p.append(step["PXXkp1_update"])
    err_p = compute_errors(ppf, xt_p, xh_p, pp_p, None, None)

    # ── 6. Baselines ──────────────────────────────────────────────────────────
    print("\n[5] Computing baselines …")
    xt_ss, xh_ss, _ = single_shot_baseline(test_list)
    mse_ss, mae_ss = _compute_mse_mae(xt_ss, xh_ss)
    print(f"    Single-shot (ỹ_n):   MSE={mse_ss:.4f}, MAE={mae_ss:.4f}")

    xt_ps, xh_ps, _ = persistence_baseline(test_list)
    mse_ps, mae_ps = _compute_mse_mae(xt_ps, xh_ps)
    print(f"    Persistence (ỹ_{{n-1}}): MSE={mse_ps:.4f}, MAE={mae_ps:.4f}")

    # Training-mean baseline
    mean_x_train = float(np.mean(train_data[:, 0]))
    x_true_arr = np.array([r[0] for r in test_list])
    mse_mean = float(np.mean((x_true_arr - mean_x_train)**2))
    mae_mean = float(np.mean(np.abs(x_true_arr - mean_x_train)))
    print(f"    Training mean (x̄_train={mean_x_train:.3f}): MSE={mse_mean:.4f}, MAE={mae_mean:.4f}")

    # ── 7. Metrics table ──────────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print(f"  Table — S&P 500 SV  (test {TEST_START} → {dates_test[-1]})")
    print("─" * 65)

    headers = ["Single-shot", "Persistence", "Clim.mean", "EPKF", "UPKF", "PPF"]
    mse_vals = [mse_ss, mse_ps, mse_mean,
                err_e["mse_total"], err_u["mse_total"], err_p["mse_total"]]
    mae_vals = [mae_ss, mae_ps, mae_mean,
                err_e["mae_total"], err_u["mae_total"], err_p["mae_total"]]
    nees_vals = [None, None, None,
                 err_e["nees_mean"], err_u["nees_mean"], err_p["nees_mean"]]
    nis_vals  = [None, None, None,
                 err_e["nis_mean"],  err_u["nis_mean"],  None]

    def _f(v): return f"{v:.4f}" if isinstance(v, float) else "—"
    def _fn(v): return f"{v:.4f}" if v is not None and isinstance(v, float) else "—"

    print(f"  {'':18s}  " + "  ".join(f"{h:>9}" for h in headers))
    print(f"  {'MSE':18s}  " + "  ".join(f"{_f(v):>9}" for v in mse_vals))
    print(f"  {'MAE':18s}  " + "  ".join(f"{_f(v):>9}" for v in mae_vals))
    print(f"  {'NEES mean':18s}  " + "  ".join(f"{_fn(v):>9}" for v in nees_vals))
    print(f"  {'NIS mean':18s}  " + "  ".join(f"{_fn(v):>9}" for v in nis_vals))

    print("\n  LaTeX table:")
    for label, vals in [
        (r"\textbf{MSE}",       mse_vals),
        (r"\textbf{MAE}",       mae_vals),
        (r"\textbf{NEES mean}", nees_vals),
        (r"\textbf{NIS mean}",  nis_vals),
    ]:
        row = " & ".join(f"{_fn(v):>8}" if v is not None else "    —   " for v in vals)
        print(f"  {label:22s} & {row} \\\\")

    # ── 8. Filtering figures ──────────────────────────────────────────────────
    print("\n[6] Generating filtering figures …")
    _plot_real_filter(xt_e, xh_e, pp_e, dates_test,
                      "EPKF — S&P 500 log-variance reconstruction (2016–2023)",
                      str(FIGURES_DIR / "epkf_sv.png"))
    _plot_real_filter(xt_u, xh_u, pp_u, dates_test,
                      "UPKF — S&P 500 log-variance reconstruction (2016–2023)",
                      str(FIGURES_DIR / "upkf_sv.png"))
    _plot_real_filter(xt_p, xh_p, pp_p, dates_test,
                      "PPF — S&P 500 log-variance reconstruction (2016–2023)",
                      str(FIGURES_DIR / "ppf_sv.png"))

    print("\nDone.")


if __name__ == "__main__":
    main()
