"""
Multi-seed sensitivity analysis for Section 4 of the paper
"Non-linear extensions to Gaussian pairwise Kalman filters".

Runs 30 independent random seeds and reports mean ± std of MSE
for EPKF, UPKF and PPF (addresses reviewer concern on result robustness).

Usage (from repo root):
    python3 -m prg.run_paper_section4_sensitivity
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prg.classes.NonLinear_EPKF import NonLinear_EPKF
from prg.classes.NonLinear_PPF import NonLinear_PPF
from prg.classes.NonLinear_UPKF import NonLinear_UPKF
from prg.classes.ParamNonLinear import ParamNonLinear
from prg.models.nonLinear import ModelFactoryNonLinear
from prg.utils.utils import compute_errors

N           = 1000
N_PARTICLES = 500
SIGMA_SET   = "wan2000"
SEEDS       = list(range(30))

def _build_param():
    model = ModelFactoryNonLinear.create("model_x1_y1_pairwise")
    p = model.get_params().copy()
    dim_x, dim_y = p.pop("dim_x"), p.pop("dim_y")
    return ParamNonLinear(0, dim_x, dim_y, **p)

def _run_filter(filt, N, data_gen=None):
    x_true_list, x_hat_list, P_list, i_list, S_list = [], [], [], [], []
    kwargs = {"N": N}
    if data_gen is not None:
        kwargs["data_generator"] = data_gen
    for _k, xt, _yk, _xp, xu in filt.process_filter(**kwargs):
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

mse_e, mse_u, mse_p = [], [], []

for seed in SEEDS:
    print(f"seed {seed:2d}...", flush=True)
    param = _build_param()

    # EPKF (simulate trajectory)
    epkf = NonLinear_EPKF(param=param, sKey=seed, verbose=0)
    xt_e, xh_e, pp_e, ii_e, ss_e = _run_filter(epkf, N)
    err_e = compute_errors(epkf, xt_e, xh_e, pp_e, ii_e, ss_e)
    mse_e.append(err_e["mse_total"])

    shared_data = [(step["k"], step["xkp1"], step["ykp1"])
                   for step in epkf.history._history]

    # UPKF on same trajectory
    upkf = NonLinear_UPKF(param=param, sigmaSet=SIGMA_SET, sKey=seed, verbose=0)
    xt_u, xh_u, pp_u, ii_u, ss_u = _run_filter(
        upkf, N, data_gen=iter(shared_data))
    err_u = compute_errors(upkf, xt_u, xh_u, pp_u, ii_u, ss_u)
    mse_u.append(err_u["mse_total"])

    # PPF on same trajectory
    ppf = NonLinear_PPF(param=param, n_particles=N_PARTICLES, sKey=seed, verbose=0)
    xt_p, xh_p, pp_p, _, _ = _run_filter(ppf, N, data_gen=iter(shared_data))
    err_p = compute_errors(ppf, xt_p, xh_p, pp_p, None, None)
    mse_p.append(err_p["mse_total"])

mse_e = np.array(mse_e)
mse_u = np.array(mse_u)
mse_p = np.array(mse_p)

print("\n=== Multi-seed sensitivity (30 seeds, N=1000) ===")
print(f"  EPKF MSE : {mse_e.mean():.4f} ± {mse_e.std():.4f}")
print(f"  UPKF MSE : {mse_u.mean():.4f} ± {mse_u.std():.4f}")
print(f"  PPF  MSE : {mse_p.mean():.4f} ± {mse_p.std():.4f}")
print(f"\n  EPKF vs UPKF diff (mean): {abs(mse_e.mean()-mse_u.mean()):.6f}")
print(f"  (relative to std): {abs(mse_e.mean()-mse_u.mean())/mse_e.std():.2f} sigma")
