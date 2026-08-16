#!/usr/bin/env python3
"""
Vector-data validity of the back-action LRT (Remark~3, Sec. IV-C).

Monte-Carlo SIZE check of Lambda = 2[ell(A^xy free) - ell(A^xy=0)] for VECTOR data
(pq > 1), using the paper's own library estimator
(prg.learning.em_partial_dynamics.back_action_lrt). Null models set A^xy = 0, freeze
A^xx, A^yx, Q at truth, and keep A^yy as a free nuisance.

The dof count is full when (A^xx, A^yx) is OBSERVABLE -- rank O = p, where
O = [A^yx; A^yx A^xx; ...] -- NOT when q >= p; dimension counting is the wrong criterion
(see Prop. 1 / Remark 3 of the paper). Both couples below are observable; they differ in
how WELL conditioned that observability is.

Cases:
  * x2y2 (p=2, q=2, pq=4): O well conditioned (singular values 0.407/0.148, ratio 2.7),
    so Lambda tracks chi2_4 (mean ~ 3.8, var ~ 7.3, size ~ 0.04, KS p ~ 0.35).
  * x2y1 (p=2, q=1, pq=2): O is full rank but nearly singular (0.406/0.0074, ratio 55).
    At finite N the weak direction contributes almost nothing, so the statistic behaves
    as though the count were deficient: mean ~ 1 rather than pq = 2, size ~ 0.01 --
    conservative (correct level, reduced power).

Run from the awesomepkf repo root with its .venv, or set AWESOMEPKF_ROOT.
    python experiments/lrt_vector.py [both|x2y1|x2y2]
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import chi2, kstest


def _locate_repo_root() -> Path:
    env_root = os.environ.get("AWESOMEPKF_ROOT")
    if env_root:
        p = Path(env_root).expanduser().resolve()
        if (p / "prg").is_dir():
            return p
    here = Path(__file__).resolve()
    for ancestor in [here.parent, *here.parents]:
        if (ancestor / "prg" / "classes" / "linear_pks.py").exists():
            return ancestor
        if (ancestor / "awesomePKF" / "prg" / "classes" / "linear_pks.py").exists():
            return ancestor / "awesomePKF"
    raise RuntimeError("Cannot locate awesomePKF repo root (set AWESOMEPKF_ROOT).")


REPO_ROOT = _locate_repo_root()
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from prg.classes.linear_pkf import Linear_PKF
from prg.classes.param_linear import ParamLinear
from prg.learning.em_partial_dynamics import back_action_lrt
from prg.models.linear._amq import LinearAmQ

OUT = Path(__file__).resolve().parent


def make_param(A, Q, dim_x, dim_y):
    dz = dim_x + dim_y
    A = np.asarray(A, float)
    Q = np.asarray(Q, float)
    model = LinearAmQ(dim_x, dim_y, A=A, mQ=0.5 * (Q + Q.T) + 1e-9 * np.eye(dz),
                      mz0=np.zeros((dz, 1)), Pz0=np.eye(dz), pairwiseModel=True)
    kw = model.get_params().copy()
    kw.pop("dim_x")
    kw.pop("dim_y")
    return ParamLinear(0, dim_x, dim_y, **kw)


def run_case(A, Q, dim_x, dim_y, label, N=400, nseed=300):
    A = np.asarray(A, float)
    sr = np.max(np.abs(np.linalg.eigvals(A)))
    param = make_param(A, Q, dim_x, dim_y)       # A^xy = 0 (null model)
    dof = dim_x * dim_y
    lams = []
    fails = 0
    for s in range(nseed):
        try:
            data = Linear_PKF(param, sKey=s).simulate_N_data(N)
            res = back_action_lrt(param, data)
            lams.append(float(res.stat))
        except Exception:
            fails += 1
        if (s + 1) % 30 == 0:
            print(f"  {label}: {s + 1}/{nseed} (fails={fails})", flush=True)
    lams = np.array(lams)
    qc = chi2.ppf(0.95, dof)
    size = float(np.mean(lams > qc))
    ks = kstest(lams, "chi2", args=(dof,))
    print(f"\n=== {label}: p={dim_x}, q={dim_y}, pq(dof)={dof} ===")
    print(f"  spectral radius A = {sr:.3f} (stable if <1); N={N}, seeds={len(lams)} (fails={fails})")
    print(f"  mean Lambda = {lams.mean():.3f}  (theory chi2 mean = {dof})")
    print(f"  var  Lambda = {lams.var():.3f}  (theory chi2 var  = {2 * dof})")
    print(f"  empirical size @ alpha=0.05 = {size:.3f}  (crit chi2_{dof},0.95 = {qc:.3f})")
    print(f"  KS vs chi2_{dof}: D={ks.statistic:.3f}, p={ks.pvalue:.3f}  (p>0.05 = tracks)")
    return {"label": label, "dim_x": dim_x, "dim_y": dim_y, "dof": dof, "sr": float(sr),
                "N": N, "nseed": len(lams), "fails": fails,
                "mean": float(lams.mean()), "var": float(lams.var()), "size": size,
                "crit": float(qc), "ks_D": float(ks.statistic), "ks_p": float(ks.pvalue)}


A_X2Y1 = [[0.50, 0.10, 0.00],
          [0.00, 0.40, 0.00],
          [0.30, 0.20, 0.40]]
Q_X2Y1 = [[0.10, 0.02, 0.01],
          [0.02, 0.10, 0.01],
          [0.01, 0.01, 0.10]]
A_X2Y2 = [[0.50, 0.10, 0.00, 0.00],
          [0.00, 0.40, 0.00, 0.00],
          [0.30, 0.10, 0.40, 0.10],
          [0.10, 0.20, 0.00, 0.30]]
Q_X2Y2 = [[0.10, 0.02, 0.01, 0.00],
          [0.02, 0.10, 0.00, 0.01],
          [0.01, 0.00, 0.10, 0.02],
          [0.00, 0.01, 0.02, 0.10]]


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    results = []
    if which in ("x2y1", "both"):
        results.append(run_case(A_X2Y1, Q_X2Y1, 2, 1, "x2y1", N=400, nseed=150))
    if which in ("x2y2", "both"):
        results.append(run_case(A_X2Y2, Q_X2Y2, 2, 2, "x2y2", N=250, nseed=100))
    with (OUT / f"lrt_vector_{which}.json").open("w") as fh:
        json.dump(results, fh, indent=1)
    print(f"\nSAVED {OUT / f'lrt_vector_{which}.json'}", flush=True)
