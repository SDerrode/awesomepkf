#!/usr/bin/env python3
"""
Vector-data validity of the back-action LRT (Remark~3, Sec. IV-C).

Monte-Carlo SIZE check of Lambda = 2[ell(A^xy free) - ell(A^xy=0)] for VECTOR data
(pq > 1). Null models set A^xy = 0, freeze A^xx, A^yx, Q at truth, and keep A^yy as a free
nuisance. Lambda is computed twice on every record:
  * "exact": both y-log-likelihoods (pairwise Kalman filter, prior z_0 ~ N(0, I)) maximized
    directly by BFGS, the free fit started from the null optimum and from the EM end point;
    this is the statistic reported in the paper;
  * "em": the library estimator prg.learning.em_partial_dynamics.back_action_lrt with
    polish=False (partial EM only, max_iter=100, relative plateau tolerance 1e-6; since
    awesomepkf 2.16.6 its default polishes the EM end points into the exact statistic), kept
    as a diagnostic: its early stop leaves the free fit short of the maximum along weakly
    observable directions and biases Lambda downward.

The dof count is full when (A^xx, A^yx) is OBSERVABLE -- rank O = p, where
O = [A^yx; A^yx A^xx; ...] -- NOT when q >= p; dimension counting is the wrong criterion
(see Prop. 1 / Remark 3 of the paper). Both couples below are observable; they differ in
how WELL conditioned that observability is.

Cases:
  * x2y2 (p=2, q=2, pq=4): O well conditioned (singular values 0.407/0.148, ratio 2.7),
    so Lambda tracks chi2_4 (mean ~ 3.8, var ~ 7.3, size ~ 0.04, KS p ~ 0.35).
  * x2y1 (p=2, q=1, pq=2): O is full rank but nearly singular (0.406/0.0074, ratio 55).
    The exact statistic still tracks chi2_2 (mean ~ 2.0, size ~ 0.05, KS p ~ 0.95); the
    EM statistic does not (mean ~ 1, size ~ 0.01): EM stops after ~20 iterations, before
    reaching the maximum along the weak direction.

Run from the awesomepkf repo root with its .venv, or set AWESOMEPKF_ROOT.
    python experiments/lrt_vector.py [both|x2y1|x2y2]
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
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
from prg.learning.em_partial_dynamics import back_action_lrt, estimate_dynamics_em
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


def y_loglik(Y, A, Q, p, q):
    """Exact log-likelihood of y_{0:N} under the pairwise model (A, Q), z_0 ~ N(0, I)."""
    d = p + q
    m, P, ll = np.zeros(d), np.eye(d), 0.0
    for n, y in enumerate(Y):
        if n > 0:
            m, P = A @ m, A @ P @ A.T + Q
        c = np.linalg.cholesky(P[p:, p:])
        w = np.linalg.solve(c, y - m[p:])
        ll -= 0.5 * (q * np.log(2 * np.pi) + 2 * np.log(np.diag(c)).sum() + w @ w)
        K = P[:, p:] @ np.linalg.inv(P[p:, p:])
        m, P = m + K @ (y - m[p:]), P - K @ P[p:, :]
        m[p:], P[p:, :], P[:, p:] = y, 0.0, 0.0
    return ll


def exact_lrt(A0, Q, p, q, Y, start_free):
    """Lambda with both fits maximized directly (A^xx, A^yx, Q held at A0, Q)."""
    def build(axy, ayy):
        A = A0.copy()
        A[:p, p:], A[p:, p:] = axy, ayy
        return A

    def best(f, starts):
        rs = []
        for x0 in starts:
            with np.errstate(all="ignore"):
                rs.append(minimize(f, x0, method="BFGS", options={"gtol": 1e-7}))
        return min(rs, key=lambda r: r.fun)

    f0 = lambda t: -y_loglik(Y, build(np.zeros((p, q)), t.reshape(q, q)), Q, p, q)
    r0 = best(f0, [A0[p:, p:].ravel()])
    f1 = lambda t: -y_loglik(Y, build(t[:p * q].reshape(p, q), t[p * q:].reshape(q, q)), Q, p, q)
    r1 = best(f1, [np.concatenate([np.zeros(p * q), r0.x]), start_free])
    return max(2.0 * (r0.fun - r1.fun), 0.0)


def summarize(lams, dof):
    lams = np.asarray(lams)
    qc = chi2.ppf(0.95, dof)
    ks = kstest(lams, "chi2", args=(dof,))
    return {"mean": float(lams.mean()), "var": float(lams.var()),
            "size": float(np.mean(lams > qc)), "crit": float(qc),
            "ks_D": float(ks.statistic), "ks_p": float(ks.pvalue)}


def run_case(A, Q, dim_x, dim_y, label, N=400, nseed=300):
    A = np.asarray(A, float)
    Qm = 0.5 * (np.asarray(Q, float) + np.asarray(Q, float).T) + 1e-9 * np.eye(dim_x + dim_y)
    sr = np.max(np.abs(np.linalg.eigvals(A)))
    param = make_param(A, Q, dim_x, dim_y)       # A^xy = 0 (null model)
    dof = dim_x * dim_y
    lam_em, lam_ex, iters = [], [], []
    fails = 0
    for s in range(nseed):
        try:
            data = Linear_PKF(param, sKey=s).simulate_N_data(N)
            Y = np.array([np.asarray(r[2], float).reshape(dim_y) for r in data])
            lam_em.append(float(back_action_lrt(param, data, polish=False).stat))
            em = estimate_dynamics_em(param, data, learn_back_action=True, learn_obs_memory=True)
            iters.append(int(em.n_iter))
            start = np.concatenate([np.ravel(em.A_xy), np.ravel(em.A_yy)])
            lam_ex.append(exact_lrt(A, Qm, dim_x, dim_y, Y, start))
        except Exception:
            fails += 1
        if (s + 1) % 30 == 0:
            print(f"  {label}: {s + 1}/{nseed} (fails={fails})", flush=True)
    ex, emr = summarize(lam_ex, dof), summarize(lam_em, dof)
    print(f"\n=== {label}: p={dim_x}, q={dim_y}, pq(dof)={dof} ===")
    print(f"  spectral radius A = {sr:.3f}; N={N}, records={len(lam_ex)} (fails={fails})")
    for name, r in (("exact", ex), ("EM   ", emr)):
        print(f"  {name}: mean {r['mean']:.3f} (chi2: {dof})  var {r['var']:.3f} (chi2: {2 * dof})"
              f"  size {r['size']:.3f}  KS p {r['ks_p']:.3f}")
    print(f"  EM iterations: median {int(np.median(iters))}, max {max(iters)}")
    return {"label": label, "dim_x": dim_x, "dim_y": dim_y, "dof": dof, "sr": float(sr),
            "N": N, "nseed": len(lam_ex), "fails": fails, **ex, "em": emr,
            "em_iterations_median": float(np.median(iters)),
            "lambda_exact": lam_ex, "lambda_em": lam_em}

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
