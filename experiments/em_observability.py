#!/usr/bin/env python3
"""Observability, not A^yx != 0, is what makes back-action recoverable (Sec. IV-A).

Two p=2, q=1 couples sharing the SAME nonzero A^yx = [0.3, 0] and differing only in
whether (A^xx, A^yx) is observable, i.e. whether [A^yx; A^yx A^xx] has rank p=2:

  observable   A^xx = [[.6,.2],[.1,.5]]   -> rank 2
  unobservable A^xx = diag(.6,.5)          -> rank 1  (second latent coordinate never
                                                       reaches y, at any lag)

Both are stationary. Same partial-EM protocol as em_identification.py: freeze
A^xx, A^yx, Q; start from the classical init A^xy = A^yy = 0; E-step is the VAR
smoother, M-step the closed-form regression on the observed y_n.

Usage:  python em_observability.py [--N 2000] [--iters 100] [--seeds 50]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


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
from prg.classes.linear_pks import Linear_PKS_VAR
from prg.classes.param_linear import ParamLinear
from prg.models.linear._amq import LinearAmQ

DX, DY, DZ = 2, 1, 3
AYX = np.array([[0.3, 0.0]])                      # identical in both couples
AXY_T = np.array([[0.20], [0.15]])
AYY_T = np.array([[0.40]])
QT = 0.10 * np.eye(DZ) + 0.05 * (np.ones((DZ, DZ)) - np.eye(DZ))
CASES = {
    "observable":   np.array([[0.6, 0.2], [0.1, 0.5]]),
    "unobservable": np.diag([0.6, 0.5]),
}


def _A(axx, axy, ayy):
    return np.block([[axx, axy.reshape(DX, DY)], [AYX, ayy.reshape(DY, DY)]])


def _param(A):
    m = LinearAmQ(DX, DY, A=A, mQ=0.5 * (QT + QT.T) + 1e-9 * np.eye(DZ),
                  mz0=np.zeros((DZ, 1)), Pz0=np.eye(DZ), pairwiseModel=True)
    p = m.get_params().copy()
    p.pop("dim_x")
    p.pop("dim_y")
    return ParamLinear(0, DX, DY, **p)


def _obs_rank(axx):
    return np.linalg.matrix_rank(np.vstack([AYX @ np.linalg.matrix_power(axx, k)
                                            for k in range(DX)]))


def _em(axx, data, iters, init=None):
    """Partial EM for (A^xy, A^yy). `init` = (axy, ayy), default the classical (0, 0).

    Unidentifiability makes the y-marginal likelihood FLAT along a direction, not the
    estimates noisy: from a fixed init EM lands on the same point for every record, so
    seed spread cannot detect it. Varying the INIT is what exposes the flat direction.
    """
    y = np.array([np.asarray(yy, float).reshape(DY) for _, _, yy in data])
    y0, y1 = y[:-1], y[1:]
    Syy = y0.T @ y0
    axy, ayy = (np.zeros((DX, DY)), np.zeros((DY, DY))) if init is None else init
    for _ in range(iters):
        sm = Linear_PKS_VAR(_param(_A(axx, axy, ayy)))
        sm.process_N_data_smoother(N=len(data) - 1, data_generator=iter(data))
        xh = np.array([np.asarray(h["Xkp1_smooth"], float).reshape(DX) for h in sm.history])
        x0, x1 = xh[:-1], xh[1:]
        axy = np.linalg.solve(Syy.T, ((x1 - x0 @ axx.T).T @ y0).T).T
        ayy = np.linalg.solve(Syy.T, ((y1 - x0 @ AYX.T).T @ y0).T).T
    return axy.reshape(DX), ayy.reshape(DY)


def main(Ns=(2000, 8000), iters=60, inits=6):
    """Init-spread of the EM fixed point, as a function of record length.

    If the pair is identified, the y-marginal likelihood has a unique maximum and the
    spread over random inits is a finite-sample effect: it must shrink as N grows.
    If a direction is exactly flat, the spread is structural and does NOT shrink.
    Sweeping N is what tells the two apart.
    """
    rng = np.random.default_rng(0)
    print(f"{'case':<13}{'N':>7}{'rank':>6}   "
          + "".join(f"{k:>16}" for k in ("ptp A^xy_1", "ptp A^xy_2", "ptp A^yy")))
    for name, axx in CASES.items():
        for N in Ns:
            data = Linear_PKF(_param(_A(axx, AXY_T, AYY_T)), sKey=0).simulate_N_data(N)
            ends = np.array([np.concatenate(_em(axx, data, iters,
                             (rng.normal(0, .3, (DX, DY)), rng.normal(0, .3, (DY, DY)))))
                             for _ in range(inits)])
            ptp = np.ptp(ends, axis=0)
            print(f"{name:<13}{N:>7}{_obs_rank(axx):>4}/{DX}   "
                  + "".join(f"{v:>16.4f}" for v in ptp), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ns", type=int, nargs="+", default=[2000, 8000])
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--inits", type=int, default=6)
    a = ap.parse_args()
    main(tuple(a.Ns), a.iters, a.inits)
