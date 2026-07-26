"""Identifiability of (A_xy, A_yy) under freezing: the observability condition (Sec. IV-A).

Freeze A_xx, A_yx, Q at truth and refit (A_xy, A_yy) to the exact y-marginal spectral
density from 120 random restarts; count how many DISTINCT exact fits exist. One means the
pair is globally identified, more means it is not.

Four configurations show the condition is observability of (A_xx, A_yx) -- rank p of
[A_yx; A_yx A_xx; ...] -- and not merely A_yx != 0: the third and fourth couples share the
same nonzero A_yx and differ only in that rank. All four models are stationary
(spectral radius < 1), without which the spectral density below is undefined.

numpy + scipy only; no awesomepkf dependency.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

W = np.linspace(0, np.pi, 80)


def run(tag, p, q, Axx, Ayx, Axy0, Ayy0, seed=11, starts=120):
    d = p + q
    rng = np.random.default_rng(seed)
    Q = np.eye(d) * 0.1 + 0.02
    obs = np.vstack([Ayx @ np.linalg.matrix_power(Axx, k) for k in range(p)])
    rank_obs = np.linalg.matrix_rank(obs)

    def A_of(axy, ayy):
        return np.block([[Axx, axy.reshape(p, q)], [Ayx, ayy.reshape(q, q)]])

    def Syy(A):
        out = []
        for w in W:
            M = np.linalg.inv(np.eye(d) - A * np.exp(-1j * w))
            out.append((M @ Q @ M.conj().T)[p:, p:].ravel())
        return np.concatenate(out)

    tgt = Syy(A_of(Axy0, Ayy0))
    scale = np.abs(tgt).max()

    def res(v):
        r = Syy(A_of(v[: p * q], v[p * q :])) - tgt
        return np.concatenate([r.real, r.imag])

    truth = np.concatenate([Axy0.ravel(), Ayy0.ravel()])
    sols = []
    for _ in range(starts):
        v0 = truth + rng.normal(0, 0.35, p * q + q * q)
        if np.abs(np.linalg.eigvals(A_of(v0[: p * q], v0[p * q :]))).max() >= 0.99:
            continue                      # non-stationary: the spectral density is undefined
        try:
            fit = least_squares(res, v0, xtol=1e-14, ftol=1e-14, gtol=1e-14)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if np.abs(res(fit.x)).max() < 1e-9 * scale:
            sols.append(fit.x)

    uniq = []
    for s in sols:
        if not any(np.abs(s - u).max() < 1e-4 for u in uniq):
            uniq.append(s)
    spread = max((np.abs(u[: p * q] - truth[: p * q]).max() for u in uniq), default=0.0)
    print(f"{tag:<44} obs rank {rank_obs}/{p}  exact fits {len(sols):>3}  "
          f"distinct {len(uniq):>3}  max A^xy spread {spread:.3f}")


run("p=q=1, Ayx!=0        (observable)",1,1,np.array([[.6]]),np.array([[.3]]),
    np.array([[.35]]),np.array([[.4]]))
run("p=q=2, Ayx inversible (observable)",2,2,np.array([[.6,.1],[0,.5]]),np.array([[.3,0],[.1,.2]]),
    np.array([[.35,.05],[.1,.2]]),np.array([[.4,.1],[0,.3]]))
run("p=2,q=1, Ayx!=0 mais NON observable",2,1,np.diag([.6,.5]),np.array([[.3,0.]]),
    np.array([[.2],[.15]]),np.array([[.4]]))
run("p=2,q=1, Ayx!=0 et observable",2,1,np.array([[.6,.2],[.1,.5]]),np.array([[.3,0.]]),
    np.array([[.2],[.15]]),np.array([[.4]]))
