"""Robustness of the classical-vs-couple gap across dimension and SNR (SPL App.).

Repeats the strong-coupling (rho=1) comparison of ``classical_vs_couple.py`` on
several pairwise models -- varying the latent/observed dimensions (p,q) and the
observation-noise level -- to show the couple advantage is not an artefact of the
single scalar model of Sec. III. For each model we report, against ground truth and
averaged over seeds: the smoothed-state MSE penalty of the best-fit classical model
(``refit``) and of the naive ablation (``ablated``) relative to the couple smoother,
and the couple/refit NEES (calibrated = 1, normalised by dim X).

Run (with ``prg`` importable):  python experiments/classical_vs_couple_multi.py
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_discrete_lyapunov

from prg.classes.linear_pkf import Linear_PKF
from prg.classes.linear_pks import Linear_PKS_VAR
from prg.classes.param_linear import ParamLinear
from prg.models.linear._amq import LinearAmQ

inv = np.linalg.inv

# Base pairwise model families (block matrices; back-action A^{xy} and noise
# cross-block S are scaled by rho, the observation noise Q^{yy} by qscale).
MODELS = {
    "x1y1": dict(dx=1, dy=1,
                 Axx=[[0.6]], Ayx=[[0.3]], Ayy=[[0.4]], Axy=[[0.4]],
                 Qxx=[[0.10]], Qyy=[[0.10]], S=[[0.05]]),
    "x2y2": dict(dx=2, dy=2,
                 Axx=[[0.5, 0.1], [0.0, 0.45]], Ayx=[[0.3, 0.0], [0.1, 0.2]],
                 Ayy=[[0.35, 0.05], [0.0, 0.3]], Axy=[[0.35, 0.1], [0.05, 0.3]],
                 Qxx=[[0.10, 0.0], [0.0, 0.10]], Qyy=[[0.10, 0.0], [0.0, 0.10]],
                 S=[[0.04, 0.0], [0.0, 0.04]]),
    "x2y1": dict(dx=2, dy=1,
                 Axx=[[0.5, 0.1], [0.0, 0.45]], Ayx=[[0.3, 0.1]], Ayy=[[0.4]],
                 Axy=[[0.4], [0.2]], Qxx=[[0.10, 0.0], [0.0, 0.10]],
                 Qyy=[[0.10]], S=[[0.05], [0.03]]),
}


def _stabilize(A: np.ndarray, cap: float = 0.95) -> np.ndarray:
    sr = np.max(np.abs(np.linalg.eigvals(A)))
    return A * (cap / sr) if sr > cap else A


def _pd(Q: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    Q = 0.5 * (Q + Q.T)
    lo = float(np.min(np.linalg.eigvalsh(Q)))
    return Q + (eps - lo) * np.eye(Q.shape[0]) if lo < eps else Q


def _make(cfg: dict, rho: float, qscale: float):
    dx, dy = cfg["dx"], cfg["dy"]
    Axx, Ayx = np.array(cfg["Axx"]), np.array(cfg["Ayx"])
    Ayy, Axy = np.array(cfg["Ayy"]), np.array(cfg["Axy"])
    A = _stabilize(np.block([[Axx, rho * Axy], [Ayx, Ayy]]))
    Qxx, Qyy = np.array(cfg["Qxx"]), qscale * np.array(cfg["Qyy"])
    S = rho * np.array(cfg["S"])
    Q = _pd(np.block([[Qxx, S], [S.T, Qyy]]))
    return A, Q, dx, dy


def _param(A: np.ndarray, Q: np.ndarray, dx: int, dy: int) -> ParamLinear:
    dz = dx + dy
    m = LinearAmQ(dx, dy, A=A, mQ=_pd(Q), mz0=np.zeros((dz, 1)),
                  Pz0=np.eye(dz), pairwiseModel=True)
    p = m.get_params().copy()
    p.pop("dim_x")
    p.pop("dim_y")
    return ParamLinear(0, dx, dy, **p)


def _best_classical(A: np.ndarray, Q: np.ndarray, dx: int, dy: int) -> ParamLinear:
    dz = dx + dy
    Sig = solve_discrete_lyapunov(A, _pd(Q))
    M0, M1 = Sig, A @ Sig
    Ac = np.zeros((dz, dz))
    Ac[:dx, :dx] = M1[:dx, :dx] @ inv(M0[:dx, :dx])
    Ac[dx:, :] = M1[dx:, :] @ inv(M0)
    Qc = np.zeros((dz, dz))
    Qc[:dx, :dx] = Sig[:dx, :dx] - Ac[:dx, :dx] @ M1[:dx, :dx].T
    Qc[dx:, dx:] = Sig[dx:, dx:] - Ac[dx:, :] @ M1[dx:, :].T
    return _param(Ac, Qc, dx, dy)


def _metrics(param: ParamLinear, data, dx: int) -> tuple[float, float]:
    sm = Linear_PKS_VAR(param)
    sm.process_N_data_smoother(N=len(data) - 1, data_generator=iter(data))
    se = nees = 0.0
    n = 0
    for h, (_k, x, _y) in zip(sm.history, data, strict=True):
        xt = np.asarray(x, float).reshape(dx, 1)
        xh = np.asarray(h["Xkp1_smooth"], float).reshape(dx, 1)
        P = np.asarray(h["PXXkp1_smooth"], float).reshape(dx, dx)
        e = xt - xh
        se += (e.T @ e).item()
        nees += (e.T @ np.linalg.solve(P, e)).item()
        n += 1
    return se / n, nees / (n * dx)          # NEES/dim X: calibrated = 1


def _run(cfg: dict, qscale: float, seeds: int, N: int) -> dict:
    A1, Q1, dx, dy = _make(cfg, 1.0, qscale)             # true couple, rho=1
    A0, Q0, _, _ = _make(cfg, 0.0, qscale)               # ablated
    p_cpl, p_abl = _param(A1, Q1, dx, dy), _param(A0, Q0, dx, dy)
    p_ref = _best_classical(A1, Q1, dx, dy)              # population-best classical
    mc, ma, mr, nc, nr, na = ([] for _ in range(6))
    for s in range(seeds):
        data = Linear_PKF(p_cpl, sKey=s).simulate_N_data(N)
        a, b = _metrics(p_cpl, data, dx); mc.append(a); nc.append(b)
        a, b = _metrics(p_abl, data, dx); ma.append(a); na.append(b)
        a, b = _metrics(p_ref, data, dx); mr.append(a); nr.append(b)
    mc, ma, mr = np.mean(mc), np.mean(ma), np.mean(mr)
    return dict(dx=dx, dy=dy, gap_refit=100 * (mr / mc - 1),
                gap_abl=100 * (ma / mc - 1),
                nees_cpl=float(np.mean(nc)), nees_refit=float(np.mean(nr)),
                nees_abl=float(np.mean(na)))


def main(seeds: int = 150, N: int = 400) -> None:
    print(f"rho=1, {seeds} seeds, N={N}. Gap = MSE penalty vs couple; "
          f"NEES/dimX (calibrated=1).\n")
    print(f"{'model':>6} {'(p,q)':>7} {'obs-noise':>10} "
          f"{'dMSE refit':>11} {'dMSE abl':>9} {'NEES cpl':>9} {'NEES refit':>11} "
          f"{'NEES abl':>9}")
    for tag, cfg in MODELS.items():
        for qscale, label in ((1.0, "nominal"), (4.0, "x4")):
            r = _run(cfg, qscale, seeds, N)
            print(f"{tag:>6} {'(%d,%d)' % (r['dx'], r['dy']):>7} {label:>10} "
                  f"{r['gap_refit']:>10.0f}% {r['gap_abl']:>8.0f}% "
                  f"{r['nees_cpl']:>9.2f} {r['nees_refit']:>11.2f} "
                  f"{r['nees_abl']:>9.2f}")


if __name__ == "__main__":
    main()
