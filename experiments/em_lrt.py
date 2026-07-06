#!/usr/bin/env python3
"""
Détection de la rétroaction par test du rapport de vraisemblance (LRT) --- Fig. 3
du papier, Sec. IV-C. On teste H0 : A^{xy}=0 (pas de rétroaction) contre A^{xy}!=0,
via Lambda = 2[ell(A^{xy} libre) - ell(A^{xy}=0)], chaque vraisemblance étant
maximisée par un EM partiel (A^{yy} nuisance libre dans les deux ; A^{xx},A^{yx},Q
fixés). Sous H0, Lambda ~ chi2_1.

Gauche  : distribution empirique de Lambda sous H0 vs densité chi2_1.
Droite  : puissance du test (taux de rejet a alpha=5%) en fonction de la vraie
          force de rétroaction A^{xy} (taille du test = puissance en A^{xy}=0).

Sortie : ``em_lrt.png``.  Usage : python em_lrt.py [--seeds 350] [--N 400]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2


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

from prg.classes.linear_pkf import Linear_PKF          # noqa: E402
from prg.classes.linear_pks import Linear_PKS_VAR      # noqa: E402
from prg.classes.param_linear import ParamLinear       # noqa: E402
from prg.models.linear._amq import LinearAmQ           # noqa: E402

DX, DY, DZ = 1, 1, 2
AXX, AYX, AYY_T = 0.6, 0.3, 0.4
QT = np.array([[0.10, 0.05], [0.05, 0.10]])
inv = np.linalg.inv
ALPHA = 0.05
CRIT = chi2.ppf(1 - ALPHA, df=1)          # ~3.841


def _A(axy, ayy):
    return np.array([[AXX, axy], [AYX, ayy]])


def _param(A, Q):
    m = LinearAmQ(DX, DY, A=A, mQ=0.5 * (Q + Q.T) + 1e-9 * np.eye(DZ),
                  mz0=np.zeros((DZ, 1)), Pz0=np.eye(DZ), pairwiseModel=True)
    p = m.get_params().copy()
    p.pop("dim_x")
    p.pop("dim_y")
    return ParamLinear(0, DX, DY, **p)


def _smooth_xhat(axy, ayy, data):
    sm = Linear_PKS_VAR(_param(_A(axy, ayy), QT))
    sm.process_N_data_smoother(N=len(data) - 1, data_generator=iter(data))
    return sm


def _loglik(sm):
    S, i = [], []
    for n, h in enumerate(sm.history):
        if n == 0:
            continue
        S.append(np.atleast_2d(h["Skp1"]).item())
        i.append(np.asarray(h["ikp1"], float).item())
    S, i = np.array(S), np.array(i)
    return float(np.sum(-0.5 * (np.log(2 * np.pi) + np.log(S) + i * i / S)))


def _fit(data, free_axy, iters=50, tol_ll=1e-4):
    """EM maximizing the marginal loglik (scalar model): A^{yy} always free,
    A^{xy} free iff free_axy (else pinned to 0). Vectorised M-step, stopped on the
    log-likelihood plateau (the quantity the LRT needs). Returns that log-likelihood."""
    y = np.array([np.asarray(yy, float).item() for _, _, yy in data])
    y0, y1 = y[:-1], y[1:]
    Syy = float(np.dot(y0, y0))
    axy = ayy = 0.0
    ll = -np.inf
    for _ in range(iters):
        sm = _smooth_xhat(axy, ayy, data)
        ll_cur = _loglik(sm)                 # marginal loglik at current params
        if ll_cur - ll < tol_ll:             # loglik has plateaued -> converged
            return ll_cur
        ll = ll_cur
        xh = np.array([np.asarray(h["Xkp1_smooth"], float).item() for h in sm.history])
        xh0, xh1 = xh[:-1], xh[1:]
        axy = float(np.dot(xh1 - AXX * xh0, y0) / Syy) if free_axy else 0.0
        ayy = float(np.dot(y1 - AYX * xh0, y0) / Syy)
    return _loglik(_smooth_xhat(axy, ayy, data))


def _lambda(a, seed, N):
    try:
        data = Linear_PKF(_param(_A(a, AYY_T), QT), sKey=seed).simulate_N_data(N)
        return max(2.0 * (_fit(data, True) - _fit(data, False)), 0.0)
    except Exception:
        return float("nan")


def main(seeds=350, N=400, npow=40):
    axys = [0.0, 0.075, 0.15, 0.225, 0.30, 0.375, 0.45]
    lam_list = []
    for s in range(seeds):
        lam_list.append(_lambda(0.0, s, N))
        if (s + 1) % 50 == 0:
            print(f"  ... H0 progress {s + 1}/{seeds}", flush=True)
    lam_h0 = np.array(lam_list)
    lam_h0 = lam_h0[np.isfinite(lam_h0)]
    size = float(np.mean(lam_h0 > CRIT))
    print(f"[H0, {lam_h0.size} seeds, N={N}] size (alpha={ALPHA}) = {size:.3f} ; "
          f"mean Lambda = {lam_h0.mean():.2f} (chi2_1 mean = 1)", flush=True)
    power, perr = [], []
    for a in axys:
        lam = np.array([_lambda(a, 1000 + s, N) for s in range(npow)])
        lam = lam[np.isfinite(lam)]
        p = float(np.mean(lam > CRIT))
        power.append(p)
        perr.append(float(np.sqrt(p * (1 - p) / max(lam.size, 1))))
        print(f"  A^xy={a:.3f} -> power = {p:.3f} (n={lam.size})", flush=True)

    # --- figure ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.7))
    xx = np.linspace(0, 12, 400)
    ax1.hist(lam_h0, bins=45, range=(0, 12), density=True, color="C0", alpha=0.55,
             label=r"$\Lambda$ under $H_0$")
    ax1.plot(xx, chi2.pdf(xx, df=1), "C3-", lw=1.8, label=r"$\chi^2_1$")
    ax1.axvline(CRIT, ls=":", color="k", lw=1)
    ax1.set_xlabel(r"$\Lambda$")
    ax1.set_ylabel("density")
    ax1.set_xlim(0, 12)
    ax1.set_ylim(0, 1.0)
    ax1.legend(fontsize=8)

    ax2.axhline(ALPHA, ls=":", color="k", lw=1, label=r"size $\alpha=0.05$")
    ax2.errorbar(axys, power, yerr=1.96 * np.array(perr), fmt="o-", ms=4,
                 color="C0", capsize=2, lw=1.4, label="power $\\pm$ 95\\% CI")
    ax2.set_xlabel(r"true back-action $A^{xy}$")
    ax2.set_ylabel("rejection rate (power)")
    ax2.set_ylim(-0.03, 1.03)
    ax2.legend(fontsize=7.5, loc="center right")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out = Path(__file__).resolve().parent / "em_lrt.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"figure written to {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=350)
    ap.add_argument("--N", type=int, default=400)
    ap.add_argument("--npow", type=int, default=40)
    args = ap.parse_args()
    main(seeds=args.seeds, N=args.N, npow=args.npow)
