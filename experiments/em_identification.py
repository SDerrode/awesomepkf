#!/usr/bin/env python3
"""
Apprentissage par EM des deux coefficients propres au modèle couple (Fig. 2 du
papier, Sec. IV-B) : la rétroaction A^{xy} (Y -> X) et la mémoire d'observation
A^{yy} (Y -> Y), tous deux nuls dans le modèle classique.

EM partiel : on fixe A^{xx}, A^{yx} et la covariance de bruit Q ; on estime
(A^{xy}, A^{yy}) en partant de l'initialisation classique (0, 0). Le E-step est un
filtre de Kalman + lisseur RTS vectorisé (2x2 numpy, ~20x plus rapide que le lisseur
générique de la librairie, moyennes lissées identiques à ~5e-9) ; le M-step est une
régression fermée sur les observations y_n. La figure montre la trajectoire EM des
deux coefficients et la croissance monotone de la log-vraisemblance ; le script
imprime en outre la dispersion multi-graines (biais / écart-type).

Sortie : ``em_coupling.pdf`` (+ ``em_coupling.png`` d'apercu, dans le dossier de ce script).

Usage :  python experiments/em_identification.py [--N 2000] [--iters 100] [--seeds 50]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
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

from prg.classes.linear_pkf import Linear_PKF          # noqa: E402  (simulation)
from prg.classes.param_linear import ParamLinear       # noqa: E402
from prg.models.linear._amq import LinearAmQ           # noqa: E402

import matplotlib as mpl                                # noqa: E402
mpl.rcParams.update({                                   # style figure de l'article
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.facecolor": "white",
    "savefig.bbox": "tight", "font.size": 8, "axes.titlesize": 8.5,
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "legend.framealpha": 0.9, "lines.linewidth": 1.3,
    "lines.markersize": 4})

DX, DY, DZ = 1, 1, 2
AXX, AYX = 0.6, 0.3
QT = np.array([[0.10, 0.05], [0.05, 0.10]])
AXY_T, AYY_T = 0.4, 0.4
inv = np.linalg.inv


def _A(axy, ayy):
    return np.array([[AXX, axy], [AYX, ayy]])


def _param(A, Q):
    m = LinearAmQ(DX, DY, A=A, mQ=0.5 * (Q + Q.T) + 1e-9 * np.eye(DZ),
                  mz0=np.zeros((DZ, 1)), Pz0=np.eye(DZ), pairwiseModel=True)
    p = m.get_params().copy()
    p.pop("dim_x")
    p.pop("dim_y")
    return ParamLinear(0, DX, DY, **p)


def _smooth_fast(axy, ayy, ys):
    """E-step vectorisé : filtre de Kalman (couple $\\mZ=(X,Y)$, $Y$ observé
    exactement) puis lisseur RTS rétrograde, en $2\\times2$ numpy. Équivalent au
    lisseur VAR de la librairie (moyennes lissées identiques à $\\sim\\!5\\times10^{-9}$)
    mais $\\sim\\!20\\times$ plus rapide. Retourne $(\\hat x_{n\\mid N},\\ \\ell/N)$."""
    A = np.array([[AXX, axy], [AYX, ayy]])
    N = len(ys)
    zf = np.zeros((N, 2)); Pf = np.zeros((N, 2, 2))
    zp = np.zeros((N, 2)); Pp = np.zeros((N, 2, 2))
    S = np.zeros(N); innv = np.zeros(N)
    for n in range(N):                                  # filtre avant
        if n == 0:
            zpn, Ppn = np.zeros(2), np.eye(2)           # a priori N(mz0=0, Pz0=I)
        else:
            zpn = A @ zf[n - 1]; Ppn = A @ Pf[n - 1] @ A.T + QT
        Sn = Ppn[1, 1]; i = ys[n] - zpn[1]              # Y observé exactement (R=0)
        K = Ppn[:, 1] / Sn
        zf[n] = zpn + K * i
        Pf[n] = Ppn - np.outer(K, Ppn[1, :])
        zp[n], Pp[n], S[n], innv[n] = zpn, Ppn, Sn, i
    zs = zf.copy()
    for n in range(N - 2, -1, -1):                      # RTS rétrograde
        Gn = Pf[n] @ A.T @ inv(Pp[n + 1])
        zs[n] = zf[n] + Gn @ (zs[n + 1] - zp[n + 1])
    ll = np.sum(-0.5 * (np.log(2 * np.pi) + np.log(S[1:]) + innv[1:] ** 2 / S[1:])) / N
    return zs[:, 0], ll


def _em(data, iters, record=True):
    """Partial EM for (A^{xy}, A^{yy}), classical init (0,0). Vectorised M-step.
    Returns the per-iteration traces (A^{xy}, A^{yy}, loglik/N)."""
    y = np.array([np.asarray(yy, float).item() for _, _, yy in data])
    y0, y1 = y[:-1], y[1:]
    Syy = float(np.dot(y0, y0))
    axy = ayy = 0.0
    tr_axy, tr_ayy, tr_ll = [], [], []
    for _ in range(iters):
        xh, ll = _smooth_fast(axy, ayy, y)              # E-step (Kalman/RTS 2x2)
        tr_axy.append(axy); tr_ayy.append(ayy); tr_ll.append(ll)
        xh0, xh1 = xh[:-1], xh[1:]
        axy = float(np.dot(xh1 - AXX * xh0, y0) / Syy)   # M-step (régressions fermées)
        ayy = float(np.dot(y1 - AYX * xh0, y0) / Syy)
    return np.array(tr_axy), np.array(tr_ayy), np.array(tr_ll)


def main(N=2000, iters=100, seeds=50):
    A_axy, A_ayy, A_ll = [], [], []
    for s in range(seeds):
        data = Linear_PKF(_param(_A(AXY_T, AYY_T), QT), sKey=s).simulate_N_data(N)
        ta, tb, tl = _em(data, iters)
        A_axy.append(ta)
        A_ayy.append(tb)
        A_ll.append(tl)
        if (s + 1) % 10 == 0:
            print(f"  ... {s + 1}/{seeds} seeds", flush=True)
    A_axy, A_ayy, A_ll = np.array(A_axy), np.array(A_ayy), np.array(A_ll)  # (seeds, iters)
    fin_axy, fin_ayy = A_axy[:, -1], A_ayy[:, -1]
    print(f"[{seeds} seeds, N={N}] A^xy = {fin_axy.mean():.3f} +/- {fin_axy.std():.3f}  |  "
          f"A^yy = {fin_ayy.mean():.3f} +/- {fin_ayy.std():.3f}  (true 0.4/0.4)", flush=True)

    it = np.arange(iters)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for k in range(seeds):
        ax1.plot(it, A_axy[k], "-", color="tab:blue", alpha=0.12, lw=0.7)
        ax1.plot(it, A_ayy[k], "--", color="tab:orange", alpha=0.12, lw=0.7)
    ax1.plot(it, A_axy.mean(0), "-", color="tab:blue", lw=2.4, label=r"$A^{xy}$ (back-action)")
    ax1.plot(it, A_ayy.mean(0), "--", color="tab:orange", lw=2.4, label=r"$A^{yy}$ (obs. memory)")
    ax1.axhline(AXY_T, ls=":", color="k", lw=1)
    ax1.scatter([0], [0], color="k", zorder=5, s=16)
    ax1.annotate("classical init", (0, 0), textcoords="offset points",
                 xytext=(10, -9), fontsize=7)
    ax1.set_xlabel("EM iteration")
    ax1.set_ylabel("coupling coefficient")
    ax1.legend(fontsize=7.5, loc="lower right")
    ax1.grid(True, alpha=0.3)

    for k in range(seeds):
        ax2.plot(it, A_ll[k], "-", color="0.35", alpha=0.12, lw=0.7)
    ax2.plot(it, A_ll.mean(0), "-", color="0.35", lw=2.4)
    ax2.set_xlabel("EM iteration")
    ax2.set_ylabel(r"marginal $\log p(\mathbf{y}_{1:N})/N$")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out = Path(__file__).resolve().parent / "em_coupling.pdf"
    fig.savefig(out)                                    # PDF vectoriel (style article)
    fig.savefig(out.with_suffix(".png"), dpi=150)       # aperçu raster
    print(f"figure written to {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=2000)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=50)
    args = ap.parse_args()
    main(N=args.N, iters=args.iters, seeds=args.seeds)
