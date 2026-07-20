#!/usr/bin/env python3
"""
Back-action detection by likelihood-ratio test (LRT) --- Fig. 6 of the paper,
Sec. IV-C. We test H0: A^{xy}=0 (no back-action) against A^{xy}!=0, via
Lambda = 2[ell(A^{xy} free) - ell(A^{xy}=0)]. Each likelihood is the marginal
likelihood of the Y-channel; A^{yy} is a free nuisance on both sides (A^{xx},
A^{yx}, Q frozen, which pins the gauge). Under H0, Lambda ~ chi2_1.

Self-contained fast implementation (~1 min, no awesomepkf dependency): the
marginal Y-likelihood is evaluated by an exact scalar Kalman filter and maximised
directly (its maximiser is the fixed point of the partial EM described in the
text). This is the script that produces the deployed figure.

Left  : empirical null of Lambda (red histogram) vs chi2_1 density (blue curve),
        dotted critical value at alpha=5%.
Right : test power (rejection rate) vs true back-action A^{xy}; the size
        (at A^{xy}=0) is close to alpha.

Output: ``figures/em_lrt.pdf`` (+ ``figures/em_lrt_preview.png``).
Usage : python experiments/em_lrt.py [--seeds 350] [--N 400] [--npow 120]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import chi2, ncx2

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
mpl.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.facecolor": "white",
    "savefig.bbox": "tight", "font.size": 8, "axes.titlesize": 8.5,
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "legend.framealpha": 0.9, "lines.linewidth": 1.3,
    "lines.markersize": 4})
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / "figures"
AXX, AYX = 0.6, 0.3
Q = np.array([[0.10, 0.05], [0.05, 0.10]])
LOG2PI = np.log(2 * np.pi)

# --- spectral (Whittle/Itakura-Saito) noncentrality lambda = 2N*KL_rate, for the
#     predicted-power overlay validating Prop. (LRT law): lambda -> chi2_1(lambda). ---
WGRID = np.linspace(-np.pi, np.pi, 4096, endpoint=False)


def y_spectrum(A):
    """Scalar spectral density of Y=Z[1] for Z_{n+1}=A Z_n + w, cov Q (vectorized 2x2)."""
    z = np.exp(-1j * WGRID)
    axx, axy, ayx, ayy = A[0, 0], A[0, 1], A[1, 0], A[1, 1]
    det = (1 - axx * z) * (1 - ayy * z) - axy * ayx * z * z
    a = ayx * z; b = 1 - axx * z
    num = (Q[0, 0] * np.abs(a)**2 + Q[1, 1] * np.abs(b)**2
           + 2 * Q[0, 1] * np.real(a * np.conj(b)))
    return num / np.abs(det)**2


def _kl_rate(f, g):
    r = f / g
    return float(np.mean(r - 1.0 - np.log(r)) * 0.5)   # (1/4pi) INT dω  == mean/2


def predicted_lambda(axy, ayy_true, N):
    """lambda = 2N * min_{A^yy} KL_rate( couple || A^xy=0 null ), A^xx,A^yx,Q frozen."""
    f = y_spectrum(np.array([[AXX, axy], [AYX, ayy_true]]))
    r = minimize_scalar(lambda ayy: _kl_rate(f, y_spectrum(np.array([[AXX, 0.0], [AYX, ayy]]))),
                        bounds=(-0.95, 0.95), method="bounded", options={"xatol": 1e-8})
    return 2 * N * r.fun


def stat_cov(A):
    return np.linalg.solve(np.eye(4) - np.kron(A, A), Q.reshape(-1)).reshape(2, 2)


def yll(Y, axy, ayy):
    """Exact marginal log-likelihood of Y (scalar Kalman filter on the Y-channel)."""
    axx, ayx, q11, q22, q12 = AXX, AYX, Q[0, 0], Q[1, 1], Q[0, 1]
    P = stat_cov(np.array([[axx, axy], [ayx, ayy]]))
    p11, p12, p22 = P[0, 0], 0.5 * (P[0, 1] + P[1, 0]), P[1, 1]
    mx = my = 0.0
    ll = 0.0
    for yn in Y:
        mxp = axx * mx + axy * my
        myp = ayx * mx + ayy * my
        a11 = axx * p11 + axy * p12
        a12 = axx * p12 + axy * p22
        a21 = ayx * p11 + ayy * p12
        a22 = ayx * p12 + ayy * p22
        P11 = a11 * axx + a12 * axy + q11
        P12 = a11 * ayx + a12 * ayy + q12
        P22 = a21 * ayx + a22 * ayy + q22
        v = yn - myp
        ll += -0.5 * (LOG2PI + np.log(P22) + v * v / P22)
        Kx, Ky = P12 / P22, P22 / P22
        mx, my = mxp + Kx * v, myp + Ky * v
        p11, p12, p22 = P11 - Kx * P12, P12 - Kx * P22, P22 - Ky * P22
    return ll


def sim(axy, ayy, N, rng, burn=200):
    A = np.array([[AXX, axy], [AYX, ayy]])
    L = np.linalg.cholesky(Q)
    z = rng.multivariate_normal(np.zeros(2), stat_cov(A))
    Y = np.empty(N)
    for t in range(N + burn):
        z = A @ z + L @ rng.standard_normal(2)
        if t >= burn:
            Y[t - burn] = z[1]
    return Y


def lrt(Y):
    with np.errstate(all="ignore"):     # infeasible optimiser probes -> nan, rejected
        l0 = -minimize_scalar(lambda a: -yll(Y, 0.0, a), bounds=(-0.95, 0.95),
                              method="bounded").fun
        best = None
        for x0 in [(0.4, 0.4), (0.0, 0.5), (0.2, 0.45)]:
            r = minimize(lambda t: -yll(Y, t[0], t[1]), np.array(x0),
                         method="Nelder-Mead",
                         options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 800})
            if best is None or r.fun < best:
                best = r.fun
    return max(2 * ((-best) - l0), 0.0)


def main(seeds=350, N=400, npow=120):
    crit = chi2.ppf(0.95, 1)
    lam0 = np.array([lrt(sim(0.0, 0.4, N, np.random.default_rng(1000 + s)))
                     for s in range(seeds)])
    size = float(np.mean(lam0 > crit))
    print(f"null: size={size:.3f} mean={lam0.mean():.3f}")
    axyg = np.array([0.0, 0.075, 0.15, 0.225, 0.30, 0.375, 0.45])
    powm, powse = [], []
    for a in axyg:
        L = np.array([lrt(sim(a, 0.4, N, np.random.default_rng(5000 + int(a * 1000) * 97 + s)))
                      for s in range(npow)])
        p = float(np.mean(L > crit))
        powm.append(p)
        powse.append(1.96 * np.sqrt(p * (1 - p) / npow))
    powm, powse = np.array(powm), np.array(powse)
    print("power:", np.round(powm, 3))
    # predicted power from the spectral noncentrality lambda(A^xy) = 2N * KL_rate
    axyf = np.linspace(0.0, axyg[-1], 60)
    lam_pred = np.array([predicted_lambda(a, 0.4, N) for a in axyf])
    pow_pred = 1.0 - ncx2.cdf(crit, 1, lam_pred)
    print(f"predicted lambda at A^xy={axyg[-1]}: {predicted_lambda(axyg[-1], 0.4, N):.1f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.6))
    ax1.hist(lam0, bins=45, range=(0, 12), density=True, color="C3", alpha=0.55,
             label=r"$\Lambda$ under $H_0$")
    xx = np.linspace(0.02, 12, 300)
    ax1.plot(xx, chi2.pdf(xx, 1), color="C0", lw=1.8, label=r"$\chi^2_1$")
    ax1.axvline(crit, ls=":", color="0.4", lw=1)
    ax1.set_xlabel(r"$\Lambda$")
    ax1.set_ylabel("density")
    ax1.set_ylim(0, 1.0)
    ax1.set_title(f"(a) null: size $={size:.3f}$")
    ax1.legend(loc="upper right")
    ax1.grid(alpha=0.3)
    ax2.axhline(0.05, ls=":", color="0.4", lw=1, label=r"size $\alpha=0.05$")
    ax2.plot(axyf, pow_pred, "--", color="C0", lw=1.6, zorder=1,
             label=r"predicted $\chi^2_1(\lambda)$, $\lambda{=}2N\,\mathrm{KL}_{\mathrm{rate}}$")
    ax2.errorbar(axyg, powm, yerr=powse, fmt="o", color="C3", capsize=2, zorder=3,
                 label="empirical $\\pm$ 95% CI")
    ax2.set_xlabel(r"true back-action $A^{xy}$")
    ax2.set_ylabel("rejection rate (power)")
    ax2.set_ylim(-0.03, 1.05)
    ax2.set_title("(b) power")
    ax2.legend(loc="upper left")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "em_lrt.pdf")
    fig.savefig(OUT / "em_lrt_preview.png", dpi=150)
    print(f"figure written to {OUT / 'em_lrt.pdf'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=350)
    ap.add_argument("--N", type=int, default=400)
    ap.add_argument("--npow", type=int, default=120)
    args = ap.parse_args()
    main(seeds=args.seeds, N=args.N, npow=args.npow)
