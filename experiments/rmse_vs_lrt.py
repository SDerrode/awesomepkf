"""Why an RMSE comparison is not a back-action test -- paper App. (RMSE vs LRT), Fig.
Scalar Y-channel model of em_lrt.py (AXX=0.6, AYX=0.3, A^yy=0.4). H0: A^xy=0 (no back-action).
Each seed FITS classical (A^xy=0) and couple (A^xy free) by exact Y-marginal MLE, then scores
four decision rules for 'is back-action present?'. A valid TEST must reject under H0 ~= alpha.

  R1 naive in-sample : 'couple has lower one-step pred. MSE'   (threshold 0; nested -> size 1)
  R2 naive held-out  : 'couple has lower held-out pred. MSE'   (threshold 0; still miscalibrated)
  R3 held-out + DM   : Diebold-Mariano one-sided at 5% on test squared errors  (calibrated*)
  R4 LRT (paper)     : Lambda = 2(ll1-ll0) > chi2_{pq,0.95} = 3.84             (calibrated)
  (* the plain DM statistic is conservative for NESTED models, cf. Clark-West 2007.)

Also checks the identity  Lambda ~= N log(MSE0/MSE1)  (steady-state approx). Self-contained
(numpy/scipy/matplotlib). Output: figures/rmse_vs_lrt.pdf   Runtime ~8 min (default seeds).
Usage: python experiments/rmse_vs_lrt.py [--seeds0 400] [--seeds1 200]
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import chi2, norm
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
mpl.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.facecolor": "white", "savefig.bbox": "tight",
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 7, "lines.linewidth": 1.6, "lines.markersize": 5})
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / "figures"
OUT.mkdir(parents=True, exist_ok=True)
AXX, AYX = 0.6, 0.3
Q = np.array([[0.10, 0.05], [0.05, 0.10]])
LOG2PI = np.log(2 * np.pi)
CRIT_LRT = chi2.ppf(0.95, 1)      # 3.84
CRIT_DM = norm.ppf(0.95)          # 1.645
RED, GREEN, BLUE = "#D55E00", "#009E73", "#0072B2"


def stat_cov(A):
    return np.linalg.solve(np.eye(4) - np.kron(A, A), Q.reshape(-1)).reshape(2, 2)


def kfilter(Y, axy, ayy):
    """Exact scalar KF on the Y-channel (from em_lrt.yll); returns (loglik, innovations v)."""
    axx, ayx, q11, q22, q12 = AXX, AYX, Q[0, 0], Q[1, 1], Q[0, 1]
    P = stat_cov(np.array([[axx, axy], [ayx, ayy]]))
    p11, p12, p22 = P[0, 0], 0.5 * (P[0, 1] + P[1, 0]), P[1, 1]
    mx = my = 0.0
    ll = 0.0
    v = np.empty(len(Y))
    for i, yn in enumerate(Y):
        mxp = axx * mx + axy * my
        myp = ayx * mx + ayy * my
        a11 = axx * p11 + axy * p12; a12 = axx * p12 + axy * p22
        a21 = ayx * p11 + ayy * p12; a22 = ayx * p12 + ayy * p22
        P11 = a11 * axx + a12 * axy + q11
        P12 = a11 * ayx + a12 * ayy + q12
        P22 = a21 * ayx + a22 * ayy + q22
        vi = yn - myp
        v[i] = vi
        ll += -0.5 * (LOG2PI + np.log(P22) + vi * vi / P22)
        Kx = P12 / P22
        mx, my = mxp + Kx * vi, myp + vi
        p11, p12, p22 = P11 - Kx * P12, P12 - Kx * P22, 0.0
    return ll, v


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


def fit_classical(Ytr):
    r = minimize_scalar(lambda a: -kfilter(Ytr, 0.0, a)[0], bounds=(-0.95, 0.95), method="bounded")
    return r.x, -r.fun


def fit_couple(Ytr):
    best = None
    for x0 in [(0.4, 0.4), (0.0, 0.5), (0.2, 0.45)]:
        with np.errstate(all="ignore"):
            r = minimize(lambda t: -kfilter(Ytr, t[0], t[1])[0], np.array(x0),
                         method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 800})
        if best is None or r.fun < best.fun:
            best = r
    return best.x[0], best.x[1], -best.fun


def one_seed(axy_true, Ntr, Nte, seed):
    rng = np.random.default_rng(seed)
    Y = sim(axy_true, 0.4, Ntr + Nte, rng)
    Ytr = Y[:Ntr]
    ayy0, ll0 = fit_classical(Ytr)
    axy1, ayy1, ll1 = fit_couple(Ytr)
    Lam = max(2.0 * (ll1 - ll0), 0.0)
    _, v0in = kfilter(Ytr, 0.0, ayy0)
    _, v1in = kfilter(Ytr, axy1, ayy1)
    mse0_in, mse1_in = np.mean(v0in ** 2), np.mean(v1in ** 2)
    _, v0f = kfilter(Y, 0.0, ayy0)
    _, v1f = kfilter(Y, axy1, ayy1)
    v0te, v1te = v0f[Ntr:], v1f[Ntr:]
    d = v0te ** 2 - v1te ** 2                       # >0 => couple predicts better
    dm = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)) + 1e-15)
    return Lam, mse1_in < mse0_in, np.mean(v1te ** 2) < np.mean(v0te ** 2), dm > CRIT_DM, \
        Ntr * np.log(mse0_in / mse1_in)


def summarize(axy_true, seeds, base, Ntr=400, Nte=400):
    R = [one_seed(axy_true, Ntr, Nte, base + s) for s in range(seeds)]
    Lam = np.array([r[0] for r in R])
    r1 = float(np.mean([r[1] for r in R]))
    r2 = float(np.mean([r[2] for r in R]))
    r3 = float(np.mean([r[3] for r in R]))
    r4 = float(np.mean(Lam > CRIT_LRT))
    ident = np.array([r[4] for r in R])
    return r1, r2, r3, r4, Lam, ident


def main(seeds0=400, seeds1=200, replot=False):
    axys = np.array([0.0, 0.15, 0.30])
    cache = OUT / "rmse_vs_lrt_data.npz"
    if replot and cache.exists():
        d = np.load(cache)
        size, pow_lrt, pow_dm, corr = list(d["size"]), list(d["pow_lrt"]), list(d["pow_dm"]), float(d["corr"])
        print(f"replot from {cache}  size={np.round(size,3)}  corr={corr:.4f}")
    else:
        res, Lam0, ident0 = {}, None, None
        print(f"{'A^xy':>6} | {'R1 in':>7} {'R2 ho':>7} {'R3 DM':>7} {'R4 LRT':>7}")
        for a in axys:
            sd = seeds0 if a == 0.0 else seeds1
            r1, r2, r3, r4, Lam, ident = summarize(a, sd, base=1000 + int(a * 1000))
            res[a] = (r1, r2, r3, r4)
            if a == 0.0:
                Lam0, ident0 = Lam, ident
            print(f"{a:>6.2f} | {r1:>7.3f} {r2:>7.3f} {r3:>7.3f} {r4:>7.3f}")
        m = np.isfinite(Lam0) & np.isfinite(ident0)
        corr = float(np.corrcoef(Lam0[m], ident0[m])[0, 1])
        print(f"identity Lambda ~= N*log(MSE0/MSE1): corr={corr:.4f}, mean Lambda={Lam0[m].mean():.3f}")
        size = list(res[0.0])                                 # [R1,R2,R3,R4] at H0
        pow_lrt = [res[a][3] for a in axys]
        pow_dm = [res[a][2] for a in axys]
        np.savez(cache, size=size, pow_lrt=pow_lrt, pow_dm=pow_dm, axys=axys, corr=corr)
    rules = ["R1 naive\nin-sample", "R2 naive\nheld-out", "R3 held-out\n+ DM (5%)", "R4 LRT\n(5%)"]
    cols = [RED, RED, GREEN, GREEN]

    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
    ax[0].bar(range(4), size, color=cols, alpha=0.85, width=0.62)
    # same alpha line in both panels (style shared with Fig. em_lrt); labelled once,
    # in the caption, rather than annotated in (a) only.
    ax[0].axhline(0.05, ls=":", color="0.4", lw=1)
    for i, s in enumerate(size):
        ax[0].text(i, s + 0.02, f"{s:.3f}", ha="center", va="bottom", fontsize=7)
    ax[0].set_xticks(range(4)); ax[0].set_xticklabels(rules)
    ax[0].set_ylabel(r"false-positive rate under $H_0$ (size)")
    ax[0].set_ylim(0, 1.12)
    ax[0].set_title("(a) uncalibrated RMSE-picking is not a test", fontsize=8)
    ax[0].annotate("in-sample: nested\ntautology", xy=(0, 1.0), xytext=(0.45, 0.83),
                   fontsize=6.3, color=RED, ha="left",
                   arrowprops=dict(arrowstyle="->", color=RED, lw=0.8))
    ax[0].annotate(r"held-out: still $\sim$6$\times$" "\nnominal", xy=(1, size[1]), xytext=(1.2, 0.52),
                   fontsize=6.3, color=RED, ha="left",
                   arrowprops=dict(arrowstyle="->", color=RED, lw=0.8))
    ax[1].plot(axys, pow_lrt, "-o", color=BLUE, label=f"R4 LRT, size {size[3]:.3f}")
    ax[1].plot(axys, pow_dm, "-s", color=GREEN, label=f"R3 held-out RMSE+DM, size {size[2]:.3f}")
    ax[1].axhline(0.05, ls=":", color="0.4", lw=1)
    ax[1].set_xlabel(r"true back-action $A^{xy}$")
    ax[1].set_ylabel("power (rejection rate)")
    ax[1].set_ylim(-0.03, 1.0)
    ax[1].set_title("(b) both calibrated tests have power (DM conservative)", fontsize=8)
    ax[1].legend(loc="upper left"); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "rmse_vs_lrt.pdf")
    fig.savefig(OUT / "rmse_vs_lrt_preview.png", dpi=150)
    print("saved", OUT / "rmse_vs_lrt.pdf")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds0", type=int, default=400)
    ap.add_argument("--seeds1", type=int, default=200)
    ap.add_argument("--replot", action="store_true", help="re-render from cached figures/rmse_vs_lrt_data.npz")
    args = ap.parse_args()
    main(seeds0=args.seeds0, seeds1=args.seeds1, replot=args.replot)
