"""Discriminating pairwise models: why MBF is the numerically safest of the six smoothers
(paper §II-E, Table + Fig). The six smoothers are algebraically the same estimator, so they
differ only NUMERICALLY under stress. We engineer pairwise models (p=2, q=1) that each drive
one inverted matrix ill-conditioned, drive the ACTUAL library smoothers, and report two
oracle-free quantities:
  M1  condition number of the matrix each variant inverts (root cause, no oracle):
      RTS -> P_{n|n-1} (dim p+q), MBF/BF -> S_n=P^yy_{n|n-1} (dim q), 2F/DWY -> Sigma_n
      (Lyapunov prior), VAR -> R_n (process noise).
  M3  worst per-step gap of each variant vs the RTS reference (corroboration).
Regimes: R1 process-noise starvation (mQ -> 0, i.e. R_n -> singular boundary of the
well-posedness theorem); R2 ill-conditioned prior (slow-mixing, non-normal A).
Run (from the awesomepkf repo root):  python experiments/discriminating_models.py
Adapted from the linear-smoothers TAC note (adds VAR + the figure). Output: figures/discriminating.pdf
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib as mpl
mpl.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.facecolor": "white", "savefig.bbox": "tight",
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 6.6, "lines.linewidth": 1.4, "lines.markersize": 3.5})
import matplotlib.pyplot as plt

from prg.classes.linear_pks import (
    Linear_PKS_RTS, Linear_PKS_BF, Linear_PKS_MBF, Linear_PKS_MF, Linear_PKS_DWY, Linear_PKS_VAR)
from prg.classes.param_linear import ParamLinear
from prg.models.linear._amq import LinearAmQ
from prg.utils.exceptions import PKFError

OUT = str(Path(__file__).resolve().parents[1] / "figures"); Path(OUT).mkdir(parents=True, exist_ok=True)
VARIANTS = {"RTS": Linear_PKS_RTS, "BF": Linear_PKS_BF, "MBF": Linear_PKS_MBF,
            "2F": Linear_PKS_MF, "DWY": Linear_PKS_DWY, "VAR": Linear_PKS_VAR}
NONREF = {k: v for k, v in VARIANTS.items() if k != "RTS"}

# engineered models (p=2, q=1)
A_R1 = np.array([[0.60, 0.10, 0.05], [0.00, 0.50, 0.03], [0.40, 0.30, 0.30]])
BASE = np.array([[1.0, 0.2, 0.0], [0.2, 1.0, 0.0], [0.0, 0.0, 1.0]])   # noise shape (R1)
PZ0 = np.eye(3) * 0.1
MQ_R2 = np.array([[0.05, 0.01, 0.0], [0.01, 0.05, 0.0], [0.0, 0.0, 0.03]])


def _param(A, mQ):
    m = LinearAmQ(2, 1, A=A, mQ=mQ, mz0=np.zeros((3, 1)), Pz0=PZ0.copy(), pairwiseModel=True)
    p = m.get_params().copy(); p.pop("dim_x"); p.pop("dim_y")
    return ParamLinear(0, 2, 1, **p)


def _cond_sigma(A, mQ, N):
    S, c = PZ0.copy(), float(np.linalg.cond(PZ0))
    for _ in range(N):
        S = A @ S @ A.T + mQ; c = max(c, float(np.linalg.cond(S)))
    return c


def _cond_filter(param, N, seed):
    """Max cond of P_{n|n-1} (RTS) and S_n (MBF/BF) over a filter run (from history)."""
    f = Linear_PKS_RTS(param, sKey=seed); f.process_N_data_smoother(N=N)
    cP = cS = 0.0
    for rec in f.history:
        S = rec["Skp1"]; Pxy = rec["Kkp1"] @ S
        P = np.block([[rec["PXXkp1_predict"], Pxy], [Pxy.T, S]])
        cP = max(cP, float(np.linalg.cond(P))); cS = max(cS, float(np.linalg.cond(S)))
    return cP, cS


def _gaps_vs_rts(param, N, seed):
    rts = Linear_PKS_RTS(param, sKey=seed); rts.process_N_data_smoother(N=N)
    out = {}
    for name, cls in NONREF.items():
        try:
            v = cls(param, sKey=seed); v.process_N_data_smoother(N=N)
            out[name] = max(float(np.max(np.abs(a["Xkp1_smooth"] - b["Xkp1_smooth"])))
                            for a, b in zip(rts.history, v.history))
        except PKFError:
            out[name] = np.nan
    return out


def _row(A, mQ, N, seed):
    param = _param(A, mQ)
    cP, cS = _cond_filter(param, N, seed)
    cSig = _cond_sigma(A, mQ, N)
    cR = float(np.linalg.cond(mQ))
    return cP, cS, cSig, cR, _gaps_vs_rts(param, N, seed)


def main():
    # ---- printed tables (paper Table) ----
    print("=== R1: process-noise starvation (mQ = eps*base -> 0) ===")
    print(f"{'eps':>8} {'cond(P)':>9} {'cond(S)':>8} {'cond(Sig)':>9} {'cond(R)':>8} | gaps: "
          + "".join(f"{k:>9}" for k in NONREF))
    for eps in (1e-5, 1e-7, 1e-11):
        cP, cS, cSig, cR, g = _row(A_R1, eps * BASE, 400, 1)
        print(f"{eps:>8.0e} {cP:>9.1e} {cS:>8.1e} {cSig:>9.1e} {cR:>8.1e} |       "
              + "".join(f"{g[k]:>9.1e}" for k in NONREF))
    print("\n=== R2: ill-conditioned prior (slow-mixing non-normal A, rho -> 1) ===")
    print(f"{'rho':>6} {'cond(P)':>9} {'cond(S)':>8} {'cond(Sig)':>9} {'cond(R)':>8} | gaps: "
          + "".join(f"{k:>9}" for k in NONREF))
    for s in (0.90, 0.97, 0.996):
        A = np.array([[s, 0.60, 0.02], [0.0, 0.85, 0.0], [0.03, 0.02, 0.45]])
        rho = float(np.max(np.abs(np.linalg.eigvals(A))))
        cP, cS, cSig, cR, g = _row(A, MQ_R2, 600, 1)
        print(f"{rho:>6.3f} {cP:>9.1e} {cS:>8.1e} {cSig:>9.1e} {cR:>8.1e} |       "
              + "".join(f"{g[k]:>9.1e}" for k in NONREF))

    # ---- R3: same starvation at q=2, where cond(S_n) is not a 1x1 tautology ----
    # At q=1 the predicted-observation covariance is a scalar, so cond(S_n)=1 identically
    # and the column carries no information. With a genuine 2-D observation channel the
    # number becomes meaningful -- and reveals that MBF's advantage, while large, is bounded.
    print("\n=== R3: noise starvation at p=q=2 (cond(S) is informative here) ===")
    A3 = np.array([[0.60, 0.10, 0.05, 0.02], [0.00, 0.50, 0.03, 0.01],
                   [0.40, 0.30, 0.30, 0.05], [0.15, 0.55, 0.04, 0.25]])
    B3 = np.array([[1.0, 0.2, 0.0, 0.0], [0.2, 1.0, 0.0, 0.0],
                   [0.0, 0.0, 1.0, 0.3], [0.0, 0.0, 0.3, 1.0]])
    print(f"{'eps':>8} {'cond(P)':>10} {'cond(S)':>9} {'cond(R)':>9}   ratio P/S")
    for e in (1e-1, 1e-3, 1e-5, 1e-7, 1e-8):
        try:
            m = LinearAmQ(2, 2, A=A3, mQ=e * B3, mz0=np.zeros((4, 1)),
                          Pz0=np.eye(4) * 0.1, pairwiseModel=True)
            pr = m.get_params().copy(); pr.pop("dim_x"); pr.pop("dim_y")
            f = Linear_PKS_RTS(ParamLinear(0, 2, 2, **pr), sKey=1)
            f.process_N_data_smoother(N=400)
            cP = cS = 0.0
            for rec in f.history:
                S = np.atleast_2d(rec["Skp1"]); Pxy = np.atleast_2d(rec["Kkp1"]) @ S
                Pm = np.block([[np.atleast_2d(rec["PXXkp1_predict"]), Pxy], [Pxy.T, S]])
                cP = max(cP, float(np.linalg.cond(Pm))); cS = max(cS, float(np.linalg.cond(S)))
            print(f"{e:>8.0e} {cP:>10.2e} {cS:>9.2f} {float(np.linalg.cond(e * B3)):>9.2e}"
                  f"  {cP / cS:>10.1e}")
        except PKFError:
            print(f"{e:>8.0e}  filter aborts: S_n numerically singular "
                  f"(so 'never ill-conditioned' is false)")

    # ---- figure: R1 fine sweep (cond + accuracy vs eps) ----
    eps = np.logspace(-1, -11, 11)
    cP, cS, cSig, cR = [], [], [], []
    gaps = {k: [] for k in NONREF}
    for e in eps:
        a, b, c, d, g = _row(A_R1, e * BASE, 400, 1)
        cP.append(a); cS.append(b); cSig.append(c); cR.append(d)
        for k in NONREF:
            gaps[k].append(g[k])
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
    COL = {"P": "#D55E00", "Sig": "#CC79A7", "R": "#999999", "S": "#0072B2"}
    ax[0].loglog(eps, cP, "-o", color=COL["P"], label=r"$\mathbf{P}_{n|n-1}$ (RTS)")
    ax[0].loglog(eps, cSig, "-D", color=COL["Sig"], label=r"$\boldsymbol{\Sigma}_n$ (2F/DWY)")
    ax[0].loglog(eps, cR, "-s", color=COL["R"], label=r"$\mathbf{R}_n$ (VAR)")
    ax[0].loglog(eps, cS, "-^", color=COL["S"], lw=2.2, label=r"$\mathbf{S}_n$ (BF/MBF)")
    ax[0].set_xlabel(r"process-noise scale $\varepsilon$"); ax[0].invert_xaxis()
    ax[0].set_ylabel("cond. of inverted matrix")
    ax[0].set_title("(a) conditioning under noise starvation")
    ax[0].legend(loc="upper left"); ax[0].grid(alpha=0.3, which="both")
    STY = {"BF": ("-^", "#0072B2", {}), "MBF": ("-v", "#009E73", {}),
           "VAR": ("-s", "#999999", {}),
           # 2F and DWY both invert Sigma_n, so their curves nearly coincide. Neither
           # may hide the other: 2F is a solid line with large hollow markers, DWY a
           # dashed line with small filled ones drawn on top -- the dashes let 2F show
           # through and DWY's marker sits inside 2F's ring.
           "2F":  ("-D",  "#CC79A7", {"mfc": "none", "ms": 9, "lw": 1.4}),
           "DWY": ("--P", "#E69F00", {"ms": 5, "lw": 1.4, "zorder": 5})}
    ORDER = ["BF", "MBF", "VAR", "2F", "DWY"]
    for k in ORDER:
        m, c, kw = STY[k]
        ax[1].loglog(eps, np.array(gaps[k]) + 1e-17, m, color=c, label=k, **kw)
    ax[1].set_xlabel(r"process-noise scale $\varepsilon$"); ax[1].invert_xaxis()
    ax[1].set_ylabel(r"worst $\|\cdot\|_\infty$ gap vs RTS")
    ax[1].set_title("(b) resulting smoothed-state error")
    hdl, lab = ax[1].get_legend_handles_labels()      # restore the NONREF reading order
    o = [lab.index(k) for k in NONREF]
    ax[1].legend([hdl[i] for i in o], [lab[i] for i in o], loc="upper left", ncol=2)
    ax[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT + "/discriminating.pdf")
    fig.savefig(OUT + "/discriminating_preview.png", dpi=150)
    print("\nsaved", OUT + "/discriminating.pdf")


if __name__ == "__main__":
    main()
