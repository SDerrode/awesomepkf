#!/usr/bin/env python3
"""
Détection et apprentissage de la rétroaction sur DONNÉES RÉELLES (rapport §3.6).

Route "beta" (couple entièrement observé) : sur une série bivariée réelle (X,Y) où
les DEUX composantes sont observées, on ajuste le VAR(1) pairwise
    Z_{n+1} = A Z_n + w,  w ~ N(0, S),  Z=(X,Y),  A=[[Axx,Axy],[Ayx,Ayy]],
et on teste H0: Axy = 0 (Y ne pilote pas X au lag 1), avec S LIBRE dans les deux
modèles — de sorte que la corrélation de bruit contemporaine R_xy = S_xy ne puisse
pas se faire passer pour une rétroaction (garde-fou). Le null est calibré par
surrogates à PHASE RANDOMISÉE (phase_randomize ci-dessous : préservent le spectre et
donc l'autocovariance, mais détruisent le couplage retardé — et aussi la non-linéarité
et les queues lourdes), les asymptotiques chi2 étant fragiles sur du réel. On teste les
deux directions (inversion X<->Y). [NB : le bootstrap par blocs plus bas ne sert qu'aux
IC des coefficients appris, pas à la calibration du null.]

Systèmes :
  * Chémostat algue-rotifère (Lotka-Volterra, prédateur-proie) -> rétroaction attendue
  * S&P 500 (volatilité/rendement)                             -> couplage attendu
  * Éolien (vent/puissance)                                    -> nuance : lien
        vent->puissance CONTEMPORAIN, donc pas une rétroaction *retardée*
  * Surrogate (une série décalée circulairement)               -> couplage détruit :
        contrôle négatif, doit garder H0

Volet apprentissage : sur le chémostat, coefficients couple (Axy, Ayy) + IC bootstrap,
et MSE one-step-ahead hors-échantillon du modèle couple vs l'ablation classique (Axy=0).

Sortie : ``em_realdata.png``.  Usage : python generate_em_realdata.py [--B 500]
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
mpl.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.facecolor": "white", "savefig.bbox": "tight",
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 6.8, "lines.linewidth": 1.3})
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2

REJECT, KEEP = "#D55E00", "#009E73"   # Okabe-Ito vermillion / bluish-green (colourblind-safe)


# --------------------------------------------------------------------------- #
#  locate the awesomePKF data directory                                       #
# --------------------------------------------------------------------------- #
def _data_root() -> Path:
    env = os.environ.get("AWESOMEPKF_ROOT")
    cands = []
    if env:
        cands.append(Path(env).expanduser() / "data")
    here = Path(__file__).resolve()
    for anc in [here.parent, *here.parents]:
        cands.append(anc / "awesomePKF" / "data")
        cands.append(anc / "data")
    for c in cands:
        if (c / "datafile" / "realdata").is_dir():
            return c
    raise RuntimeError("Cannot locate awesomePKF/data (set AWESOMEPKF_ROOT).")


DATA = _data_root()


def _std(a):
    a = np.asarray(a, float)
    return (a - a.mean()) / a.std(ddof=1)


# --------------------------------------------------------------------------- #
#  beta engine: fully-observed bivariate Gaussian VAR(1), LRT of Axy = 0      #
# --------------------------------------------------------------------------- #
def _ll(Z0, Z1, A, S):
    E = Z1 - Z0 @ A.T
    Si = np.linalg.inv(S)
    _, ld = np.linalg.slogdet(S)
    q = np.einsum("ni,ij,nj->n", E, Si, E)
    return float(np.sum(-0.5 * (2 * np.log(2 * np.pi) + ld + q)))


def fit_full(Z):
    Z0, Z1 = Z[:-1], Z[1:]
    A = (Z1.T @ Z0) @ np.linalg.inv(Z0.T @ Z0)          # OLS = ML (regressors égaux)
    E = Z1 - Z0 @ A.T
    S = (E.T @ E) / len(Z0)
    return A, S, _ll(Z0, Z1, A, S)


def fit_restricted(Z, iters=100, tol=1e-10):
    """ML avec A[0,1]=0 (l'équation-X exclut Y_n), S libre — FGLS."""
    Z0, Z1 = Z[:-1], Z[1:]
    X0, Y0, X1, Y1 = Z0[:, 0], Z0[:, 1], Z1[:, 0], Z1[:, 1]
    T = len(Z0)
    axx = (X0 @ X1) / (X0 @ X0)
    G = np.column_stack([X0, Y0])
    ayx, ayy = np.linalg.solve(G.T @ G, G.T @ Y1)
    A = np.array([[axx, 0.0], [ayx, ayy]])
    ll_prev = -np.inf
    for _ in range(iters):
        ex, ey = X1 - axx * X0, Y1 - ayx * X0 - ayy * Y0
        S = np.array([[ex @ ex, ex @ ey], [ex @ ey, ey @ ey]]) / T
        A = np.array([[axx, 0.0], [ayx, ayy]])
        ll = _ll(Z0, Z1, A, S)
        if ll - ll_prev < tol:
            break
        ll_prev = ll
        w = np.linalg.inv(S)
        w00, w01, w11 = w[0, 0], w[0, 1], w[1, 1]
        sxx, sxy, syy = X0 @ X0, X0 @ Y0, Y0 @ Y0
        M = np.array([[sxx * w00, sxx * w01, sxy * w01],
                      [sxx * w01, sxx * w11, sxy * w11],
                      [sxy * w01, sxy * w11, syy * w11]])
        v = np.array([X0 @ (w00 * X1 + w01 * Y1),
                      X0 @ (w01 * X1 + w11 * Y1),
                      Y0 @ (w01 * X1 + w11 * Y1)])
        axx, ayx, ayy = np.linalg.solve(M, v)
    return A, S, ll


def lrt(Z):
    _, _, llf = fit_full(Z)
    Ar, Sr, llr = fit_restricted(Z)
    return max(2.0 * (llf - llr), 0.0), Ar, Sr


def detect(Z, B, seed):
    """Test surrogate de H0: Y ne pilote pas X (Axy=0). Le null est obtenu en
    randomisant la PHASE du pilote Y (col.1) — spectre/autocov préservés, couplage
    croisé détruit —, X (col.0) inchangé. p_surr = P(Lambda_surr >= Lambda_obs).
    Sans modèle : robuste à la mauvaise spécification VAR(1) et à la non-normalité,
    contrairement à l'asymptotique chi2 (qui sur-rejette ici). Retourne aussi p_chi2
    en référence, et la distribution surrogate."""
    stat = lrt(Z)[0]
    rng = np.random.default_rng(seed)
    X, Y = Z[:, 0], Z[:, 1]
    nulls = np.array([lrt(np.column_stack([X, phase_randomize(Y, rng)]))[0]
                      for _ in range(B)])
    p_surr = (1 + int(np.sum(nulls >= stat))) / (1 + B)
    return stat, p_surr, float(chi2.sf(stat, 1)), nulls


# --------------------------------------------------------------------------- #
#  loaders                                                                    #
# --------------------------------------------------------------------------- #
def load_chemostat():
    d = pd.read_csv(DATA / "datafile/realdata/10045976/C1_clean_xy.csv")
    return _std(np.log(d["X0"])), _std(np.log(d["Y0"]))        # algue(proie), rotif.(préd.)


def load_sp500():
    d = pd.read_csv(DATA / "datafile/realdata/sv_sp500/sv_train.csv")
    return _std(d["X0"]), _std(d["Y0"])


def load_wind():
    d = pd.read_csv(DATA / "samples/windfarms/site1_202210_Month_586_norm.csv")
    return _std(d["ActivePower_KWh"]), _std(d["WindSpeed"])


def phase_randomize(y, rng):
    """Surrogate préservant le spectre (donc l'autocovariance) de y mais randomisant
    sa phase — détruit le couplage croisé avec l'autre série (contrôle négatif)."""
    y = np.asarray(y, float)
    m = y.mean()
    Y = np.fft.rfft(y - m)
    ph = np.exp(1j * rng.uniform(0, 2 * np.pi, Y.shape[0]))
    ph[0] = 1.0
    if len(y) % 2 == 0:
        ph[-1] = 1.0
    return np.fft.irfft(Y * ph, n=len(y)) + m


# --------------------------------------------------------------------------- #
#  learning: couple coefficients + held-out one-step MSE (chemostat)          #
# --------------------------------------------------------------------------- #
def learning_chemostat(B, seed):
    algae, rot = load_chemostat()
    Z = np.column_stack([algae, rot])                 # X=algue, Y=rotifère ; Axy = rotif.->algue
    A, _, _ = fit_full(Z)
    axy, ayy = A[0, 1], A[1, 1]
    # IC bootstrap par blocs mobiles sur Axy, Ayy
    rng = np.random.default_rng(seed)
    T = len(Z) - 1
    L = max(10, int(round(len(Z) ** (1 / 3) * 2)))
    axys, ayys = [], []
    for _ in range(B):
        idx = []
        while len(idx) < T:
            s = int(rng.integers(0, T - L + 1))
            idx.extend(range(s, s + L))
        idx = np.array(idx[:T])
        Zb = np.vstack([Z[idx], Z[idx[-1] + 1]])
        Ab, _, _ = fit_full(Zb)
        axys.append(Ab[0, 1]); ayys.append(Ab[1, 1])
    ci = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    # held-out one-step-ahead MSE of X: couple (full) vs classical ablation (Axy=0)
    cut = int(0.8 * len(Z))
    Ztr, Zte = Z[:cut], Z[cut - 1:]
    Af, _, _ = fit_full(Ztr)
    Ar, _, _ = fit_restricted(Ztr)
    xte1 = Zte[1:, 0]
    mse_c = float(np.mean((xte1 - (Zte[:-1] @ Af.T)[:, 0]) ** 2))
    mse_k = float(np.mean((xte1 - (Zte[:-1] @ Ar.T)[:, 0]) ** 2))
    return dict(axy=float(axy), ayy=float(ayy), axy_ci=ci(axys), ayy_ci=ci(ayys),
                mse_couple=mse_c, mse_classical=mse_k)


# --------------------------------------------------------------------------- #
def main(B=500, seed=1):
    algae, rot = load_chemostat()
    x0, y0 = load_sp500()
    power, speed = load_wind()

    tests = [
        ("chemostat: rotif.$\\to$algae", np.column_stack([algae, rot])),
        ("chemostat: algae$\\to$rotif.", np.column_stack([rot, algae])),
        ("S&P: return$\\to$volat.",    np.column_stack([x0, y0])),
        ("S&P: volat.$\\to$return",    np.column_stack([y0, x0])),
        ("wind: speed$\\to$power",       np.column_stack([power, speed])),
        ("wind: power$\\to$speed",       np.column_stack([speed, power])),
        # negative control: circularly shifting the driver keeps each marginal (and its
        # autocorrelation) but destroys the cross-coupling -> the test must keep H0.
        ("control: shifted rotif.$\\to$algae",
         np.column_stack([algae, np.roll(rot, len(rot) // 2)])),
    ]
    rows, chemo_stat, chemo_nulls = [], None, None
    print(f"{'test':32s} {'N':>5s} {'Lambda':>9s} {'p_chi2':>10s} {'p_surr':>8s}  verdict")
    for i, (lab, Z) in enumerate(tests):
        stat, ps, pc, nulls = detect(Z, B=B, seed=seed + i)
        rows.append((lab, len(Z), stat, pc, ps))
        v = "REJECT" if ps < 0.05 else "keep"
        print(f"{lab:32s} {len(Z):5d} {stat:9.2f} {pc:10.2e} {ps:8.3f}  {v}")
        if lab.startswith("chemostat: rotif"):
            chemo_stat, chemo_nulls = stat, nulls

    lrn = learning_chemostat(B=B, seed=seed)
    print("\n[learning — chemostat, couple VAR]")
    print(f"  A_xy (rotif.->algae) = {lrn['axy']:+.3f}  95% CI {tuple(round(v,3) for v in lrn['axy_ci'])}")
    print(f"  A_yy (obs. memory)   = {lrn['ayy']:+.3f}  95% CI {tuple(round(v,3) for v in lrn['ayy_ci'])}")
    print(f"  held-out 1-step MSE(X): couple={lrn['mse_couple']:.4f}  classical(Axy=0)={lrn['mse_classical']:.4f}"
          f"  ({100*(lrn['mse_classical']/lrn['mse_couple']-1):+.1f}% vs couple)")

    # ---------------- figure ----------------
    # Single panel: the per-system Lambda bar chart that used to sit alongside merely
    # restated the paper's Table V, so only the surrogate-null panel is kept.
    fig, ax1 = plt.subplots(1, 1, figsize=(3.5, 2.7))

    # chemostat surrogate null (coupling destroyed) vs χ²₁ + observed Λ
    xx = np.linspace(0, 14, 400)
    ax1.hist(chemo_nulls, bins=30, range=(0, 14), density=True, color=KEEP, alpha=0.45,
             label="surrogate null $\\Lambda$")
    ax1.plot(xx, chi2.pdf(xx, 1), "k-", lw=1.4, label="$\\chi^2_1$")
    ax1.axvline(chi2.ppf(0.95, 1), ls=":", color="k", lw=1)
    ax1.annotate(f"observed\n$\\Lambda={chemo_stat:.0f}$ (reject)",
                 xy=(13.4, 0.03), xytext=(6.0, 0.30), fontsize=8, color=REJECT,
                 arrowprops=dict(arrowstyle="->", color=REJECT, lw=1.2))
    ax1.set(xlim=(0, 14), ylim=(0, 0.6), xlabel="$\\Lambda$", ylabel="density",
            title="Chemostat: surrogate null vs observed")
    ax1.legend(fontsize=7, loc="upper right")

    fig.tight_layout()
    figdir = Path(__file__).resolve().parents[1] / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir / "em_realdata.pdf")
    fig.savefig(figdir / "em_realdata_preview.png", dpi=150)
    print(f"\nfigure written to {figdir / 'em_realdata.pdf'}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    main(B=args.B, seed=args.seed)
