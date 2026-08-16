"""Out-of-class oscillator experiment (paper Fig. 8): back-action buys complex-pole
approximation power. TRUE system is 3rd-order (out of BOTH 2-D VAR(1) classes): a damped
oscillator (position y observed, velocity x latent) with AR(1) correlated forcing. We fit,
by exact y-marginal MLE, four nested 2-D pairwise VAR(1) models and compare recovered poles
+ held-out 1-step y-prediction, over many realizations (mean +/- s.e., paired DM test).
Self-contained. Prints stats; --fig writes backaction_poles.pdf."""
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import t as student_t


# ---- true 3rd-order damped oscillator: state (p=position, v=velocity, u=AR1 forcing) ----
def true_system(poles, ar_phi=0.90, ar_sig=0.20):
    """Build the true 3rd-order system so the (p,v) block has the given eigenvalues
    ``poles`` (a complex-conjugate pair -> underdamped, or two reals -> overdamped).
    Companion form: M2=[[1,wdt],[axy,axx]] with axy=-wdt^2 (position drives velocity =
    back-action). trace=1+axx=Re(l1+l2); det=l1 l2=axx+wdt^3 => wdt^3=(1-l1)(1-l2).
    Observation: position p; latent: velocity v; forcing: AR(1) u driving v (3rd order)."""
    l1, l2 = poles
    axx = float(np.real(l1 + l2) - 1.0)
    wdt = float(np.real((1 - l1) * (1 - l2)) ** (1.0 / 3.0))
    axy = -wdt * wdt
    F = np.array([[1.0, wdt, 0.0],         # p_{t+1} = p + wdt*v
                  [axy, axx, wdt],          # v_{t+1} = axy*p + axx*v + wdt*u  (u = forcing)
                  [0.0, 0.0, ar_phi]])      # u_{t+1} = ar_phi*u + noise (AR1 forcing)
    L = np.zeros((3, 3)); L[2, 2] = ar_sig * np.sqrt(1 - ar_phi**2)
    M2 = np.array([[1.0, wdt], [axy, axx]])                       # (p,v) block
    return F, L, np.linalg.eigvals(M2)

def simulate(F, L, N, rng, burn=300):
    z = np.zeros(3); Y = np.empty(N)
    for t in range(N + burn):
        z = F @ z + L @ rng.standard_normal(3)
        if t >= burn:
            Y[t - burn] = z[0]                                   # observe position p
    return Y

def simulate_couple2(A, Q, N, rng, burn=300):
    """IN-CLASS control: a true 2-D couple Z=(X,Y) with A^xy=0 (triangular, real poles)
    and white noise -> the classical model is exactly correct, so back-action buys ~0."""
    Lc = np.linalg.cholesky(Q)
    z = rng.multivariate_normal(np.zeros(2), stat_cov(A, Q))
    Y = np.empty(N)
    for t in range(N + burn):
        z = A @ z + Lc @ rng.standard_normal(2)
        if t >= burn:
            Y[t - burn] = z[1]                                   # observe Y
    return Y

# ---- 2-D pairwise couple VAR(1), Y observed exactly: exact y-marginal Kalman filter ----
def _unpack(theta, free_axy, free_qxy):
    axx, ayx, ayy = theta[0], theta[1], theta[2]
    axy = theta[3] if free_axy else 0.0
    i = 3 + (1 if free_axy else 0)
    l11, l22 = np.exp(theta[i]), np.exp(theta[i + 1])
    l21 = theta[i + 2] if free_qxy else 0.0
    A = np.array([[axx, axy], [ayx, ayy]])
    L = np.array([[l11, 0.0], [l21, l22]]); Q = L @ L.T
    return A, Q

def stat_cov(A, Q):
    return np.linalg.solve(np.eye(4) - np.kron(A, A), Q.reshape(-1)).reshape(2, 2)

def yfilter(Y, A, Q):
    """Return (loglik, one-step predicted y).  X latent, Y=Z[1] observed exactly."""
    LOG2PI = np.log(2 * np.pi)
    try:
        P = stat_cov(A, Q)
    except np.linalg.LinAlgError:
        return -np.inf, None
    if not np.all(np.isfinite(P)) or P[1, 1] <= 0:
        return -np.inf, None
    p11, p12, p22 = P[0, 0], 0.5 * (P[0, 1] + P[1, 0]), P[1, 1]
    mx = my = 0.0; ll = 0.0; yp = np.empty(len(Y))
    axx, axy, ayx, ayy = A[0, 0], A[0, 1], A[1, 0], A[1, 1]
    q11, q12, q22 = Q[0, 0], Q[0, 1], Q[1, 1]
    for n, yn in enumerate(Y):
        mxp = axx * mx + axy * my; myp = ayx * mx + ayy * my
        a11 = axx * p11 + axy * p12; a12 = axx * p12 + axy * p22
        a21 = ayx * p11 + ayy * p12; a22 = ayx * p12 + ayy * p22
        P11 = a11 * axx + a12 * axy + q11
        P12 = a11 * ayx + a12 * ayy + q12
        P22 = a21 * ayx + a22 * ayy + q22
        if P22 <= 0:
            return -np.inf, None
        yp[n] = myp; v = yn - myp
        ll += -0.5 * (LOG2PI + np.log(P22) + v * v / P22)
        Kx = P12 / P22
        mx = mxp + Kx * v; my = yn
        p11 = P11 - Kx * P12; p12 = P12 - Kx * P22; p22 = 0.0
    return ll, yp

def fit(Ytr, free_axy, free_qxy):
    npar = 3 + (1 if free_axy else 0) + 2 + (1 if free_qxy else 0)
    best = None
    for s in range(3):
        x0 = np.zeros(npar)
        x0[0] = 0.5; x0[1] = 0.3; x0[2] = 0.5
        rng = np.random.default_rng(s)
        x0 = x0 + 0.15 * rng.standard_normal(npar)
        x0[3 + (1 if free_axy else 0)] = np.log(0.3)      # l11
        x0[4 + (1 if free_axy else 0)] = np.log(0.3)      # l22
        with np.errstate(all="ignore"):
            r = minimize(lambda th: -yfilter(Ytr, *_unpack(th, free_axy, free_qxy))[0],
                         x0, method="Nelder-Mead",
                         options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 4000})
        if best is None or r.fun < best.fun:
            best = r
    return _unpack(best.x, free_axy, free_qxy)

def heldout_mse(Yte, A, Q):
    _ll, yp = yfilter(Yte, A, Q)
    if yp is None:
        return np.nan, None
    err = Yte - yp
    return float(np.mean(err[1:] ** 2)), err[1:]

MODELS = {  # (free_axy, free_qxy)
    "classical": (False, False),   # A^xy=0, R^xy=0
    "Rxy_only": (False, True),     # A^xy=0, R^xy free
    "Axy_only": (True, False),     # A^xy free, R^xy=0
    "couple": (True, True),        # both free
}

def run(sim_fn, true_poles, label, M=60, N=600, split=0.7, seed0=0):
    ntr = int(N * split)
    poles_c, poles_cl = [], []
    mse = {k: [] for k in MODELS}
    err = {k: [] for k in MODELS}
    ncomplex_c = ncomplex_cl = 0
    for m in range(M):
        rng = np.random.default_rng(seed0 + m)
        Y = sim_fn(N, rng)
        Ytr, Yte = Y[:ntr], Y[ntr:]
        fits = {k: fit(Ytr, *MODELS[k]) for k in MODELS}
        for k in MODELS:
            a, q = fits[k]
            mm, ee = heldout_mse(Yte, a, q)
            mse[k].append(mm); err[k].append(ee)
        ev_c = np.linalg.eigvals(fits["couple"][0])
        ev_cl = np.linalg.eigvals(fits["classical"][0])
        poles_c.append(ev_c); poles_cl.append(ev_cl)
        if np.max(np.abs(ev_c.imag)) > 1e-6:
            ncomplex_c += 1
        if np.max(np.abs(ev_cl.imag)) > 1e-6:
            ncomplex_cl += 1
    mse = {k: np.array(v) for k, v in mse.items()}
    base = mse["classical"]
    def gain(k):
        g = 100 * (base - mse[k]) / base
        return g.mean(), g.std(ddof=1) / np.sqrt(len(g))
    # Diebold-Mariano (paired, couple vs classical) on per-realization squared-error loss
    d = base - mse["couple"]
    dm_t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    dm_p = 2 * student_t.sf(abs(dm_t), len(d) - 1)
    print(f"\n=== {label} (M={M}, N={N}) ===")
    print(f"  true (p,v) poles: {true_poles[0]:.3f}")
    pc = np.concatenate(poles_c); pcl = np.concatenate(poles_cl)
    print(f"  couple  fitted poles: complex in {ncomplex_c}/{M} runs; "
          f"mean |Im| = {np.abs(pc.imag).mean():.3f}, mean pole ~ {pc.mean():.3f}")
    print(f"  classical fitted poles: complex in {ncomplex_cl}/{M} runs; "
          f"mean = {pcl.real.mean():.3f} (real)")
    for k in ("Rxy_only", "Axy_only", "couple"):
        g, se = gain(k)
        print(f"  held-out y-pred gain, {k:>9}: {g:5.1f}% +/- {se:.1f}")
    print(f"  Diebold-Mariano couple vs classical: t={dm_t:.2f}, p={dm_p:.1e}")
    return {"label": label, "true_poles": true_poles, "poles_c": poles_c,
                "poles_cl": poles_cl, "mse": mse, "ncomplex_c": ncomplex_c, "ncomplex_cl": ncomplex_cl,
                "gains": {k: gain(k) for k in ("Rxy_only", "Axy_only", "couple")},
                "dm_t": dm_t, "dm_p": dm_p, "M": M}


OUT = str(Path(__file__).resolve().parents[1] / "figures")
Path(OUT).mkdir(parents=True, exist_ok=True)


def make_figure(data):
    import matplotlib as mpl
    mpl.use("Agg")
    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.facecolor": "white",
        "savefig.bbox": "tight", "font.size": 8, "axes.titlesize": 8.5,
        "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 6.5, "lines.linewidth": 1.3, "lines.markersize": 4})
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.0))
    # fitted poles vs true (the held-out gains ug/og are reported in the paper text)
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(th), np.sin(th), color="0.8", lw=0.8, zorder=1)
    uc = data["uc"].ravel(); ucl = data["ucl"].ravel(); tp = np.atleast_1d(data["true"])
    ax.scatter(ucl.real, ucl.imag, s=16, marker="s", color="tab:orange", alpha=0.55,
               label="classical", zorder=2)
    ax.scatter(uc.real, uc.imag, s=14, color="tab:blue", alpha=0.45,
               label="pairwise", zorder=3)
    ax.scatter(tp.real, tp.imag, s=70, marker="x", color="k", lw=1.8,
               label="true poles", zorder=4)
    ax.axhline(0, color="0.7", lw=0.5)
    ax.set_xlabel("Re"); ax.set_ylabel("Im"); ax.set_aspect("equal", "box")
    ax.set_xlim(0.2, 1.05); ax.set_ylim(-0.55, 0.55)
    ax.set_title("fitted transition poles")
    # Upper-left quadrant is empty ("lower left" used to sit on the lower true-pole cross);
    # labels name the series only -- real vs complex is the finding, stated in the caption.
    ax.legend(loc="upper left", fontsize=6, framealpha=0.85, borderpad=0.3,
              labelspacing=0.25, handletextpad=0.35, borderaxespad=0.25, markerscale=0.8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT + "/backaction_poles.pdf")
    fig.savefig(OUT + "/backaction_poles_preview.png", dpi=150)
    print("figure written to", OUT + "/backaction_poles.pdf")


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    if "--replot" in args:
        d = np.load(OUT + "/backaction_oscillator_data.npz", allow_pickle=True)
        make_figure(d)
    else:
        M = next((int(a) for a in args if a.isdigit()), 40)
        Fo, Lo, osc_poles = true_system(np.array([0.74 + 0.32j, 0.74 - 0.32j]))
        under = run(lambda N, rng: simulate(Fo, Lo, N, rng), osc_poles,
                    "out-of-class oscillator (has back-action)", M=M)
        A_in = np.array([[0.6, 0.0], [0.3, 0.5]])            # A^xy=0 (triangular) -> in-class
        Q_in = np.array([[0.10, 0.0], [0.0, 0.10]])          # white noise (R^xy=0)
        inclass = run(lambda N, rng: simulate_couple2(A_in, Q_in, N, rng),
                      np.linalg.eigvals(A_in), "in-class control (no back-action)", M=M)
        np.savez(OUT + "/backaction_oscillator_data.npz",
                 true=np.asarray(under["true_poles"]),
                 uc=np.asarray(under["poles_c"]), ucl=np.asarray(under["poles_cl"]),
                 ug=np.asarray(under["gains"]["couple"]), og=np.asarray(inclass["gains"]["couple"]),
                 M=M)
        make_figure(np.load(OUT + "/backaction_oscillator_data.npz", allow_pickle=True))
