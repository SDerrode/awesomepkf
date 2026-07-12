"""Fig (well-posedness/choice): why MBF is the numerically safest of the six smoothers.
Two self-contained panels on a stationary pairwise couple (p=q=2), as the predicted couple
covariance P_{n|n-1} is driven ill-conditioned by shrinking one process-noise eigenvalue:
 (a) condition numbers of the matrix each family inverts -- cond(S_n) [BF/MBF] is a principal
     submatrix of P_{n|n-1} [RTS], so cond(S_n) <= cond(P_{n|n-1}) (eigenvalue interlacing);
 (b) PSD preservation: the smallest eigenvalue of the updated covariance, computed in float32,
     from the Joseph/PSD form (MBF) vs a naive P - K S K^T form -- the naive form loses
     definiteness (negative eigenvalues) where the Joseph form does not.
Output: figures/mbf_conditioning.pdf"""
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib as mpl
mpl.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.facecolor": "white", "savefig.bbox": "tight",
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 6.8, "lines.linewidth": 1.4, "axes.axisbelow": True})
import matplotlib.pyplot as plt

OUT = str(Path(__file__).resolve().parents[1] / "figures"); Path(OUT).mkdir(parents=True, exist_ok=True)
p = q = 2; d = p + q

# stable couple transition A = [[Axx, Axy],[Ayx, Ayy]]
A = np.array([[0.60, 0.10, 0.20, 0.00],
              [0.00, 0.50, 0.10, 0.15],
              [0.30, 0.00, 0.40, 0.05],
              [0.00, 0.25, 0.00, 0.35]])
assert np.max(np.abs(np.linalg.eigvals(A))) < 1, "A must be stable"
H = np.zeros((q, d)); H[:, p:] = np.eye(q)            # observe the Y-block
# a fixed 2x2 rotation of the LATENT (X) block, so the shrinking-noise direction is a
# genuine unobserved latent direction (this is what drives P_{n|n-1} ill-conditioned as the
# process noise approaches the R_n>0 boundary the well-posedness theorem assumes).
th = 0.7; Ux = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
QY = np.array([[0.10, 0.02], [0.02, 0.08]])          # well-conditioned observed-block noise

def Qmat(kappa):
    Qx = 0.1 * (Ux @ np.diag([1.0, 1.0 / kappa]) @ Ux.T)   # near-singular latent noise
    Q = np.zeros((d, d)); Q[:p, :p] = Qx; Q[p:, p:] = QY
    return Q

def statcov(Q):
    return np.linalg.solve(np.eye(d * d) - np.kron(A, A), Q.reshape(-1)).reshape(d, d)

def predicted_cov(Q, n_it=8000, tol=1e-14):
    """Steady-state predicted couple covariance P_{n|n-1} (singular-obs Riccati)."""
    P = statcov(Q)
    for _ in range(n_it):
        S = H @ P @ H.T
        K = P @ H.T @ np.linalg.inv(S)
        Pnew = A @ (P - K @ H @ P) @ A.T + Q
        if np.max(np.abs(Pnew - P)) < tol:
            P = Pnew; break
        P = Pnew
    return P

rhos = np.logspace(0, 10, 40)                    # latent-noise conditioning kappa: 1 -> 1e10
cS, cP, cR, cSig = [], [], [], []
min_joseph, min_naive = [], []
for kap in rhos:
    Q = Qmat(kap)
    P = predicted_cov(Q)                          # P_{n|n-1} (float64 reference)
    S = H @ P @ H.T                               # what BF/MBF invert (q x q)
    Sig = statcov(Q)                              # what 2F/DWY invert (prior)
    cS.append(np.linalg.cond(S)); cP.append(np.linalg.cond(P))
    cR.append(np.linalg.cond(Q)); cSig.append(np.linalg.cond(Sig))
    # one measurement update in float32, two ways
    P32 = P.astype(np.float32); S32 = (H @ P32 @ H.T)
    K32 = (P32 @ H.T @ np.linalg.inv(S32)).astype(np.float32)
    IKH = (np.eye(d, dtype=np.float32) - K32 @ H)
    P_joseph = IKH @ P32 @ IKH.T                  # Joseph/PSD form (exact obs, R=0)
    P_naive = P32 - K32 @ S32 @ K32.T             # naive form
    min_joseph.append(float(np.min(np.linalg.eigvalsh((P_joseph + P_joseph.T) / 2))))
    min_naive.append(float(np.min(np.linalg.eigvalsh((P_naive + P_naive.T) / 2))))

cS, cP, cR, cSig = map(np.array, (cS, cP, cR, cSig))
min_joseph, min_naive = np.array(min_joseph), np.array(min_naive)
print(f"at kappa={rhos[-1]:.0e}: cond(S)={cS[-1]:.2e}  cond(P)={cP[-1]:.2e}  "
      f"cond(S)<=cond(P) throughout: {np.all(cS <= cP * (1 + 1e-9))}")
print(f"naive min-eig min={min_naive.min():.2e}; Joseph min-eig min={min_joseph.min():.2e} "
      f"(Joseph stays >=0: {np.all(min_joseph >= -1e-6)})")

fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
ax[0].loglog(rhos, cP, "-", color="C1", label=r"$\mathbf{P}_{n|n-1}$ (RTS)")
ax[0].loglog(rhos, cSig, "-.", color="C2", label=r"$\mathbf{\Sigma}_n$ (2F/DWY)")
ax[0].loglog(rhos, cR, ":", color="0.5", label=r"$\mathbf{R}_n$ (VAR)")
ax[0].loglog(rhos, cS, "-", color="C0", lw=2.2, label=r"$\mathbf{S}_n$ (BF/MBF)")
ax[0].set_xlabel(r"latent-noise conditioning $\kappa$")
ax[0].set_ylabel("cond. number of inverted matrix")
ax[0].set_title("(a) MBF inverts the best-conditioned block")
ax[0].legend(loc="upper left"); ax[0].grid(alpha=0.3, which="both")

ax[1].axhline(0, color="k", lw=0.8)
ax[1].semilogx(rhos, min_naive, "s-", color="C3", ms=3, label=r"naive $\mathbf{P}-\mathbf{KSK}^\top$")
ax[1].semilogx(rhos, min_joseph, "o-", color="C0", ms=3, label="Joseph / MBF form")
ax[1].set_xlabel(r"latent-noise conditioning $\kappa$")
ax[1].set_ylabel(r"min eigenvalue of update (float32)")
ax[1].set_title("(b) Joseph form stays positive semidefinite")
ax[1].legend(loc="lower left"); ax[1].grid(alpha=0.3, which="both")

fig.tight_layout()
fig.savefig(OUT + "/mbf_conditioning.pdf")
fig.savefig(OUT + "/mbf_conditioning_preview.png", dpi=150)
print("saved", OUT + "/mbf_conditioning.pdf")
