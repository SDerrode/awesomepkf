"""Estimability vs testability, related to Petetin-Desbouvries (IEEE TSP 62(14), 2014, Eq. 71)
-- paper Fig. 10, Sec. V (When back-action is worth modelling). Scalar AR(1)+observation model.
(a) Petetin's per-step KL (their Eq. 71, approximation/estimability) rises with the noise
    ratio R/Q, while the paper's spectral KL_rate (LRT noncentrality lambda/2N, testability)
    is identically zero on THEIR couple (its back-action sits in the A^yx=0 gauge -> y stays
    exact AR(1) -> the y-marginal test is powerless).
(b) Opening a forward path A^yx (at fixed R/Q, back-action fixed) makes the back-action leave
    a y-footprint and the paper's KL_rate rises from zero -> the two are different quantities.
Uses the paper's OWN _kl_rate / y_spectrum functional (identical to em_lrt.py). Self-contained
(numpy/scipy/matplotlib). Output: figures/petetin_kl_comparison.pdf
"""
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
mpl.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.facecolor": "white", "savefig.bbox": "tight",
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 7, "lines.linewidth": 1.6, "lines.markersize": 3.5})
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / "figures"
OUT.mkdir(parents=True, exist_ok=True)
WGRID = np.linspace(-np.pi, np.pi, 4096, endpoint=False)
Z = np.exp(-1j * WGRID)
ORANGE, BLUE, GREEN, GREY = "#E69F00", "#0072B2", "#009E73", "#999999"


def y_spectrum(A, Qc):
    """Scalar spectral density of Y=Z[1] for Z_k = A Z_{k-1} + w, cov(w)=Qc (paper's formula)."""
    axx, axy, ayx, ayy = A[0, 0], A[0, 1], A[1, 0], A[1, 1]
    det = (1 - axx * Z) * (1 - ayy * Z) - axy * ayx * Z * Z
    aa, bb = ayx * Z, 1 - axx * Z
    num = (Qc[0, 0] * np.abs(aa) ** 2 + Qc[1, 1] * np.abs(bb) ** 2
           + 2 * Qc[0, 1] * np.real(aa * np.conj(bb)))
    return num / np.abs(det) ** 2


def kl_rate(f, g):
    """(1/4pi) INT [f/g - 1 - ln(f/g)] dω  == mean(...)/2   (paper's _kl_rate)."""
    r = f / g
    return float(np.mean(r - 1.0 - np.log(r)) * 0.5)


def petetin_couple(a, b, Q, R):
    """Petetin Eq.(69-70) with constraint (54) d=a and KL-optimal c=abQ/(R+b^2 Q). Note A^yx=0."""
    c = a * b * Q / (R + b ** 2 * Q)
    sig2_c = R * (1 - a ** 2) + b ** 2 * Q
    Qc = np.array([[Q - c ** 2 * R, b * Q - c * a * R], [b * Q - c * a * R, sig2_c]])
    A = np.array([[a, -b * c], [0.0, a]])
    return A, Qc


def paper_kl(A_couple, Qc):
    """Paper's KL_rate: couple y-spectrum vs its A^xy=0 projection (A^yy free; A^xx,A^yx,Q frozen)."""
    a, ayx = A_couple[0, 0], A_couple[1, 0]
    S = y_spectrum(A_couple, Qc)
    r = minimize_scalar(lambda nu: kl_rate(S, y_spectrum(np.array([[a, 0.0], [ayx, nu]]), Qc)),
                        bounds=(-0.98, 0.98), method="bounded", options={"xatol": 1e-10})
    return max(r.fun, 0.0)


def main():
    a, b, R = 0.8, 1.0, 1.0

    # panel (a): both KL along the noise ratio R/Q, on Petetin's couple
    Qs = np.geomspace(0.05, 20, 40)
    ratio = R / Qs
    p71 = [-0.5 * np.log(1 - a ** 2 * R / (R + b ** 2 * Q)) for Q in Qs]
    kl_pap = [paper_kl(*petetin_couple(a, b, Q, R)) for Q in Qs]

    # panel (b): paper's KL_rate as A^yx opens a y-footprint, at R/Q=1, back-action fixed
    A_co1, Qc1 = petetin_couple(a, b, 1.0, R)
    axy = A_co1[0, 1]
    ayx_grid = np.linspace(0.0, 0.6, 40)
    kl_vs_ayx = [paper_kl(np.array([[a, axy], [ayx, a]]), Qc1) for ayx in ayx_grid]

    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
    # Cite by author, never by bibliography number: the figure is generated outside LaTeX,
    # so a hardcoded "[n]" silently goes stale as soon as the references are renumbered.
    ax[0].semilogx(ratio, p71, "-o", color=ORANGE,
                   label=r"Petetin & Desbouvries: approximation (estimability)")
    ax[0].semilogx(ratio, kl_pap, "-s", color=BLUE,
                   label=r"spectral $\mathrm{KL_{rate}}=\lambda/2N$: testability")
    ax[0].axhline(0, color=GREY, lw=0.8, ls=":")
    ax[0].set_xlabel(r"noise ratio $R/Q$")
    ax[0].set_ylabel("KL divergence (per step / rate)")
    ax[0].set_title(r"(a) two KL functionals on Petetin's couple ($A^{yx}{=}0$)")
    ax[0].legend(loc="upper left"); ax[0].grid(alpha=0.3, which="both")
    ax[0].annotate("test powerless:\nno $y$-footprint", xy=(6, 0.004),
                   xytext=(1.1, 0.16), fontsize=6.5, color=BLUE,
                   arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.8))

    ax[1].plot(ayx_grid, kl_vs_ayx, "-D", color=GREEN)
    ax[1].axvline(0, color=ORANGE, lw=1.0, ls="--")
    ax[1].set_xlabel(r"forward path $A^{yx}$ (at $R/Q{=}1$, back-action $A^{xy}$ fixed)")
    ax[1].set_ylabel(r"spectral $\mathrm{KL_{rate}}$ (testability)")
    ax[1].set_title(r"(b) back-action becomes testable only via a $\mathbf{y}$-footprint")
    ax[1].annotate("Petetin couple\n$A^{yx}{=}0$", xy=(0.006, 0.006),
                   xytext=(0.06, 0.33), fontsize=6.5, color=ORANGE,
                   arrowprops=dict(arrowstyle="->", color=ORANGE, lw=0.8))
    ax[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "petetin_kl_comparison.pdf")
    fig.savefig(OUT / "petetin_kl_comparison_preview.png", dpi=150)
    print("saved", OUT / "petetin_kl_comparison.pdf")
    print(f"(a) at R/Q={ratio[0]:.2f}: Petetin={p71[0]:.3f}  paperKL={kl_pap[0]:.2e}")
    print(f"(b) paper KL_rate: A^yx=0 -> {kl_vs_ayx[0]:.2e} ; A^yx=0.6 -> {kl_vs_ayx[-1]:.4f}")


if __name__ == "__main__":
    main()
