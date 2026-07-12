"""Graphical-model schematic of the Gaussian pairwise Markov model (paper Fig. 1 / Sec. II).
Left: classical state-space (latent Markov X, memoryless observation Y). Right: the PMM
couple Z=(X,Y) jointly Markov, adding the back-action A^xy (Y->X, highlighted) and
correlated process noise R^xy. Self-contained (matplotlib only). Output: figures/pmm_schematic.pdf"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
mpl.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.facecolor": "white",
    "savefig.bbox": "tight", "font.size": 8, "axes.titlesize": 9})
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

OUT = str(Path(__file__).resolve().parents[1] / "figures")
Path(OUT).mkdir(parents=True, exist_ok=True)

R = 0.26
POS = {"Xn": (0, 1), "Xn1": (2, 1), "Yn": (0, 0), "Yn1": (2, 0)}
LBL = {"Xn": r"$\mathbf{X}_n$", "Xn1": r"$\mathbf{X}_{n+1}$",
       "Yn": r"$\mathbf{Y}_n$", "Yn1": r"$\mathbf{Y}_{n+1}$"}


def node(ax, key, latent):
    x, y = POS[key]
    fc = "#e8e8e8" if latent else "white"
    ax.add_patch(Circle((x, y), R, fc=fc, ec="0.25", lw=1.1, zorder=3))
    ax.text(x, y, LBL[key], ha="center", va="center", fontsize=8.5, zorder=4)


def edge(ax, a, b, color="0.25", lw=1.2, label=None, lab_off=(0, 0), style="-|>",
         rad=0.0, ls="-"):
    (x0, y0), (x1, y1) = POS[a], POS[b]
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), shrinkA=15,
                 shrinkB=15, arrowstyle=style, mutation_scale=11,
                 color=color, lw=lw, ls=ls,
                 connectionstyle=f"arc3,rad={rad}", zorder=2, patchA=None, patchB=None))
    if label:
        mx, my = (x0 + x1) / 2 + lab_off[0], (y0 + y1) / 2 + lab_off[1]
        ax.text(mx, my, label, ha="center", va="center", fontsize=7.5,
                color=color, zorder=5,
                bbox=dict(boxstyle="round,pad=0.05", fc="white", ec="none", alpha=0.85))


def frame(ax, title):
    ax.set_xlim(-0.7, 2.7); ax.set_ylim(-0.5, 1.5)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=9, pad=3)


fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.0, 2.05))

# ---- (a) classical state-space ----
for k, lat in [("Xn", True), ("Xn1", True), ("Yn", False), ("Yn1", False)]:
    node(axL, k, lat)
edge(axL, "Xn", "Xn1", label=r"$\mathbf{A}^{xx}$", lab_off=(0, 0.16))
edge(axL, "Xn", "Yn", label=None)
edge(axL, "Xn1", "Yn1", label=None)
frame(axL, "(a) classical state space")

# ---- (b) PMM couple ----
for k, lat in [("Xn", True), ("Xn1", True), ("Yn", False), ("Yn1", False)]:
    node(axR, k, lat)
edge(axR, "Xn", "Xn1", label=r"$\mathbf{A}^{xx}$", lab_off=(0, 0.16))
edge(axR, "Yn", "Yn1", label=r"$\mathbf{A}^{yy}$", lab_off=(0, -0.16))
edge(axR, "Xn", "Yn1", label=r"$\mathbf{A}^{yx}$", lab_off=(0.0, -0.22), rad=0.0)
edge(axR, "Yn", "Xn1", color="#d62728", lw=2.0, label=r"$\mathbf{A}^{xy}$ (back-action)",
     lab_off=(0.0, 0.22), rad=0.0)
# correlated process noise between the two n+1 states
edge(axR, "Xn1", "Yn1", color="0.45", lw=1.0, style="<|-|>", ls=(0, (4, 2)),
     label=r"$\mathbf{R}^{xy}$", lab_off=(0.34, 0.0), rad=0.0)
frame(axR, "(b) pairwise Markov model $\\mathbf{Z}=(\\mathbf{X},\\mathbf{Y})$")

fig.tight_layout()
fig.savefig(OUT + "/pmm_schematic.pdf")
fig.savefig(OUT + "/pmm_schematic_preview.png", dpi=150)
print("saved", OUT + "/pmm_schematic.pdf")
