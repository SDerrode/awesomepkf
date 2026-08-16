"""Fig (Sec. II-D): the six smoothers as different elimination orders / factorizations of
ONE block-tridiagonal system J x = eta. A matplotlib schematic (no data), complementary to
Table I: it shows the shared linear system on the left and the six solvers grouped by family
(forward-filter only / needs prior marginal Sigma_n / batch), each tagged with the matrix it
inverts and its dimension, and a glyph for its sweep pattern.
Output: figures/schematic_smoothers.pdf"""
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
mpl.rcParams.update({"savefig.dpi": 300, "savefig.facecolor": "white", "savefig.bbox": "tight",
                     "font.size": 8})
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = str(Path(__file__).resolve().parents[1] / "figures"); Path(OUT).mkdir(parents=True, exist_ok=True)

# Okabe-Ito pale group tints (colour-blind safe)
T_FWD, T_PRIOR, T_BATCH = "#D7ECF6", "#FCE7D2", "#D9F0E3"
EDGE = "#333333"

fig, ax = plt.subplots(figsize=(7.0, 2.35))
ax.set_xlim(0, 100); ax.set_ylim(0, 42); ax.axis("off")

def box(x, y, w, h, title, sublines, fill, tfs=9, sfs=6.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.6",
                                linewidth=1.0, edgecolor=EDGE, facecolor=fill))
    ax.text(x + w / 2, y + h * 0.70, title, ha="center", va="center", fontsize=tfs, fontweight="bold")
    for i, ln in enumerate(sublines):
        ax.text(x + w / 2, y + h * 0.40 - i * h * 0.24, ln, ha="center", va="center",
                fontsize=sfs, color="#222")

# ---- central shared system: block-tridiagonal chain + label ----
ax.text(15, 39, r"one block-tridiagonal system", ha="center", fontsize=8.5, fontweight="bold")
ax.text(15, 34.7, r"$\mathbf{J}\,\mathbf{x}=\boldsymbol{\eta}$   (Sec. II-D)", ha="center", fontsize=9)
cy = 22
for i in range(4):                                  # sketch the chain of blocks
    ax.add_patch(FancyBboxPatch((4 + i * 6.3, cy), 5.0, 5.0, boxstyle="round,pad=0.2,rounding_size=0.8",
                                linewidth=0.9, edgecolor=EDGE, facecolor="#EEEEEE"))
    ax.text(6.5 + i * 6.3, cy + 2.5, [r"$\mathbf{x}_0$", r"$\mathbf{x}_1$", r"$\cdots$", r"$\mathbf{x}_N$"][i],
            ha="center", va="center", fontsize=7.5)
    if i < 3:
        ax.plot([9 + i * 6.3, 10.3 + i * 6.3], [cy + 2.5, cy + 2.5], color=EDGE, lw=1.1)
ax.text(15, cy - 3.2, "tridiagonal coupling", ha="center", fontsize=6.4, color="#555")

# ---- fan-out arrow ----
ax.annotate("", xy=(34, 21), xytext=(30, 21),
            arrowprops={"arrowstyle": "-|>", "lw": 1.4, "color": EDGE})
ax.text(32, 24.8, "six elimination", ha="center", fontsize=6.6, color="#444")
ax.text(32, 22.4, "orders", ha="center", fontsize=6.6, color="#444")

# ---- six solver boxes: 3 columns x 2 rows, grouped by family ----
W, H = 19.0, 12.5
xs = [37, 58, 79]; ytop, ybot = 23.5, 6
box(xs[0], ytop, W, H, "RTS", [r"inverts $\mathbf{P}_{n|n-1}$  ($p{+}q$)", "forward $\\to$ back-sub."], T_FWD)
box(xs[1], ytop, W, H, "BF",  [r"inverts $\mathbf{S}_n$  ($q$)", "forward adjoint"], T_FWD)
box(xs[2], ytop, W, H, "MBF", [r"inverts $\mathbf{S}_n$  ($q$)", "PSD -- safe default"], T_FWD)
box(xs[0], ybot, W, H, "2F",  [r"inverts $\boldsymbol{\Sigma}_n$", "two filters, fused"], T_PRIOR)
box(xs[1], ybot, W, H, "DWY", [r"inverts $\boldsymbol{\Sigma}_n$", "backward RTS"], T_PRIOR)
box(xs[2], ybot, W, H, "VAR", [r"inverts $\mathbf{R}_n$", "single batch solve"], T_BATCH)

# group labels
ax.text((xs[0] + xs[2] + W) / 2, ytop + H + 1.8, "forward filter only", ha="center",
        fontsize=6.8, style="italic", color="#2779a7")
ax.text((xs[0] + xs[1] + W) / 2, ybot - 2.6, r"need prior marginal $\boldsymbol{\Sigma}_n$",
        ha="center", fontsize=6.8, style="italic", color="#c07a1e")
ax.text(xs[2] + W / 2, ybot - 2.6, "batch / EM stats", ha="center", fontsize=6.8,
        style="italic", color="#1f8a5b")

fig.tight_layout()
fig.savefig(OUT + "/schematic_smoothers.pdf")
fig.savefig(OUT + "/schematic_smoothers_preview.png", dpi=150)
print("saved", OUT + "/schematic_smoothers.pdf")
