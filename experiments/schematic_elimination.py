#!/usr/bin/env python
"""Schematic of the three block-elimination orders of the block-tridiagonal matrix J
(Proposition 1 of the smoothing letter): top-down (RTS, BF, MBF, VAR), bottom-up (DWY)
and twisted (2F, the two eliminations meeting at block n).

Run (from the letter directory): python experiments/schematic_elimination.py
Output: figures/elimination_orders.pdf (+ .png)
"""
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "figures" / "elimination_orders"

avail = {f.name for f in fm.fontManager.ttflist}
family = next((f for f in ("Arial", "Helvetica", "Liberation Sans") if f in avail), "DejaVu Sans")
mpl.rcParams.update({"pdf.fonttype": 42, "font.family": "sans-serif", "font.sans-serif": [family],
                     "mathtext.fontset": "stixsans", "font.size": 8,
                     "savefig.facecolor": "white", "figure.facecolor": "white"})

NB, B = 6, 0.165                    # diagonal blocks shown, block size (in)
W, H = 3.5, 1.62
C_EL, C_BS = "#000000", "#777777"   # elimination, back-substitution (neutral: Fig. 1 uses colours)
C_DIAG, C_OFF, C_MEET = "#bdbdbd", "#e0e0e0", "#ffe08a"
TOP = 1.40
PANELS = [(0.10, "top-down", "RTS, BF, MBF, VAR", "down"),
          (1.22, "bottom-up", "DWY", "up"),
          (2.34, "twisted", "2F", "twist")]
MEET = 3

fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")


def arrow(p0, p1, col, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=7, color=col, lw=0.9,
                                 ls=ls, shrinkA=0, shrinkB=0, zorder=5))


for x0, name, who, mode in PANELS:
    for i in range(NB):
        for j in range(NB):
            if abs(i - j) > 1:
                continue
            fc = C_DIAG if i == j else C_OFF
            if mode == "twist" and i == j == MEET:
                fc = C_MEET
            ax.add_patch(Rectangle((x0 + j * B, TOP - (i + 1) * B), B, B, fc=fc, ec="white", lw=0.6))
    ax.add_patch(Rectangle((x0, TOP - NB * B), NB * B, NB * B, fill=False, ec="black", lw=0.6))

    def c(k):
        return x0 + (k + 0.5) * B, TOP - (k + 0.5) * B

    o = 0.10
    first, last = c(0), c(NB - 1)
    if mode == "down":
        arrow((first[0] + o, first[1] + o), (last[0] + o, last[1] + o), C_EL)
        arrow((last[0] - o, last[1] - o), (first[0] - o, first[1] - o), C_BS, "--")
    elif mode == "up":
        arrow((last[0] + o, last[1] + o), (first[0] + o, first[1] + o), C_EL)
        arrow((first[0] - o, first[1] - o), (last[0] - o, last[1] - o), C_BS, "--")
    else:
        m = c(MEET)
        g = 0.075            # both arrows on the same 45-degree line as the others,
        arrow((first[0] + o, first[1] + o), (m[0] + o - g, m[1] + o + g), C_EL)   # heads
        arrow((last[0] + o, last[1] + o), (m[0] + o + g, m[1] + o - g), C_EL)     # meet at n
        ax.text(m[0], m[1], r"$n$", ha="center", va="center", fontsize=7)
    ax.text(first[0], first[1], r"$0$", ha="center", va="center", fontsize=6.5, color="#333333")
    ax.text(last[0], last[1], r"$N$", ha="center", va="center", fontsize=6.5, color="#333333")
    ax.text(x0 + NB * B / 2, TOP - NB * B - 0.11, name, ha="center", va="center", fontsize=8,
            weight="bold")
    ax.text(x0 + NB * B / 2, TOP - NB * B - 0.25, who, ha="center", va="center", fontsize=7.5)

ax.plot([0.42, 0.62], [1.54, 1.54], color=C_EL, lw=0.9)
ax.text(0.66, 1.54, "elimination", fontsize=7.5, va="center")
ax.plot([1.72, 1.92], [1.54, 1.54], color=C_BS, lw=0.9, ls="--")
ax.text(1.96, 1.54, "back-substitution", fontsize=7.5, va="center")

OUT.parent.mkdir(exist_ok=True)
fig.savefig(str(OUT) + ".pdf")
fig.savefig(str(OUT) + ".png", dpi=300)
print(f"saved {OUT}.pdf/.png ({W} x {H} in, font {family})")
