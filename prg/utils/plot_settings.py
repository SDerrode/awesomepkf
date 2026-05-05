"""
Global matplotlib style configuration.
Must be imported BEFORE any figure creation.

This module modifies ``matplotlib.rcParams`` without importing ``pyplot``,
which avoids any GUI side effects at import time.
"""

# FIX: matplotlib imported instead of pyplot — no GUI initialisation at import time
import matplotlib as mpl

# --------------------------------------------------
# Default display window
# (used by plot scripts, not applied to rcParams)
# --------------------------------------------------
WINDOW = {"xmin": 6000, "xmax": 7000}

# --------------------------------------------------
# General parameters
# FIX: DPI and FACECOLOR now applied to rcParams
# --------------------------------------------------
DPI = 150
FACECOLOR = "#AAAAAA"

mpl.rcParams["figure.dpi"] = DPI
mpl.rcParams["figure.facecolor"] = FACECOLOR

# --------------------------------------------------
# Cycle couleurs / styles
# FIX : matplotlib.cycler au lieu de plt.cycler
# --------------------------------------------------
mpl.rcParams["axes.prop_cycle"] = mpl.cycler(
    color=["r", "g", "b"], linestyle=["-", "--", ":"]
)

# --------------------------------------------------
# Tailles de police
# --------------------------------------------------
SMALL_SIZE = 6
MEDIUM_SIZE = 8
BIG_SIZE = 10

mpl.rc("font", size=SMALL_SIZE)  # default text size
mpl.rc("axes", titlesize=SMALL_SIZE)  # titre des axes
mpl.rc("axes", labelsize=MEDIUM_SIZE)  # labels x et y
mpl.rc("xtick", labelsize=SMALL_SIZE)  # labels des ticks x
mpl.rc("ytick", labelsize=SMALL_SIZE)  # labels des ticks y
mpl.rc("legend", fontsize=SMALL_SIZE)  # legend
mpl.rc("figure", titlesize=BIG_SIZE)  # titre de la figure
