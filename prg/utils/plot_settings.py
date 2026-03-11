#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Global matplotlib style configuration.
Must be imported BEFORE any figure creation.

This module modifies ``matplotlib.rcParams`` without importing ``pyplot``,
which avoids any GUI side effects at import time.
"""

# FIX: matplotlib imported instead of pyplot — no GUI initialisation at import time
import matplotlib

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

matplotlib.rcParams["figure.dpi"] = DPI
matplotlib.rcParams["figure.facecolor"] = FACECOLOR

# --------------------------------------------------
# Cycle couleurs / styles
# FIX : matplotlib.cycler au lieu de plt.cycler
# --------------------------------------------------
matplotlib.rcParams["axes.prop_cycle"] = matplotlib.cycler(
    color=["r", "g", "b"], linestyle=["-", "--", ":"]
)

# --------------------------------------------------
# Tailles de police
# --------------------------------------------------
SMALL_SIZE = 6
MEDIUM_SIZE = 8
BIG_SIZE = 10

matplotlib.rc("font", size=SMALL_SIZE)  # default text size
matplotlib.rc("axes", titlesize=SMALL_SIZE)  # titre des axes
matplotlib.rc("axes", labelsize=MEDIUM_SIZE)  # labels x et y
matplotlib.rc("xtick", labelsize=SMALL_SIZE)  # labels des ticks x
matplotlib.rc("ytick", labelsize=SMALL_SIZE)  # labels des ticks y
matplotlib.rc("legend", fontsize=SMALL_SIZE)  # legend
matplotlib.rc("figure", titlesize=BIG_SIZE)  # titre de la figure
