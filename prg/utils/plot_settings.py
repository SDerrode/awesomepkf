#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Configuration globale du style matplotlib.
À importer AVANT toute création de figures.

Ce module modifie ``matplotlib.rcParams`` sans importer ``pyplot``,
ce qui évite tout effet de bord GUI au moment de l'import.
"""

# FIX : matplotlib importé à la place de pyplot — pas d'initialisation GUI à l'import
import matplotlib

# --------------------------------------------------
# Fenêtre d'affichage par défaut
# (utilisée par les scripts de plot, non appliquée à rcParams)
# --------------------------------------------------
WINDOW = {"xmin": 6000, "xmax": 7000}

# --------------------------------------------------
# Paramètres généraux
# FIX : DPI et FACECOLOR maintenant appliqués à rcParams
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

matplotlib.rc("font", size=SMALL_SIZE)  # taille de texte par défaut
matplotlib.rc("axes", titlesize=SMALL_SIZE)  # titre des axes
matplotlib.rc("axes", labelsize=MEDIUM_SIZE)  # labels x et y
matplotlib.rc("xtick", labelsize=SMALL_SIZE)  # labels des ticks x
matplotlib.rc("ytick", labelsize=SMALL_SIZE)  # labels des ticks y
matplotlib.rc("legend", fontsize=SMALL_SIZE)  # légende
matplotlib.rc("figure", titlesize=BIG_SIZE)  # titre de la figure
