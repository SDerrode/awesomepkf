#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Configuration globale du style matplotlib.
À importer AVANT toute création de figures.
"""

import matplotlib.pyplot as plt

# --------------------------------------------------
# Fenêtre d'affichage par défaut
# --------------------------------------------------
WINDOW = {'xmin': 50, 'xmax': 250}

# --------------------------------------------------
# Paramètres généraux
# --------------------------------------------------
DPI        = 150
FACECOLOR = '#AAAAAA'

# --------------------------------------------------
# Cycle couleurs / styles
# --------------------------------------------------
plt.rcParams['axes.prop_cycle'] = plt.cycler(
    color=['r', 'g', 'b'],
    linestyle=['-', '--', ':']
)

# --------------------------------------------------
# Tailles de police
# --------------------------------------------------
SMALL_SIZE  = 6
MEDIUM_SIZE = 8
BIG_SIZE    = 10

plt.rc('font',   size      = SMALL_SIZE)
plt.rc('axes',   titlesize = SMALL_SIZE)
plt.rc('axes',   labelsize = MEDIUM_SIZE)
plt.rc('xtick',  labelsize = SMALL_SIZE)
plt.rc('ytick',  labelsize = SMALL_SIZE)
plt.rc('legend', fontsize  = SMALL_SIZE)
plt.rc('figure', titlesize = BIG_SIZE)
