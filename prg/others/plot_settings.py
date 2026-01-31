#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib
import matplotlib.pyplot as plt

WINDOW     = {'xmin':0, 'xmax':400}
# window    = {'xmin': min(20, N), 'xmax': min(min(20, N)+100, N) }

dpi        = 150
facecolor  = '#AAAAAA'
plt.rcParams['axes.prop_cycle'] = plt.cycler(
    color=['r', 'g', 'b'],
    linestyle=['-', '--', ':']
)

SMALL_SMALL_SIZE  = 6
SMALL_SIZE        = 6
MEDIUM_SIZE       = 8
BIGGER_SIZE       = 10

plt.rc('font',   size      = SMALL_SIZE)  # controls default text sizes
plt.rc('axes',   titlesize = SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes',   labelsize = MEDIUM_SIZE) # fontsize of the x and y labels
plt.rc('xtick',  labelsize = SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick',  labelsize = SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize  = SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize = BIGGER_SIZE) # fontsize of the figure title
