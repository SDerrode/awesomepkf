#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

def addParseToParser(parser, listOptions):
    """
    Ajoute des arguments à un ArgumentParser.
    
    :param parser: argparse.ArgumentParser
    :param listOptions: liste de chaînes, options optionnelles à ajouter
    """
    
    # =========================
    # Options toujours disponibles
    # =========================
    parser.add_argument(
        '--verbose', type=int, choices=range(0, 3), default=0, dest='verbose',
        help='Set the verbose level (0=quiet, 2=maximum, default=0)'
    )
    parser.add_argument(
        '--traceplot', action='store_true', dest='traceplot',
        help='Save parameters trace on disk and plot figures (default: False if not specified)'
    )

    # =========================
    # Options optionnelles configurables
    # =========================
    option_config = {
        'N': {
            'type': int,
            'default': 5000,
            'help': 'Set the number of samples to process (default: 5000)'
        },
        'sKey': {
            'type': int,
            'default': None,
            'help': 'Set the random generator seed (default: None)'
        },
        'linearModelName': {
            'choices': ['A_mQ_x1_y1', 'A_mQ_x2_y1_augmented', 'A_mQ_x1_y1_VPgreaterThan1', 'A_mQ_x2_y2', 
                        'A_mQ_x3_y1', 'Sigma_x1_y1', 'Sigma_x2_y2', 'Sigma_x3_y1'],
            'default': 'A_mQ_x3_y1',
            'help': 'Linear model to process data (default: A_mQ_x3_y1)'
        },
        'sigmaSet': {
            'choices': ['julier1995', 'wan2000', 'cpkf', 'lerner2002'],
            'default': 'wan2000',
            'help': 'Sigma set points to use with UPKF (default: wan2000)'
        },
        'nonLinearModelName': {
            'choices': ['x1_y1_cubique', 'x1_y1_ext_saturant', 'x1_y1_gordon', 'x1_y1_sinus', 
                        'x1_y1_withRetroactions', 'x2_y1_withRetroactions_augmented', 'x2_y1', 'x2_y1_rapport', 'x2_y1_withRetroactionsOfObservations'],
            'default': 'x2_y1_rapport',
            'help': 'Non linear model to process data (default: x2_y1_rapport)'
        },
        'dataFileName': {
            'type': str,
            'default': None,
            'help': 'Full path where trajectories are stored (default: None)'
        },
        'withoutX': {
            'action': 'store_true',
            'default': False,
            'help': 'Save true X or not (default: False)'
        }
    }

    # Ajouter dynamiquement les options demandées
    for opt in listOptions:
        if opt in option_config:
            kwargs = option_config[opt].copy()
            # Définir le nom de destination
            kwargs['dest'] = opt
            parser.add_argument(f'--{opt}', **kwargs)
