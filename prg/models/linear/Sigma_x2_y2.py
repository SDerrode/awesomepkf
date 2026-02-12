#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from .base_model_linear import LinearSigma  # On utilise directement la sous-classe LinearSigma


class Model_Sigma_x2_y2(LinearSigma):
    """
    Modèle linéaire Sigma avec dim_x=2 et dim_y=1.

    Paramétrisation : sxx, syy, a, b, c, d, e
    """
    
    MODEL_NAME = "Sigma_x2_y2"
    
    def __init__(self) -> None:
        
        # Dimensions x=2, y=2
        dim_x = 2
        dim_y = 2

        sxx = np.array([[1.0, 0.4],
                        [0.4, 1.0]])
        b   = np.array([[0.6, 0.2],
                        [0.2, 0.6]])
        syy = np.array([[1.0, 0.3],
                        [0.3, 1.0]])
        a   = np.array([[0.5, 0.2],
                        [0.2, 0.5]])
        d   = np.array([[0.1, 0.05],
                        [0.05,0.1]])
        e   = np.array([[0.2, 0.15],
                        [0.15,0.2]])
        c   = np.array([[0.2, 0.1],
                        [0.1, 0.2]])

        Q1  = np.block([[sxx, b.T], [b, syy]])
        Q2  = np.block([[a,   e],   [d, c]])
        
        super().__init__(dim_x=dim_x, dim_y=dim_y, sxx=sxx, syy=syy, a=a, b=b, c=c, d=d, e=e)
