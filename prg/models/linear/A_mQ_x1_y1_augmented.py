#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from .base_model_linear import LinearAmQ  # On utilise directement la sous-classe LinearAmQ

from .A_mQ_x1_y1 import Model_A_mQ_x1_y1

class Model_A_mQ_x1_y1_augmented(LinearAmQ):
    
    # Nom du modèle
    MODEL_NAME = "A_mQ_x1_y1_augmented"
    
    def __init__(self) -> None:
        
        # pour récupérer les paramètre du modèle non augmenté
        mod = Model_A_mQ_x1_y1()

        # Dimensions état augmenté x=2, y=1
        # Dimensions état original (non augmenté) x=1, y=1
        dim_x  = mod.dim_x
        dim_y  = mod.dim_y
        dim_xy = mod.dim_xy
        # print(f'dim_x={dim_x}, dim_y={dim_y}, 'dim_xy={dim_xy}')

        Fn  = mod.A
        Cn  = mod.B
        Cov = mod.mQ
        Hn  = np.zeros((dim_y, dim_xy))
        Hn[:, dim_x:] = np.eye(dim_y)
        Dn  = np.zeros((dim_y, dim_y))

        A = np.block([ [Fn,      np.zeros((dim_xy, dim_y))],
                       [Hn @ Fn, np.zeros((1,      dim_y))] ])
        
        B = np.block([ [Cn,      np.zeros((dim_xy, dim_y))],
                       [Hn @ Cn, Dn]])

        mQ = np.zeros((dim_xy+dim_y, dim_xy+dim_y))
        mQ[0:dim_xy, 0:dim_xy] = Cov

        z00  = np.zeros((dim_xy+dim_y, 1))
        Pz00 = np.eye(dim_xy+dim_y)

        super().__init__(dim_x=dim_xy, dim_y=dim_y, A=A, B=B, mQ=mQ, z00=z00, Pz00=Pz00, augmented=True)
