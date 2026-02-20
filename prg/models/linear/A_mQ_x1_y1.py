#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from .base_model_linear import LinearAmQ


class Model_A_mQ_x1_y1(LinearAmQ):
    
    MODEL_NAME = "A_mQ_x1_y1"

    def __init__(self) -> None:
        
        # Dimensions x=1, y=1
        dim_x  = 1
        dim_y  = 1
        dim_xy = dim_x+dim_y

        a, b, c, d = 0.95, -0.3, 0.2, 0.85
        # a, b, c, d =  0.533, -0.1099, 0.0418, 0.0275
        A = np.array( [[a, b],
                       [c, d]] )
        B = np.eye(A.shape[0])
        # mQ   = np.array( [[0.739,  0.2777], 
        #                   [0.2777, 0.9968]])
        mQ   = np.diag( [0.1, 0.2])

        z0  = np.zeros((dim_xy, 1))
        Pz0 = np.zeros((dim_xy, dim_xy))
        Pz0[0:dim_x, 0:dim_x] = np.eye(dim_x)
        
        x0 = np.zeros((dim_x, 1)) + 1.13

        super().__init__(dim_x=dim_x, dim_y=dim_y, A=A, B=B, mQ=mQ, z0=z0, Pz0=Pz0)
