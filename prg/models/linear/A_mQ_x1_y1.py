#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from .base_model_linear import LinearAmQ


class Model_A_mQ_x1_y1(LinearAmQ):
    
    MODEL_NAME = "A_mQ_x1_y1"

    def __init__(self) -> None:
        
        # Dimensions x=1, y=1
        dim_x = 1
        dim_y = 1

        a, b, c, d = 0.95, -0.3, 0.2, 0.85
        # a, b, c, d =  0.533, -0.1099, 0.0418, 0.0275
        A    = np.array( [[a, b],
                          [c, d]] )
        # mQ   = np.array( [[0.739,  0.2777], 
        #                   [0.2777, 0.9968]])
        mQ   = np.diag( [0.4, 0.2])
        
        z00  = np.zeros(shape=(dim_x+dim_y, 1))
        Pz00 = np.eye(dim_x+dim_y)

        super().__init__(dim_x=dim_x, dim_y=dim_y, A=A, mQ=mQ, z00=z00, Pz00=Pz00)