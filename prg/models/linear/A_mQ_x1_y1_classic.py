#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from prg.models.linear.base_model_linear import LinearAmQ

__all__ = ["A_mQ_x1_y1_classic"]


class Model_A_mQ_x1_y1_classic(LinearAmQ):

    MODEL_NAME = "A_mQ_x1_y1_classic"

    def __init__(self):
        dim_x, dim_y = 1, 1
        A = np.array([[0.95, 0.0], [0.2, 0.0]])
        mQ, mz0, Pz0 = LinearAmQ._init_random_params(dim_x, dim_y, val_max=0.15)
        super().__init__(
            dim_x=dim_x,
            dim_y=dim_y,
            A=A,
            mQ=mQ,
            mz0=mz0,
            Pz0=Pz0,
            pairwiseModel=False,
        )
