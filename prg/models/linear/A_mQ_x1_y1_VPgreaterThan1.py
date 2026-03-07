#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from prg.models.linear.base_model_linear import LinearAmQ

__all__ = ["Model_A_mQ_x1_y1_VPgreaterThan1"]


class Model_A_mQ_x1_y1_VPgreaterThan1(LinearAmQ):
    """The A matrix has a eigenvalue > 1.
    A is not ergodic, with no stationary distribution.
    """

    MODEL_NAME = "A_mQ_x1_y1_VPgreaterThan1"

    def __init__(self):
        dim_x, dim_y = 1, 1
        A = np.array([[0.94992007, -0.00282614], [-0.00282614, 0.85007993]])
        mQ, mz0, Pz0 = LinearAmQ._init_random_params(dim_x, dim_y, val_max=0.15)
        super().__init__(dim_x=dim_x, dim_y=dim_y, A=A, mQ=mQ, mz0=mz0, Pz0=Pz0)
