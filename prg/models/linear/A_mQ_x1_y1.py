#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from .base_model_linear import LinearAmQ

# from others.geneMatriceCov import generate_block_matrix


class Model_A_mQ_x1_y1(LinearAmQ):

    MODEL_NAME = "A_mQ_x1_y1"

    def __init__(self) -> None:

        # Dimensions x=1, y=1
        dim_x = 1
        dim_y = 1
        dim_xy = dim_x + dim_y

        a, b, c, d = 0.95, -0.3, 0.2, 0.85
        A = np.array([[a, b], [c, d]])
        B = np.eye(A.shape[0])
        mQ = np.diag([0.1, 0.2])
        # mQ = generate_block_matrix(rng, dim_x, dim_y, 0.1)

        mz0 = np.zeros((dim_xy, 1)) + 1.13
        # mz0 = rng.standard_normal(dim_x + dim_y)
        Pz0 = np.eye(dim_xy) + 1.14
        # Pz0 = generate_block_matrix(rng, dim_x, dim_y, 0.1)

        super().__init__(dim_x=dim_x, dim_y=dim_y, A=A, B=B, mQ=mQ, mz0=mz0, Pz0=Pz0)
