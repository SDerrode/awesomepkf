#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from .base_model_linear import LinearAmQ

from others.geneMatriceCov import generate_block_matrix
from classes.SeedGenerator import SeedGenerator


class Model_A_mQ_x1_y1(LinearAmQ):

    MODEL_NAME = "A_mQ_x1_y1"

    def __init__(self) -> None:

        randMatrices = SeedGenerator(())

        # Dimensions x=1, y=1
        dim_x = 1
        dim_y = 1
        dim_xy = dim_x + dim_y

        a, b, c, d = 0.95, -0.3, 0.2, 0.85
        A = np.array([[a, b], [c, d]])
        B = np.eye(A.shape[0])

        # mQ = np.diag([0.1, 0.2])
        # mz0 = np.zeros((dim_xy, 1)) + 1.13
        # Pz0 = np.eye(dim_xy) + 1.14

        mQ = generate_block_matrix(randMatrices.rng, dim_x, dim_y, 0.15)
        mz0 = randMatrices.rng.standard_normal((dim_xy, 1))
        Pz0 = generate_block_matrix(randMatrices.rng, dim_x, dim_y, 0.15)

        super().__init__(dim_x=dim_x, dim_y=dim_y, A=A, B=B, mQ=mQ, mz0=mz0, Pz0=Pz0)
