#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from prg.models.Generate_MatrixCov import generate_block_matrix
from prg.classes.SeedGenerator import SeedGenerator
from prg.models.linear.base_model_linear import LinearAmQ

__all__ = ["Model_A_mQ_x1_y1_VPgreaterThan1"]


class Model_A_mQ_x1_y1_VPgreaterThan1(LinearAmQ):
    """The A matrix got a VP>1.
    A is not ergodic, with no stationary distribution.
    """

    MODEL_NAME = "A_mQ_x1_y1_VPgreaterThan1"

    def __init__(self) -> None:

        randMatrices = SeedGenerator(5)

        # Dimensions x=1, y=1
        dim_x = 1
        dim_y = 1
        dim_xy = dim_x + dim_y

        # A = np.array( [ [0.5813651,  0.22435528], [0.22435528, 1.1186349 ]] )
        A = np.array([[0.94992007, -0.00282614], [-0.00282614, 0.85007993]])
        B = np.eye(A.shape[0])
        # mQ = np.array( [[0.739010989010989,   0.27774725274725276],
        #                      [0.27774725274725276, 0.9968131868131868]] )
        # mQ = np.diag([0.4, 0.2])
        # mz0 = np.zeros((dim_x + dim_y, 1))
        # Pz0 = np.eye(dim_x + dim_y)
        mQ = generate_block_matrix(randMatrices.rng, dim_x, dim_y, 0.15)
        mz0 = randMatrices.rng.standard_normal((dim_xy, 1))
        Pz0 = generate_block_matrix(randMatrices.rng, dim_x, dim_y, 0.15)

        super().__init__(dim_x=dim_x, dim_y=dim_y, A=A, B=B, mQ=mQ, mz0=mz0, Pz0=Pz0)
