#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from .base_model_linear import LinearAmQ
from others.geneMatriceCov import generate_block_matrix
from classes.SeedGenerator import SeedGenerator


class Model_A_mQ_x2_y2(LinearAmQ):

    MODEL_NAME = "A_mQ_x2_y2"

    def __init__(self) -> None:

        randMatrices = SeedGenerator()

        # Dimensions x=2, y=2
        dim_x = 2
        dim_y = 2
        dim_xy = dim_x + dim_y

        A = np.array(
            [
                [
                    0.6323337679269881,
                    -0.09843546284224247,
                    -0.20273794002607562,
                    0.1434159061277705,
                ],
                [
                    -0.09843546284224246,
                    0.6323337679269883,
                    0.14341590612777047,
                    -0.20273794002607565,
                ],
                [
                    -0.028683181225554088,
                    -0.009452411994784882,
                    0.2040417209908735,
                    0.05019556714471971,
                ],
                [
                    -0.009452411994784863,
                    -0.028683181225554123,
                    0.05019556714471966,
                    0.20404172099087356,
                ],
            ]
        )
        B = np.eye(A.shape[0])
        # mQ = np.array(
        #     [
        #         [
        #             0.7225554106910039,
        #             0.3244784876140809,
        #             0.5678943937418514,
        #             0.1698174706649283,
        #         ],
        #         [
        #             0.3244784876140808,
        #             0.7225554106910039,
        #             0.1698174706649283,
        #             0.5678943937418514,
        #         ],
        #         [
        #             0.5678943937418514,
        #             0.1698174706649283,
        #             0.957513037809648,
        #             0.2719361147327249,
        #         ],
        #         [
        #             0.1698174706649283,
        #             0.5678943937418514,
        #             0.2719361147327249,
        #             0.957513037809648,
        #         ],
        #     ]
        # )

        # mz0 = np.zeros((dim_x + dim_y, 1))
        # Pz0 = np.eye(dim_x + dim_y)

        mQ = generate_block_matrix(randMatrices.rng, dim_x, dim_y, 0.15)
        mz0 = randMatrices.rng.standard_normal((dim_xy, 1))
        Pz0 = generate_block_matrix(randMatrices.rng, dim_x, dim_y, 0.15)

        super().__init__(dim_x=dim_x, dim_y=dim_y, A=A, B=B, mQ=mQ, mz0=mz0, Pz0=Pz0)
