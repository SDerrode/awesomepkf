#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from prg.models.Generate_MatrixCov import generate_block_matrix
from prg.classes.SeedGenerator import SeedGenerator
from prg.models.linear.base_model_linear import LinearAmQ
from prg.exceptions import NumericalError

__all__ = ["Model_A_mQ_x2_y2"]


class Model_A_mQ_x2_y2(LinearAmQ):

    MODEL_NAME = "A_mQ_x2_y2"

    def __init__(self) -> None:

        randMatrices = SeedGenerator(5)

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

        try:
            mQ = generate_block_matrix(randMatrices.rng, dim_x, dim_y, 0.15)
            mz0 = randMatrices.rng.standard_normal((dim_xy, 1))
            Pz0 = generate_block_matrix(randMatrices.rng, dim_x, dim_y, 0.15)
        except (ValueError, np.exceptions.AxisError) as e:
            raise NumericalError(
                f"[{Model_A_mQ_x2_y2.MODEL_NAME}] Initialization failed: {e}"
            ) from e

        super().__init__(dim_x=dim_x, dim_y=dim_y, A=A, B=B, mQ=mQ, mz0=mz0, Pz0=Pz0)
