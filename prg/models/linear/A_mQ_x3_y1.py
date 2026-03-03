#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from prg.models.Generate_MatrixCov import generate_block_matrix
from prg.classes.SeedGenerator import SeedGenerator
from prg.models.linear.base_model_linear import LinearAmQ
from prg.exceptions import NumericalError

__all__ = ["Model_A_mQ_x3_y1"]


class Model_A_mQ_x3_y1(LinearAmQ):

    MODEL_NAME = "A_mQ_x3_y1"

    def __init__(self) -> None:

        randMatrices = SeedGenerator()

        dim_x = 3
        dim_y = 1
        dim_xy = dim_x + dim_y

        A = np.array(
            [
                [
                    0.6399176954732511,
                    -0.1502057613168724,
                    0.07818930041152264,
                    -0.18518518518518517,
                ],
                [
                    0.26851851851851855,
                    0.5462962962962961,
                    -0.09259259259259262,
                    -0.08333333333333329,
                ],
                [
                    0.22119341563786007,
                    0.17798353909465012,
                    0.36625514403292186,
                    -0.06481481481481483,
                ],
                [
                    -0.2777777777777778,
                    0.055555555555555546,
                    -0.11111111111111113,
                    0.49999999999999994,
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
                f"[{Model_A_mQ_x3_y1.MODEL_NAME}] Initialization failed: {e}"
            ) from e

        super().__init__(dim_x=dim_x, dim_y=dim_y, A=A, B=B, mQ=mQ, mz0=mz0, Pz0=Pz0)
