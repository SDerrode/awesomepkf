#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from prg.models.Generate_MatrixCov import generate_block_matrix
from prg.classes.SeedGenerator import SeedGenerator
from prg.models.linear.base_model_linear import LinearAmQ
from prg.exceptions import NumericalError

__all__ = ["Model_A_mQ_x1_y1"]


class Model_A_mQ_x1_y1(LinearAmQ):

    MODEL_NAME = "A_mQ_x1_y1"

    def __init__(self) -> None:

        randMatrices = SeedGenerator()

        # Dimensions x=1, y=1
        dim_x = 1
        dim_y = 1
        dim_xy = dim_x + dim_y

        a, b, c, d = 0.95, -0.3, 0.2, 0.85
        A = np.array([[a, b], [c, d]])
        B = np.eye(A.shape[0])

        try:
            mQ = generate_block_matrix(randMatrices.rng, dim_x, dim_y, 0.01)
            mz0 = randMatrices.rng.standard_normal((dim_xy, 1))
            Pz0 = generate_block_matrix(randMatrices.rng, dim_x, dim_y, 0.05)
        except (ValueError, np.exceptions.AxisError) as e:
            raise NumericalError(
                f"[{Model_A_mQ_x1_y1.MODEL_NAME}] Initialization failed: {e}"
            ) from e

        super().__init__(dim_x=dim_x, dim_y=dim_y, A=A, B=B, mQ=mQ, mz0=mz0, Pz0=Pz0)
