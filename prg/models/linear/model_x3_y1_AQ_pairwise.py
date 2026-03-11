#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from prg.models.linear.base_model_linear import LinearAmQ

__all__ = ["model_x3_y1_AQ_pairwise"]


class Model_x3_y1_AQ_pairwise(LinearAmQ):

    def __init__(self):
        dim_x, dim_y = 3, 1
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
        mQ, mz0, Pz0 = LinearAmQ._init_random_params(dim_x, dim_y, val_max=0.15)
        super().__init__(
            dim_x=dim_x,
            dim_y=dim_y,
            A=A,
            mQ=mQ,
            mz0=mz0,
            Pz0=Pz0,
            pairwiseModel=True,
        )
