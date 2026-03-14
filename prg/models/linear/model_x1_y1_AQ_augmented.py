#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from prg.models.linear.base_model_linear import LinearAmQ
from prg.models.linear.model_x1_y1_AQ_pairwise import Model_x1_y1_AQ_pairwise
from prg.utils.exceptions import NumericalError

__all__ = ["Model_x1_y1_AQ_augmented"]


class Model_x1_y1_AQ_augmented(LinearAmQ):

    def __init__(self):

        mod = Model_x1_y1_AQ_pairwise()

        super().__init__(
            *self.classic2pairwise(mod),
            augmented=True,
            pairwiseModel=False,
        )

        # print(f"mQ={self.mQ}")
        # print(f"Pz0={self.Pz0}")
        # print(f"mz0={self.mz0}")
        # input("ATTENTE")
