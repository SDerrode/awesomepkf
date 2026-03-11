#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from prg.models.linear.base_model_linear import LinearAmQ
from prg.models.linear.model_x1_y1_AQ_pairwise import model_x1_y1_AQ_pairwise
from prg.utils.exceptions import NumericalError

__all__ = ["Model_x1_y1_AQ_augmented"]


class Model_x1_y1_AQ_augmented(LinearAmQ):

    MODEL_NAME = "Model_x1_y1_AQ_augmented"

    def __init__(self):
        mod = model_x1_y1_AQ_pairwise()

        try:
            dim_x = mod.dim_x
            dim_y = mod.dim_y
            dim_xy = mod.dim_xy

            Fn = mod.A
            Cn = mod.B
            Cov = mod.mQ
            Hn = np.zeros((dim_y, dim_xy))
            Hn[:, dim_x:] = np.eye(dim_y)
            Dn = np.zeros((dim_y, dim_y))

            A = np.block(
                [
                    [Fn, np.zeros((dim_xy, dim_y))],
                    [Hn @ Fn, np.zeros((1, dim_y))],
                ]
            )
            B = np.block(
                [
                    [Cn, np.zeros((dim_xy, dim_y))],
                    [Hn @ Cn, Dn],
                ]
            )

            mQ = np.zeros((dim_xy + dim_y, dim_xy + dim_y))
            mQ[0:dim_xy, 0:dim_xy] = Cov

            mz0 = np.zeros((dim_xy + dim_y, 1))
            mz0[0:dim_xy] = mod.mz0
            mz0[dim_xy : dim_xy + dim_y] = mz0[dim_xy - dim_y : dim_xy]

            Pz0 = np.zeros((dim_xy + dim_y, dim_xy + dim_y))
            Pz0[0:dim_xy, 0:dim_xy] = mod.Pz0
            Pz0[dim_xy : dim_xy + dim_y, :] = Pz0[dim_xy - dim_y : dim_xy, :]
            Pz0[:, dim_xy : dim_xy + dim_y] = Pz0[:, dim_xy - dim_y : dim_xy]

        except (ValueError, IndexError, np.exceptions.AxisError) as e:
            raise NumericalError(
                f"[{model_x1_y1_AQ_augmented.MODEL_NAME}] Initialization failed: {e}"
            ) from e

        super().__init__(
            dim_x=dim_xy,
            dim_y=dim_y,
            A=A,
            B=B,
            mQ=mQ,
            mz0=mz0,
            Pz0=Pz0,
            augmented=True,
            pairwiseModel=False,
        )
