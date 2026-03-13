#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from prg.models.linear.base_model_linear import LinearAmQ

__all__ = ["Model_x3_y1_AQ_classic"]


class Model_x3_y1_AQ_classic(LinearAmQ):

    def __init__(self):

        dim_x, dim_y = 3, 1

        # 1- Ecriture dun modele classique selon
        # \begin{align}
        #     \mX_n &= \mF_n \, \mX_{n-1} + \mC_n \, \mU_n,\\
        #     \mY_n &= \mH_n \, \mX_n + \mD_n \, \mW_n,
        # \end{align}

        try:

            F = np.zeros((dim_x, dim_x))
            C = np.zeros((dim_x, dim_x))

            H = np.zeros((dim_y, dim_x))
            D = np.zeros((dim_y, dim_y))

            F = np.array([[0.6, 0.1, 0.2], [0.15, 0.4, 0.15], [0.1, 0.2, 0.7]])
            C = np.array([[0.1, 0.05, 0.05], [0.05, 0.1, 0.05], [0.05, 0.05, 0.1]])
            H = np.array([[0.4, 0.3, 0.2]])
            D = np.array([[0.1]])

        except (ValueError, IndexError, np.exceptions.AxisError) as e:
            raise NumericalError(
                f"[{model_x3_y1_AQ_classic.MODEL_NAME}] Initialization failed: {e}"
            ) from e

        mQ, mz0, Pz0 = LinearAmQ._init_random_params(dim_x, dim_y, val_max=0.15)
        # mQ = np.diag(np.diag(mQ))  # bruit non corrélé

        # 2- Ecriture du meme modele sous forme couple pour utiliser les algo couples
        A = np.block(
            [
                [F, np.zeros((dim_x, dim_y))],
                [H @ F, np.zeros((dim_y, dim_y))],
            ]
        )

        B = np.block(
            [
                [C, np.zeros((dim_x, dim_y))],
                [H @ C, D],
            ]
        )

        super().__init__(
            dim_x=dim_x,
            dim_y=dim_y,
            A=A,
            mQ=mQ,
            mz0=mz0,
            Pz0=Pz0,
            B=B,
            pairwiseModel=False,
        )
