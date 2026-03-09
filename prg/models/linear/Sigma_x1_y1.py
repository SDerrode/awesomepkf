#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from prg.models.linear.base_model_linear import LinearSigma
from prg.utils.exceptions import NumericalError

__all__ = ["Model_Sigma_x1_y1"]


class Model_Sigma_x1_y1(LinearSigma):
    """
    Modèle linéaire Sigma avec dim_x=1 et dim_y=1.

    Paramétrisation : sxx, syy, a, b, c, d, e
    """

    MODEL_NAME = "Sigma_x1_y1"

    def __init__(self) -> None:

        dim_x = 1
        dim_y = 1

        try:
            sxx = np.array([[1]])
            b = np.array([[0.3]])
            syy = np.array([[1]])
            a = np.array([[0.5]])
            d = np.array([[0.05]])
            e = np.array([[0.05]])
            c = np.array([[0.04]])
        except (ValueError, np.exceptions.AxisError) as ex:
            raise NumericalError(
                f"[{Model_Sigma_x1_y1.MODEL_NAME}] Parameter initialization failed: {ex}"
            ) from ex

        # NumericalError (Cholesky, bloc) remonte naturellement depuis LinearSigma._initSigma()
        super().__init__(
            dim_x=dim_x,
            dim_y=dim_y,
            sxx=sxx,
            syy=syy,
            a=a,
            b=b,
            c=c,
            d=d,
            e=e,
            pairwiseModel=True,
        )
