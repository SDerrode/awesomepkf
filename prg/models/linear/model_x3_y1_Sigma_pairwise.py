#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from prg.models.linear.base_model_linear import LinearSigma
from prg.utils.exceptions import NumericalError

__all__ = ["Model_x3_y1_Sigma_pairwise"]


class Model_x3_y1_Sigma_pairwise(LinearSigma):
    """
    Modèle linéaire Sigma avec dim_x=3 et dim_y=1.

    Paramétrisation : sxx, syy, a, b, c, d, e
    """

    def __init__(self) -> None:

        dim_x = 3
        dim_y = 1

        sxx = np.array([[1.0, 0.4, 0.4], [0.4, 1.0, 0.4], [0.4, 0.4, 1.0]])
        b = np.array([[0.6, 0.2, 0.4]])
        syy = np.array([[1.0]])
        a = np.array([[0.5, 0.1, 0.2], [0.4, 0.6, 0.2], [0.4, 0.4, 0.5]])
        d = np.array([[0.0, 0.0, 0.0]])
        e = np.array([[0.20], [0.15], [0.25]])
        c = np.array([[0.30]])

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
