#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from .base_model_nonLinear import BaseModelNonLinear
from .model_x1_y1_withRetroactions import ModelX1Y1_withRetroactions


class ModelX1Y1_withRetroactions_augmented(BaseModelNonLinear):
    """
    Nonlinear model with retro-actions of observations and of states.
    The model includes additive Gaussian process and observation noises.
    ATTENTION : ce modèle a été construit pour être utilisé avec un filtre
                UKF et comparé avec le modèle 'ModelX1Y1_withRetroactions'
                pour un filtre UPKF, cf rapport.
    """

    MODEL_NAME: str = "x1_y1_withRetroactions_augmented"

    def __init__(self) -> None:
        super().__init__(dim_x=2, dim_y=1, model_type="nonlinear", augmented=True)

        # Le fait d'utiliser le modèle non augmenté garanti que l'on fait les mêmes choses
        self.mod = ModelX1Y1_withRetroactions()

        # (C) Sustained oscillations / limit-cycle-like: INTERESSANT
        # (a,b,c,d) = (0.99,\;1.2,\;0.9,\;1.5)
        # Expected behaviour: persistent oscillations of self.moderate amplitude; nonlinear terms drive and sustain the cycles.
        # Numeric tips: choose \(x_0,y_0\) small but nonzero, \(\sigma\) very small (e.g.\ 0.005) to reveal deterministic oscillation, \(N\ge 300\).
        self.mQ = np.zeros((self.dim_xy, self.dim_xy))
        self.mQ[: self.dim_x, : self.dim_x] = self.mod.mQ

        # Dimensions état augmenté x=2, y=1
        # Dimensions état original (non augmenté) x=1, y=1
        dim_x = self.mod.dim_x
        dim_y = self.mod.dim_y
        dim_xy = self.mod.dim_xy

        self.mz0 = np.zeros((dim_xy + dim_y, 1))
        self.mz0[0:dim_xy] = self.mod.mz0
        self.mz0[dim_xy : dim_xy + dim_y] = self.mz0[dim_xy - dim_y : dim_xy]

        self.Pz0 = np.zeros((dim_xy + dim_y, dim_xy + dim_y))
        self.Pz0[0:dim_xy, 0:dim_xy] = self.mod.Pz0
        # On recopie la derniere ligne
        self.Pz0[dim_xy : dim_xy + dim_y, :] = self.Pz0[dim_xy - dim_y : dim_xy, :]
        # On recopie la derniere colonne
        self.Pz0[:, dim_xy : dim_xy + dim_y] = self.Pz0[:, dim_xy - dim_y : dim_xy]

        self.a, self.b, self.c, self.d = self.mod.a, self.mod.b, self.mod.c, self.mod.d

    # ------------------------------------------------------------------
    def _fx(self, x, t, dt):
        """
        Nonlinear state function with retro-action of observations on state.
        """

        # Le fait d'utiliser le modèle non augmenté garanti que l'on fait les mêmes choses
        ax = self.mod._gx(
            x[: self.dim_x - self.dim_y],
            x[self.dim_x - self.dim_y :],
            t[: self.dim_x - self.dim_y],
            t[self.dim_x - self.dim_y :],
            dt,
        )
        ay = self.mod._gy(
            x[: self.dim_x - self.dim_y],
            x[self.dim_x - self.dim_y :],
            t[: self.dim_x - self.dim_y],
            t[self.dim_x - self.dim_y :],
            dt,
        )

        return np.block([[ax], [ay]])

    # ------------------------------------------------------------------
    def _hx(self, x, u, dt):
        """
        Nonlinear state function with retro-action of states on observation.
        Le bruit $u$ est nul dans cette formulation.
        """
        return x[-1].reshape(-1, 1)

    # ------------------------------------------------------------------
    def _g(self, x, y, t, u, dt):
        """Combine state and observation using Wojciech’s formulation."""
        if __debug__:
            assert x.shape == (2, 1), f"x must be (2,1), got {x.shape}"
            assert y.shape == (1, 1), f"y must be (1,1), got {y.shape}"
            assert t.shape == (2, 1), f"t must be (2,1), got {t.shape}"
            assert u.shape == (1, 1), f"u must be (1,1), got {u.shape}"
            assert isinstance(dt, (float, int)), "dt must be a float"

        fx_val = self._fx(x, t, dt)
        hx_val = self._hx(fx_val, u, dt)
        return np.vstack((fx_val, hx_val))

    # ------------------------------------------------------------------
    def _jacobiens_g(self, x, y, t, u, dt):
        if __debug__:
            assert x.shape == (2, 1), f"x must be (2,1), got {x.shape}"
            assert y.shape == (1, 1), f"y must be (1,1), got {y.shape}"
            assert t.shape == (2, 1), f"t must be (2,1), got {t.shape}"
            assert u.shape == (1, 1), f"u must be (1,1), got {u.shape}"
            assert isinstance(dt, (float, int)), "dt must be a float"

        x1, x2 = x.flatten()

        An = np.array(
            [
                [self.a, self.b * (1.0 - np.tanh(x2) ** 2), 0.0],
                [self.d * np.cos(x1), self.c, 0.0],
                [self.d * np.cos(x1), self.c, 0.0],
            ]
        )
        Bn = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])

        return An, Bn
