#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from .base_model_nonLinear import BaseModelNonLinear
from .model_x1_y1_withRetroaction import ModelX1Y1_withRetroactions
from others.utils import check_consistency

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
        self.mQ   = np.zeros(shape=(self.dim_xy, self.dim_xy))
        self.mQ[0:self.dim_x, 0:self.dim_x] = self.mod.mQ
        self.z00  = np.zeros((self.dim_xy, 1))
        self.z00[0:self.dim_x] = self.mod.z00
        self.Pz00 = np.eye(self.dim_xy)
        self.Pz00[0:self.dim_x, 0:self.dim_x] = self.mod.Pz00
        
        if __debug__:
            check_consistency(mQ=self.mQ, Pz00=self.Pz00)
            
        self.a, self.b, self.c, self.d = self.mod.a, self.mod.b, self.mod.c, self.mod.d

    # ------------------------------------------------------------------
    def _fx(self, x, t, dt):
        """
        Nonlinear state function with retro-action of observations on state.
        """

        # Le fait d'utiliser le modèle non augmenté garanti que l'on fait les mêmes choses
        ax = self.mod._gx(x[0:self.dim_x-self.dim_y], x[self.dim_x-self.dim_y:], t[0:self.dim_x-self.dim_y], t[self.dim_x-self.dim_y:], dt)
        ay = self.mod._gy(x[0:self.dim_x-self.dim_y], x[self.dim_x-self.dim_y:], t[0:self.dim_x-self.dim_y], t[self.dim_x-self.dim_y:], dt)
        
        return np.block([[ax],[ay]])

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

        An = np.array([[self.a,              self.b * (1.-np.tanh(x2)**2), 0.],
                       [self.d * np.cos(x1), self.c,                       0.],
                       [self.d * np.cos(x1), self.c,                       0.]])
        Bn = np.array([[1., 0., 0.],
                       [0., 1., 0.],
                       [0., 1., 0.]])

        return An, Bn
