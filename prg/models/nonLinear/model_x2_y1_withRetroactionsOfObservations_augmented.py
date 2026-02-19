#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from .base_model_nonLinear import BaseModelNonLinear
from others.utils import check_consistency
from .model_x2_y1_withRetroactionsOfObservations import ModelX2Y1_withRetroactionsOfObservations


class ModelX2Y1_withRetroactionsOfObservations_augmented(BaseModelNonLinear):
    """
    Nonlinear model with retro-actions of observations and of states.
    The model includes additive Gaussian process and observation noises.
    ATTENTION : ce modèle a été construit pour être utilisé avec un filtre 
                UKF et comparé avec le modèle 'ModelX2Y1_withRetroactionsOfObservations'
                pour un filtre UPKF, cf rapport.
    """

    MODEL_NAME: str = "x2_y1_withRetroactionsOfObservations_augmented"

    def __init__(self) -> None: 
        super().__init__(dim_x=3, dim_y=1, model_type="nonlinear", augmented=True)
        
        # pour récupérer les paramètre du modèle non augmenté
        self.mod = ModelX2Y1_withRetroactionsOfObservations()
        
        self.mQ   = np.zeros((self.dim_xy, self.dim_xy))
        self.mQ[0:self.dim_x, 0:self.dim_x] = self.mod.mQ
        self.z00  = np.zeros((self.dim_xy, 1))
        self.z00[0:self.dim_x] = self.mod.z00
        self.Pz00 = np.eye(self.dim_xy)
        self.Pz00[0:self.dim_x, 0:self.dim_x] = self.mod.Pz00
        
        if __debug__:
            check_consistency(mQ=self.mQ, Pz00=self.Pz00)

        self.a, self.b, self.c, self.d, self.e, self.f = self.mod.a, self.mod.b, self.mod.c, self.mod.d, self.mod.e, self.mod.f


    # ------------------------------------------------------------------
    def _fx(self, x, t, dt):
        """
        Nonlinear state function with retro-action on observation.
        """

        # Le fait d'utiliser le modèle non augmenté garanti que l'on fait les mêmes choses
        ax = self.mod._gx(x[0:self.dim_x-self.dim_y], x[self.dim_x-self.dim_y:], t[0:self.dim_x-self.dim_y], t[self.dim_x-self.dim_y:], dt)
        ay = self.mod._gy(x[0:self.dim_x-self.dim_y], x[self.dim_x-self.dim_y:], t[0:self.dim_x-self.dim_y], t[self.dim_x-self.dim_y:], dt)

        return np.block([[ax],[ay]])

    # ------------------------------------------------------------------
    def _hx(self, x, u, dt):
        """
        Nonlinear observation function with retro-action on previous observation.
        Le bruit $u$ est nul dans cette formulation.
        """
        
        return x[-1].reshape(-1, 1)

    # ------------------------------------------------------------------
    def _g(self, x, y, t, u, dt):
        """
        Combined state and observation using Wojciech’s formulation.
        """
        if __debug__:
            assert x.shape == (3, 1), f"x must be (3,1), got {x.shape}"
            assert y.shape == (1, 1), f"y must be (1,1), got {y.shape}"
            assert t.shape == (3, 1), f"t must be (3,1), got {t.shape}"
            assert u.shape == (1, 1), f"u must be (1,1), got {u.shape}"
            assert isinstance(dt, (float, int)), "dt must be a float"

        fx_val = self._fx(x, t, dt)
        hx_val = self._hx(fx_val, u, dt)
        return np.vstack((fx_val, hx_val))

    # ------------------------------------------------------------------
    def _jacobiens_g(self, x, y, t, u, dt):
        """
        Jacobians of combined state and observation function.
        """
        if __debug__:
            assert x.shape == (3, 1), f"x must be (3,1), got {x.shape}"
            assert y.shape == (1, 1), f"y must be (1,1), got {y.shape}"
            assert t.shape == (3, 1), f"t must be (3,1), got {t.shape}"
            assert u.shape == (1, 1), f"u must be (1,1), got {u.shape}"
            assert isinstance(dt, (float, int)), "dt must be a float"

        x1, x2, x3 = x.flatten()

        An = np.array([[self.a,   self.b, self.c * (1.-np.tanh(x3)**2), 0.],
                       [0,        self.d, self. e * np.cos(x3),         0.],
                       [2 * x1,       0., self.f,                       0.],
                       [2 * x1,       0., self.f,                       0.]])
        Bn = np.array([[1., 0., 0., 0.],
                       [0., 1., 0., 0.],
                       [0., 0., 1., 0.],
                       [0., 0., 1., 0.]])

        return An, Bn
