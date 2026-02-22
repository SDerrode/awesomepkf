#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from .base_model_nonLinear import BaseModelNonLinear

class ModelX2Y1(BaseModelNonLinear):
    """
    Nonlinear model with:
      - 2-dimensional state vector x = [x1, x2]
      - 1-dimensional observation y

    System dynamics (nonlinear):
        f(x) = [
            x1 + T * x2,
            x2 - T*(alpha * sin(x1) + beta * x2)
        ]

    Measurement equation (non linear):
        h(x) = x1**2 / (1 + x1**2) + gamma * sin(x2)

    The model includes additive Gaussian process and observation noises.
    """

    MODEL_NAME: str = "x2_y1_rapport"

    def __init__(self) -> None:
        super().__init__(dim_x=2, dim_y=1, model_type="nonlinear")

        self.alpham = 1.
        self.betam  = 0.1
        self.gammam = 0.5
        self.mQ     = np.diag([1E-4, 1E-4, 1e-4])
        self.mz0    = np.zeros((self.dim_xy, 1))
        self.Pmz0   = np.eye(self.dim_xy)

    # ------------------------------------------------------------------
    def _fx(self, x: np.ndarray, t: np.ndarray, dt: float) -> np.ndarray:
        """State transition function f(x) with process noise."""

        return np.array([
            [x[0,0] + dt*x[1,0]                                               + t[0,0]],
            [x[1,0] - dt*(self.alpham * np.sin(x[0,0]) + self.betam * x[1,0]) + t[1,0]]
        ])

    # ------------------------------------------------------------------
    def _hx(self, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        """Measurement function h(x) with observation noise."""

        return np.array([
            [x[0,0]**2 / (1. + x[0,0]**2) + self.gammam * np.sin(x[1,0]) + u[0,0]]
        ])

    # ------------------------------------------------------------------
    def _g(self, x: np.ndarray, y: np.ndarray, t: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        """Combine state and observation using Wojciech’s formulation."""
        if __debug__:
            assert x.shape == (2, 1)
            assert y.shape == (1, 1)
            assert t.shape == (2, 1)
            assert u.shape == (1, 1)
            assert isinstance(dt, (float, int))

        fx_val = self._fx(x, t, dt)
        hx_val = self._hx(fx_val, u, dt)
        return np.vstack((fx_val, hx_val))

    # ------------------------------------------------------------------
    def _jacobiens_g(self, x: np.ndarray, y: np.ndarray, t: np.ndarray, u: np.ndarray, dt: float):
        if __debug__:
            assert x.shape == (2, 1)
            assert y.shape == (1, 1)
            assert t.shape == (2, 1)
            assert u.shape == (1, 1)
            assert isinstance(dt, (float, int))

        x1, x2 = x.flatten()
        t1, t2 = t.flatten()

        A = x1 + dt*x2 + t1
        B = x2 - dt*(self.alpham*np.sin(x1) + self.betam*x2) + t2
        Z = 2*A/(1.+A**2)**2
        W = self.gammam*np.cos(B)
        An = np.array([[1.,                            dt,                        0.],
                       [-self.alpham*dt*np.cos(x1),    1.-self.betam*dt,          0.],
                       [Z-self.alpham*dt*np.cos(x1)*W, Z*dt+(1.-self.betam*dt)*W, 0.]])
        Bn = np.array([[1., 0., 0.],
                       [0., 1., 0.],
                       [Z,  W,  1.]])

        return An, Bn
