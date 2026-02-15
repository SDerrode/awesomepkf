#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import sys

directory = Path(__file__)
sys.path.append(str(directory.parent.parent))

import logging
from typing import Any, Union, Optional
import warnings

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_discrete_lyapunov

# Linear models
from models.linear import BaseModelLinear, ModelFactoryLinear
# A few utils functions that are used several fois
from others.utils import is_covariance, check_consistency

# ----------------------------------------------------------------------
# Configuration du logging global
# ----------------------------------------------------------------------
logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# ParamLinear class
# ----------------------------------------------------------------------
class ParamLinear:
    """
    Manage PKF parameters with optional debug checks.

    Attributes:
        verbose: logging level
        dim_x, dim_y, dim_xy: state and observation dimensions
        kwargs: models parameters
    """

    def __init__(self, verbose: int, dim_x: int, dim_y: int, **kwargs) -> None:
        if __debug__:
            assert isinstance(dim_x, int) and dim_x > 0, "dim_x must be int > 0"
            assert isinstance(dim_y, int) and dim_y > 0, "dim_y must be int > 0"
            assert verbose in [0, 1, 2], "verbose must be 0, 1 or 2"

        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_xy = dim_x + dim_y
        self.verbose = verbose
        self._set_log_level()

        # Two ways to construct the object
        if len(kwargs.keys()) == 7:  # parametrization (A, mQ, z00, Pz00)
            self.constructorFrom_AB_mQ(kwargs['g'], kwargs['A'], kwargs['B'],
                                       kwargs['mQ'], kwargs['z00'], kwargs['Pz00'], kwargs['augmented'])
        elif len(kwargs.keys()) == 9:  # parametrization (sxx, syy, a, b, c, d, e) --> Sigma
            self.constructorFrom_Sigma(kwargs['g'], kwargs['sxx'], kwargs['syy'],
                                       kwargs['a'], kwargs['b'], kwargs['c'], kwargs['d'], kwargs['e'],
                                       kwargs['augmented'])
        else:
            logger.warning(f"⚠️ Le modèle n'est pas bien paramétré : {kwargs.keys()}")

        if __debug__:
            self._check_dimensions()
            self._check_consistency()

    # ------------------------------------------------------------------
    # Constructeurs
    # ------------------------------------------------------------------
    def constructorFrom_AB_mQ(self, g, A: np.ndarray, B: np.ndarray, mQ: np.ndarray,
                              z00: np.ndarray, Pz00: np.ndarray, augmented: bool) -> None:
        self.augmented = augmented
        self.g = g

        self._A = np.array(A, dtype=float)
        if __debug__:
            eigvals = np.linalg.eigvals(self._A)
            if np.any(np.abs(eigvals) >= 1.0):
                logger.warning(f"⚠️ Certaines valeurs propres de A ont un module >= 1 : {eigvals}")

        self._B = np.array(B, dtype=float)
        self._mQ = np.array(mQ, dtype=float)
        self._z00 = np.array(z00, dtype=float)
        self._Pz00 = np.array(Pz00, dtype=float)

        self._update_Sigma_from_A_B_mQ()

    def constructorFrom_Sigma(self, g, sxx: np.ndarray, syy: np.ndarray,
                              a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, e: np.ndarray,
                              augmented: bool) -> None:
        self.augmented = augmented
        self.g = g

        self._sxx, self._syy = np.array(sxx), np.array(syy)
        self._a, self._b, self._c, self._d, self._e = map(np.array, [a, b, c, d, e])

        self._update_A_B_mQ_from_Sigma()

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def _set_log_level(self) -> None:
        if self.verbose in [0, 1]:
            logger.setLevel(logging.CRITICAL + 1)
        elif self.verbose == 2:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.DEBUG)

    # ------------------------------------------------------------------
    # Update derived matrices
    # ------------------------------------------------------------------
    def _update_A_B_mQ_from_Sigma(self) -> None:
        self._Q1 = np.block([[self._sxx, self._b.T], [self._b, self._syy]])
        self._Q2 = np.block([[self._a, self._e], [self._d, self._c]])

        c, low = cho_factor(self._Q1)
        self._A = self._Q2 @ cho_solve((c, low), np.eye(self.dim_xy))
        self._B = np.eye(self.dim_xy)
        self._mQ = self._Q1 - self._A @ self._Q2.T

        self._z00 = np.zeros((self.dim_xy, 1))
        self._Pz00 = self._Q1.copy()

    def _update_Sigma_from_A_B_mQ(self) -> None:

        self._Q1    = solve_discrete_lyapunov(self._A, self._mQ)
        self._Q2    = self._A @ self._Q1
        self._Sigma = np.block([[self._Q1, self._Q2.T], [self._Q2, self._Q1]])

        # Vérification cohérence
        if __debug__:
            Q_est = self._Q1 - self._A @ self._Q2.T
            diff = self._mQ - Q_est
            rel_error = np.linalg.norm(diff) / (np.linalg.norm(self._mQ) + 1e-12)
            if rel_error > 1e-8:
                logger.warning(f"⚠️ Incohérence : Q ≉ Q1 - A Q2^T (erreur relative = {rel_error:.2e})")
                if self.verbose >= 2:
                    logger.debug(f"Différence :\n{diff}")
            else:
                logger.debug(f"♻️ Vérification OK : ||Q - (Q1 - A Q2^T)||_rel = {rel_error:.2e}")

        # Sous-blocs
        self._a   = self._Sigma[self.dim_xy:self.dim_xy+self.dim_x, 0:self.dim_x]
        self._b   = self._Sigma[self.dim_x:self.dim_xy, 0:self.dim_x]
        self._c   = self._Sigma[self.dim_xy+self.dim_x:2*self.dim_xy, self.dim_x:self.dim_xy]
        self._d   = self._Sigma[self.dim_xy+self.dim_x:2*self.dim_xy, 0:self.dim_x]
        self._e   = self._Sigma[self.dim_xy:self.dim_xy+self.dim_x, self.dim_x:self.dim_xy]
        self._sxx = self._Sigma[0:self.dim_x, 0:self.dim_x]
        self._syy = self._Sigma[self.dim_x:self.dim_xy, self.dim_x:self.dim_xy]

    # ------------------------------------------------------------------
    # Consistency checks
    # ------------------------------------------------------------------
    def _check_consistency(self) -> None:
        for attr, name in [('_mQ', 'mQ'), ('_Q1', 'Q1'), ('_Sigma', 'Sigma'),
                           ('_sxx', 'sxx'), ('_syy', 'syy'), ('_Pz00', 'Pz00')]:
            if hasattr(self, attr):
                is_covariance(getattr(self, attr), name)

    # ------------------------------------------------------------------    
    # Check dimensions
    # ------------------------------------------------------------------
    def _check_dimensions(self) -> None:
        expected_shapes = {
            'mQ': (self.dim_xy, self.dim_xy),
            'z00': (self.dim_xy, 1),
            'Pz00': (self.dim_xy, self.dim_xy),
        }
        for attr, shape in expected_shapes.items():
            if hasattr(self, f"_{attr}"):
                actual = getattr(self, f"_{attr}")
                if actual.shape != shape:
                    raise ValueError(f"⚠️ Matrice {attr} a une forme {actual.shape}, attendue {shape}")

    # ------------------------------------------------------------------
    # Getters / Setters and Properties
    # ------------------------------------------------------------------
    @property
    def A(self) -> np.ndarray: return self._A
    @A.setter
    def A(self, new_A: np.ndarray) -> None:
        self._A = np.array(new_A, dtype=float)
        self._update_Sigma_from_A_B_mQ()
        if __debug__:
            self._check_consistency()

    @property
    def B(self) -> np.ndarray: return self._B
    @B.setter
    def B(self, new_B: np.ndarray) -> None:
        self._B = np.array(new_B, dtype=float)
        self._update_Sigma_from_A_B_mQ()
        if __debug__:
            self._check_consistency()

    @property
    def mQ(self) -> np.ndarray: return self._mQ
    @mQ.setter
    def mQ(self, new_Q: np.ndarray) -> None:
        self._mQ = np.array(new_Q, dtype=float)
        self._update_Sigma_from_A_B_mQ()
        if __debug__:
            self._check_consistency()

    @property
    def z00(self) -> np.ndarray: return self._z00
    @property
    def Pz00(self) -> np.ndarray: return self._Pz00

    @property
    def sxx(self) -> np.ndarray: return self._sxx
    @property
    def syy(self) -> np.ndarray: return self._syy
    @property
    def a(self) -> np.ndarray: return self._a
    @property
    def b(self) -> np.ndarray: return self._b
    @property
    def c(self) -> np.ndarray: return self._c
    @property
    def d(self) -> np.ndarray: return self._d
    @property
    def e(self) -> np.ndarray: return self._e

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    def summary(self) -> None:
        def fmt(M: Any) -> str:
            return np.array2string(M, formatter={'float_kind': lambda x: f"{x:6.2f}"})

        print("=== ParamLinear Summary ===")
        print(f"dim_x={self.dim_x}, dim_y={self.dim_y}, verbose={self.verbose}\n")
        print("A:\n", fmt(self.A))
        print("B:\n", fmt(self.B))
        print("mQ:\n", fmt(self.mQ))
        print("z00:\n", fmt(self.z00))
        print("Pz00:\n", fmt(self.Pz00))
        print("========================\n")

        if __debug__:
            self._check_consistency()

