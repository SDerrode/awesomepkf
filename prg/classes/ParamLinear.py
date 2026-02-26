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
from others.utils import check_consistency
from others.numerics import EPS_ABS, EPS_REL

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
        self.augmented = None
        self._set_log_level()

        # Two ways to construct the object
        if len(kwargs.keys()) == 12:  # parametrization (A, mQ, mz0, Pz0)
            self.constructorFrom_AB_mQ(
                kwargs["A"], kwargs["B"], kwargs["mQ"], kwargs["mz0"], kwargs["Pz0"]
            )
        elif (
            len(kwargs.keys()) == 14
        ):  # parametrization (sxx, syy, a, b, c, d, e) --> Sigma
            self.constructorFrom_Sigma(
                kwargs["sxx"],
                kwargs["syy"],
                kwargs["a"],
                kwargs["b"],
                kwargs["c"],
                kwargs["d"],
                kwargs["e"],
            )
        else:
            logger.warning(f"⚠️ Le modèle n'est pas bien paramétré : {kwargs.keys()}")

        # Paramètre communs
        self.augmented = kwargs["augmented"]
        self.g = kwargs["g"]

        # Paramètres spécifiques UPKF - lorsque souhaite filtrer des données linéaire par upkf
        self.alpha = kwargs["alpha"]
        self.beta = kwargs["beta"]
        self.kappa = kwargs["kappa"]
        self.lambda_ = kwargs["lambda_"]

        # Paramètres spécifiques EPKF - lorsque souhaite filtrer des données linéaire par epkf
        self.jacobiens_g = kwargs["jacobiens_g"]

        if __debug__:
            self._check_consistency()

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return f"<ParamLinear(dim_y={self.dim_y}, dim_x={self.dim_x}, augmented={self.augmented}, verbose={self.verbose}>"

    # ------------------------------------------------------------------
    # Constructeurs
    # ------------------------------------------------------------------
    def constructorFrom_AB_mQ(
        self,
        A: np.ndarray,
        B: np.ndarray,
        mQ: np.ndarray,
        mz0: np.ndarray,
        Pz0: np.ndarray,
    ) -> None:

        self._A = np.array(A, dtype=float)
        if __debug__:
            eigvals = np.linalg.eigvals(self._A)
            if np.any(np.abs(eigvals) >= 1.0):
                logger.warning(
                    f"⚠️ Certaines valeurs propres de A ont un module >= 1 : {eigvals}"
                )

        self._B = np.array(B, dtype=float)
        self._mQ = np.array(mQ, dtype=float)
        self._mz0 = np.array(mz0, dtype=float)
        self._Pz0 = np.array(Pz0, dtype=float)

        self._update_Sigma_from_A_B_mQ()

    def constructorFrom_Sigma(
        self,
        sxx: np.ndarray,
        syy: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        d: np.ndarray,
        e: np.ndarray,
    ) -> None:

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
        self._Sigma = np.block([[self._Q1, self._Q2.T], [self._Q2, self._Q1]])

        c, low = cho_factor(self._Q1)
        self._A = self._Q2 @ cho_solve((c, low), np.eye(self.dim_xy))
        self._B = np.eye(self.dim_xy)
        self._mQ = self._Q1 - self._A @ self._Q2.T

        self._mz0 = np.zeros((self.dim_xy, 1))
        self._Pz0 = self._Q1.copy()

    def _update_Sigma_from_A_B_mQ(self) -> None:

        self._Q1 = solve_discrete_lyapunov(self._A, self._mQ)
        self._Q2 = self._A @ self._Q1
        self._Sigma = np.block([[self._Q1, self._Q2.T], [self._Q2, self._Q1]])

        # Vérification cohérence
        if __debug__:
            Q_est = self._Q1 - self._A @ self._Q2.T
            diff = self._mQ - Q_est
            rel_error = np.linalg.norm(diff) / (np.linalg.norm(self._mQ) + EPS_ABS)
            if rel_error > EPS_REL:
                logger.warning(
                    f"⚠️ Incohérence : Q ≉ Q1 - A Q2^T (erreur relative = {rel_error:.2e})"
                )
                if self.verbose >= 2:
                    logger.debug(f"Différence :\n{diff}")
            else:
                logger.debug(
                    f"♻️ Vérification OK : ||Q - (Q1 - A Q2^T)||_rel = {rel_error:.2e}"
                )

        # Sous-blocs
        self._a = self._Sigma[self.dim_xy : self.dim_xy + self.dim_x, : self.dim_x]
        self._b = self._Sigma[self.dim_x : self.dim_xy, : self.dim_x]
        self._c = self._Sigma[
            self.dim_xy + self.dim_x : 2 * self.dim_xy, self.dim_x : self.dim_xy
        ]
        self._d = self._Sigma[self.dim_xy + self.dim_x : 2 * self.dim_xy, : self.dim_x]
        self._e = self._Sigma[
            self.dim_xy : self.dim_xy + self.dim_x, self.dim_x : self.dim_xy
        ]
        self._sxx = self._Sigma[: self.dim_x, : self.dim_x]
        self._syy = self._Sigma[self.dim_x : self.dim_xy, self.dim_x : self.dim_xy]

    # ------------------------------------------------------------------
    # Consistency checks
    # ------------------------------------------------------------------
    def _check_consistency(self) -> None:
        if self.augmented:
            check_consistency(Q1=self._Q1, sxx=self._sxx, syy=self._syy)
        else:
            check_consistency(
                Q1=self._Q1,
                sxx=self._sxx,
                syy=self._syy,
                mQ=self._mQ,
                Sigma=self._Sigma,
                Pz0=self._Pz0,
            )

    # ------------------------------------------------------------------
    # Getters / Setters and Properties
    # ------------------------------------------------------------------
    @property
    def A(self) -> np.ndarray:
        return self._A

    @A.setter
    def A(self, new_A: np.ndarray) -> None:
        self._A = np.array(new_A, dtype=float)
        self._update_Sigma_from_A_B_mQ()
        if __debug__:
            self._check_consistency()

    @property
    def B(self) -> np.ndarray:
        return self._B

    @B.setter
    def B(self, new_B: np.ndarray) -> None:
        self._B = np.array(new_B, dtype=float)
        self._update_Sigma_from_A_B_mQ()
        if __debug__:
            self._check_consistency()

    @property
    def mQ(self) -> np.ndarray:
        return self._mQ

    @mQ.setter
    def mQ(self, new_Q: np.ndarray) -> None:
        self._mQ = np.array(new_Q, dtype=float)
        self._update_Sigma_from_A_B_mQ()
        if __debug__:
            self._check_consistency()

    @property
    def mz0(self) -> np.ndarray:
        return self._mz0

    @property
    def Pz0(self) -> np.ndarray:
        return self._Pz0

    @property
    def sxx(self) -> np.ndarray:
        return self._sxx

    @property
    def syy(self) -> np.ndarray:
        return self._syy

    @property
    def a(self) -> np.ndarray:
        return self._a

    @property
    def b(self) -> np.ndarray:
        return self._b

    @property
    def c(self) -> np.ndarray:
        return self._c

    @property
    def d(self) -> np.ndarray:
        return self._d

    @property
    def e(self) -> np.ndarray:
        return self._e

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    def summary(self) -> None:
        def fmt(M: Any) -> str:
            return np.array2string(M, formatter={"float_kind": lambda x: f"{x:6.2f}"})

        print("=== ParamLinear Summary ===")
        print(f"dim_x={self.dim_x}, dim_y={self.dim_y}, verbose={self.verbose}\n")
        print("A:\n", fmt(self.A))
        print("B:\n", fmt(self.B))
        print("mQ:\n", fmt(self.mQ))
        print("mz0:\n", fmt(self.mz0))
        print("Pz0:\n", fmt(self.Pz0))
        print("========================\n")
