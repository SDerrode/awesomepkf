#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
from typing import Any

import numpy as np

# Non linear models
from prg.models.nonLinear import ModelFactoryNonLinear
from prg.classes.MatrixDiagnostics import CovarianceMatrix

__all__ = ["ParamNonLinear"]

# ----------------------------------------------------------------------
# Configuration du logging global
# ----------------------------------------------------------------------
logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# ParamNonLinear class
# ----------------------------------------------------------------------
class ParamNonLinear:
    """
    Manage Non linear parameters with optional debug checks.

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

        # Logger config according to verbose
        self._set_log_level()

        # Le modèle est-il un modèle augmenté ?
        self.augmented = kwargs["augmented"]

        # Non-linear function
        self.g = kwargs["g"]

        # Covariance matrices
        self._mQ = np.array(kwargs["mQ"], dtype=float)
        self._mz0 = np.array(kwargs["mz0"], dtype=float)
        self._Pz0 = np.array(kwargs["Pz0"], dtype=float)

        # UPKF specific parameters
        self.alpha = kwargs["alpha"]
        self.beta = kwargs["beta"]
        self.kappa = kwargs["kappa"]
        self.lambda_ = kwargs["lambda_"]

        # EPKF specific parameters
        self.jacobiens_g = kwargs["jacobiens_g"]

        if __debug__:
            if not self.augmented:
                for arr in [self._mQ, self._Pz0]:
                    report = CovarianceMatrix(arr).check()  # single diagnostic call
                    if not report.is_valid:
                        raise ValueError(f"Matrix  is not positive semi-definite.")

                # print(f"mQ = {self._mQ}")
                # print(f"Pz0 = {self._Pz0}")
                # check_consistency(mQ=self._mQ, Pz0=self._Pz0)

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"<ParamNonLinear(dim_y={self.dim_y}, dim_x={self.dim_x}, augmented={self.augmented}, verbose={self.verbose}, "
            f"alpha={self.alpha}, beta={self.beta}, kappa={self.kappa}, lambda_={self.lambda_})>"
        )

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
    # Getters / Setters and Properties
    # ------------------------------------------------------------------
    @property
    def mz0(self) -> np.ndarray:
        return self._mz0

    @property
    def Pz0(self) -> np.ndarray:
        return self._Pz0

    @property
    def mQ(self) -> np.ndarray:
        return self._mQ

    @mQ.setter
    def mQ(self, new_Q: np.ndarray) -> None:
        new_Q = np.array(new_Q, dtype=float)
        if __debug__:
            assert new_Q.shape == (
                self.dim_xy,
                self.dim_xy,
            ), f"mQ must be ({self.dim_xy},{self.dim_xy})"
        self._mQ = new_Q
        if __debug__:
            if not self.augmented:
                for arr in [self._mQ]:
                    report = CovarianceMatrix(arr).check()  # single diagnostic call
                    if not report.is_valid:
                        raise ValueError(f"Matrix  is not positive semi-definite.")
                # check_consistency(mQ=self._mQ)
        logger.info("[ParamNonLinear] ✅ mQ matrix updated")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    def summary(self) -> None:
        """Display a complete summary of vectors and matrices."""

        def fmt(M: Any) -> str:
            return np.array2string(M, formatter={"float_kind": lambda x: f"{x:6.2f}"})

        print("=== ParamNonLinear Summary ===")
        print(f"dim_x={self.dim_x}, dim_y={self.dim_y}, verbose={self.verbose}\n")
        print("g:\n", self.g)
        print("mQ:\n", fmt(self.mQ))
        print("mz0:\n", fmt(self.mz0))
        print("Pz0:\n", fmt(self.Pz0))

        if self.verbose > 0:
            print("========================")
            print("  Q_xx:\n", fmt(self._mQ[: self.dim_x, : self.dim_x]))
            print(
                "  Q_yy:\n",
                fmt(self._mQ[self.dim_x : self.dim_xy, self.dim_x : self.dim_xy]),
            )
        print("========================")
        if self.verbose > 1:  # Ready to copy in python code
            print("mQ = np.array(", repr(self.mQ.tolist()), ")")
            print("mz0 = np.array(", repr(self.mz0.tolist()), ")")
            print("Pz0 = np.array(", repr(self.Pz0.tolist()), ")")
