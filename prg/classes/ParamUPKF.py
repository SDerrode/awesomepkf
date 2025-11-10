#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import path, sys
directory = path.Path(__file__)
sys.path.append(directory.parent.parent)

import sys
from pathlib import Path
import logging
from typing import Callable, Any

import numpy as np

from classes.ActiveView import ActiveView
from models.nonLinear import ModelFactory

# ----------------------------------------------------------------------
# Configuration du logging global
# ----------------------------------------------------------------------
logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# ParamUPKF class
# ----------------------------------------------------------------------
class ParamUPKF:
    """
    Manage UPKF parameters with optional debug checks.

    Attributes:
        dim_x, dim_y, dim_xy: state and observation dimensions
        g: model function _g
        mQ: process covariance matrix
        z00: initial state vector
        Pz00: initial covariance matrix
        alpha, beta, kappa: UKF parameters
        lambda_, gamma: derived UKF parameters
        verbose: logging level
    """

    def __init__(
        self,
        verbose: int,
        dim_x: int,
        dim_y: int,
        g: Callable,
        mQ: np.ndarray,
        z00: np.ndarray,
        Pz00: np.ndarray,
        alpha: float,
        beta: float,
        kappa: float
    ) -> None:

        if __debug__:
            assert isinstance(dim_x, int) and dim_x > 0, "dim_x must be int > 0"
            assert isinstance(dim_y, int) and dim_y > 0, "dim_y must be int > 0"
            assert verbose in [0, 1, 2], "verbose must be 0, 1 or 2"

        self.dim_x   = dim_x
        self.dim_y   = dim_y
        self.dim_xy  = dim_x + dim_y
        self.verbose = verbose
        self._set_log_level()

        self.g     = g
        self._z00  = np.array(z00, dtype=float)
        self._Pz00 = np.array(Pz00, dtype=float)
        self._mQ   = np.array(mQ, dtype=float)
        self._update_mQ_views()

        # UKF parameters
        self._alpha   = alpha
        self._beta    = beta
        self._kappa   = kappa
        self._lambda_ = alpha**2 * (self.dim_x + kappa) - self.dim_x
        self._gamma   = np.sqrt(self.dim_x + self._lambda_)

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"<ParamUPKF(dim_y={self.dim_y}, dim_x={self.dim_x}, verbose={self.verbose}, "
            f"alpha={self.alpha}, beta={self.beta}, kappa={self.kappa})>"
        )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def _set_log_level(self) -> None:
        if self.verbose == 0:
            logger.setLevel(logging.WARNING)
        elif self.verbose == 1:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.DEBUG)

    # ------------------------------------------------------------------
    # Dynamic views on mQ
    # ------------------------------------------------------------------
    def _update_mQ_views(self) -> None:
        def _callback():
            if __debug__:
                self._check_consistency()
            logger.debug("[ActiveView] ✅ mQ matrices updated")

        self._mQ_xx = ActiveView(self._mQ, slice(0, self.dim_x), slice(0, self.dim_x), _callback)
        self._mQ_xy = ActiveView(self._mQ, slice(0, self.dim_x), slice(self.dim_x, self.dim_xy), _callback)
        self._mQ_yx = ActiveView(self._mQ, slice(self.dim_x, self.dim_xy), slice(0, self.dim_x), _callback)
        self._mQ_yy = ActiveView(self._mQ, slice(self.dim_x, self.dim_xy), slice(self.dim_x, self.dim_xy), _callback)

    # ------------------------------------------------------------------
    # Consistency checks
    # ------------------------------------------------------------------
    def _check_consistency(self) -> None:
        """Check internal matrices for symmetry and positive semi-definiteness."""
        def _is_covariance(M: np.ndarray, name: str) -> None:
            if not np.allclose(M, M.T, atol=1e-12):
                logger.warning(f"⚠️ {name} matrix is not symmetrical")
            eigvals = np.linalg.eigvals(M)
            if np.any(eigvals < -1e-12):
                logger.warning(f"⚠️ {name} matrix is not PSD (min eig={eigvals.min():.3e})")
            logger.debug(f"Eig of {name} matrix: {eigvals}")

        if hasattr(self, "_mQ"):
            _is_covariance(self._mQ, "mQ")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def alpha(self) -> float: return self._alpha
    @property
    def lambda_(self) -> float: return self._lambda_
    @property
    def gamma(self) -> float: return self._gamma
    @property
    def beta(self) -> float: return self._beta
    @property
    def kappa(self) -> float: return self._kappa
    @property
    def z00(self) -> np.ndarray: return self._z00
    @property
    def Pz00(self) -> np.ndarray: return self._Pz00

    @property
    def mQ(self) -> np.ndarray: return self._mQ
    @mQ.setter
    def mQ(self, new_Q: np.ndarray) -> None:
        new_Q = np.array(new_Q, dtype=float)
        if __debug__:
            assert new_Q.shape == (self.dim_xy, self.dim_xy), f"mQ must be ({self.dim_xy},{self.dim_xy})"
        self._mQ = new_Q
        self._update_mQ_views()
        if __debug__:
            self._check_consistency()
        logger.info("[ParamUPKF] ✅ mQ matrix updated")

    @property
    def mQ_xx(self): return self._mQ_xx
    @property
    def mQ_xy(self): return self._mQ_xy
    @property
    def mQ_yx(self): return self._mQ_yx
    @property
    def mQ_yy(self): return self._mQ_yy

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    def summary(self) -> None:
        """Display a complete summary of vectors and matrices."""
        def fmt(M: Any) -> str:
            if hasattr(M, "_parent"):
                M = M._parent[M._rows, M._cols]
            return np.array2string(M, formatter={'float_kind': lambda x: f"{x:6.2f}"})

        print("=== ParamUPKF Summary ===")
        print(f"dim_x={self.dim_x}, dim_y={self.dim_y}, verbose={self.verbose}\n")
        print("g:\n", self.g)
        print("mQ:\n", fmt(self.mQ))
        print("z00:\n", fmt(self.z00))
        print("Pz00:\n", fmt(self.Pz00))

        if self.verbose > 0:
            print("========================")
            print("  Q_xx:\n  ", fmt(self.mQ_xx))
            print("  Q_yy:\n  ", fmt(self.mQ_yy))
            print("========================")
        if self.verbose > 1:
            print("mQ = np.array(", repr(self.mQ.tolist()), ')')
            print("z00 = np.array(", repr(self.z00.tolist()), ')')
            print("Pz00 = np.array(", repr(self.Pz00.tolist()), ')')

        if __debug__:
            self._check_consistency()


# ----------------------------------------------------------------------
# Main program
# ----------------------------------------------------------------------
if __name__ == "__main__":
    verbose = 1

    model = ModelFactory.create("x2_y1_withRetroactionsOfObservations")
    print(f'model={model}')
    print(f'model.get_params()={model.get_params()}')

    param = ParamUPKF(*model.get_params(), verbose=verbose)
    if verbose > 0:
        param.summary()
