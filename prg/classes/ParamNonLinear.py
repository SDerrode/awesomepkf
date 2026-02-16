#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import sys

directory = Path(__file__)
sys.path.append(str(directory.parent.parent))

import logging
from typing import Any

import numpy as np

# Non linear models
from models.nonLinear import ModelFactoryNonLinear
# A few utils functions that are used several times
from others.utils import is_covariance

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
        self.augmented = kwargs['augmented']

        # Non-linear function
        self.g         = kwargs['g']

        # Covariance matrices
        self._mQ       = np.array(kwargs['mQ'], dtype=float)
        self._z00      = np.array(kwargs['z00'], dtype=float)
        self._Pz00     = np.array(kwargs['Pz00'], dtype=float)

        # UPKF specific parameters
        self.alpha     = kwargs['alpha']
        self.beta      = kwargs['beta']
        self.kappa     = kwargs['kappa']
        self.lambda_   = self.alpha**2 * (self.dim_x + self.kappa) - self.dim_x

        # EPKF specific parameters
        self.jacobiens_g = kwargs['jacobiens_g']

        if __debug__:
            self._check_dimensions()
            self._check_consistency()

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"<ParamNonLinear(dim_y={self.dim_y}, dim_x={self.dim_x}, verbose={self.verbose}, "
            f"alpha={self.alpha}, beta={self.beta}, kappa={self.kappa})>"
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
    # Check dimensions
    # ------------------------------------------------------------------
    def _check_dimensions(self) -> None:
        expected_shapes = {
            'mQ': (self.dim_xy, self.dim_xy),
            'z00': (self.dim_xy, 1),
            'Pz00': (self.dim_xy, self.dim_xy),
        }
        for attr, shape in expected_shapes.items():
            actual = getattr(self, f"_{attr}")
            if actual.shape != shape:
                raise ValueError(f"⚠️ Matrice {attr} a une forme {actual.shape}, attendue {shape}")

    # ------------------------------------------------------------------
    # Consistency checks
    # ------------------------------------------------------------------
    def _check_consistency(self) -> None:
        """Check internal matrices for symmetry and positive semi-definiteness."""
        is_covariance(self._mQ, "mQ")
        is_covariance(self._Pz00, "Pz00")

    # ------------------------------------------------------------------
    # Getters / Setters and Properties
    # ------------------------------------------------------------------
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
        if __debug__:
            self._check_consistency()
        logger.info("[ParamNonLinear] ✅ mQ matrix updated")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    def summary(self) -> None:
        """Display a complete summary of vectors and matrices."""
        def fmt(M: Any) -> str:
            return np.array2string(M, formatter={'float_kind': lambda x: f"{x:6.2f}"})

        print("=== ParamNonLinear Summary ===")
        print(f"dim_x={self.dim_x}, dim_y={self.dim_y}, verbose={self.verbose}\n")
        print("g:\n", self.g)
        print("mQ:\n", fmt(self.mQ))
        print("z00:\n", fmt(self.z00))
        print("Pz00:\n", fmt(self.Pz00))

        if self.verbose > 0:
            print("========================")
            print("  Q_xx:\n", fmt(self._mQ[0:self.dim_x, 0:self.dim_x]))
            print("  Q_yy:\n", fmt(self._mQ[self.dim_x:self.dim_xy, self.dim_x:self.dim_xy]))
        print("========================")
        if self.verbose > 1:  # Ready to copy in python code
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

    # Available non linear models:
    # ['x1_y1_cubique', 'x1_y1_ext_saturant', 'x1_y1_gordon', 'x1_y1_sinus', 'x2_y1', 'x2_y1_rapport', 'x2_y1_withRetroactionsOfObservations']
    model = ModelFactoryNonLinear.create("x2_y1_rapport")
    print(f'model={model}')
    print(f'model.get_params()={model.get_params()}')

    params = model.get_params().copy()
    param = ParamNonLinear(verbose, params.pop('dim_x'), params.pop('dim_y'), **params)
    if verbose > 0:
        param.summary()
