#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import sys

directory = Path(__file__)
sys.path.append(str(directory.parent.parent))

import logging
from typing import Callable, Any, Union, Optional
import warnings

import numpy as np
from scipy.linalg import solve_discrete_lyapunov, cho_factor, cho_solve

# Linear models
from models.linear import BaseModelLinear, ModelFactoryLinear
from classes.ActiveView import ActiveView
# A few utils functions that are used several times
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

    def __init__(
        self,
        verbose: int,
        dim_x:   int,
        dim_y:   int,
        **kwargs ) -> None:
        
        if __debug__:
            assert isinstance(dim_x, int) and dim_x > 0, "dim_x must be int > 0"
            assert isinstance(dim_y, int) and dim_y > 0, "dim_y must be int > 0"
            assert verbose in [0, 1, 2], "verbose must be 0, 1 or 2"

        self.dim_y   = dim_y
        self.dim_x   = dim_x
        self.dim_xy  = dim_x + dim_y
        self.verbose = verbose

        # Logger config according to verbose
        self._set_log_level()

        # Deux façons de construire un objet de cette classe
        if len(kwargs.keys()) == 6:  # parametrization (A, mQ, z00, Pz00)
            self.constructorFrom_A_mQ(kwargs['g'], kwargs['A'], kwargs['mQ'], kwargs['z00'], kwargs['Pz00'], kwargs['augmented'])
        elif len(kwargs.keys()) == 9:  # parametrization (sxx, syy, a, b, c, d, e) --> Sigma
            self.constructorFrom_Sigma(kwargs['g'], kwargs['sxx'], kwargs['syy'], kwargs['a'], kwargs['b'], kwargs['c'], kwargs['d'], kwargs['e'], kwargs['augmented'])
        else:
            logger.warning(f"⚠️ Le modèle n'est pas bien paramétré : {kwargs.keys()}")

        # Check dimensions of all matrices
        if __debug__:
            self._check_dimensions()


    # ------------------------------------------------------------------
    # Constructeurs
    # ------------------------------------------------------------------
    def constructorFrom_A_mQ(self, g, A: np.ndarray, mQ: np.ndarray, z00: np.ndarray, Pz00: np.ndarray, augmented: bool) -> None:
        
        # Est-ce un modèle augmenté ?
        self.augmented = augmented
        
        # The linear equation to update the system
        self.g = g
        
        self._A = np.array(A, dtype=float)
        if __debug__:
            eigvals = np.linalg.eigvals(self._A)
            if np.any(np.abs(eigvals) >= 1.0):
                logger.warning(f"⚠️ Certaines valeurs propres de A ont un module >= 1 : {eigvals}")
        self._update_A_views()

        self._mQ = np.array(mQ, dtype=float)
        self._update_mQ_views()

        self._z00  = np.array(z00,  dtype=float)
        self._Pz00 = np.array(Pz00, dtype=float)

        self._update_Sigma_from_A_mQ()
        self._check_consistency()

    def constructorFrom_Sigma(self, g, sxx: np.ndarray, syy: np.ndarray, a: np.ndarray, b: np.ndarray,
                              c: np.ndarray, d: np.ndarray, e: np.ndarray, augmented: bool) -> None:
        
        # Est-ce un modèle augmenté ?
        self.augmented = augmented
        
        # The linear function to update the system equations
        self.g = g
        
        self._sxx = np.array(sxx, dtype=float)
        self._syy = np.array(syy, dtype=float)
        self._a   = np.array(a,   dtype=float)
        self._b   = np.array(b,   dtype=float)
        self._c   = np.array(c,   dtype=float)
        self._d   = np.array(d,   dtype=float)
        self._e   = np.array(e,   dtype=float)

        self._update_A_mQ_from_Sigma()
        self._check_consistency()

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def _set_log_level(self) -> None:
        if self.verbose==0 or self.verbose==1:
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
            'A':     (  self.dim_xy,   self.dim_xy),
            'mQ':    (  self.dim_xy,   self.dim_xy),
            'z00':   (  self.dim_xy,             1),
            'Pz00':  (  self.dim_xy,   self.dim_xy),
            'Q1':    (  self.dim_xy,   self.dim_xy),
            'Q2':    (  self.dim_xy,   self.dim_xy),
            'Sigma': (2*self.dim_xy, 2*self.dim_xy),
            'sxx':   (  self.dim_x,    self.dim_x ),
            'syy':   (  self.dim_y,    self.dim_y ),
            'a':     (  self.dim_x,    self.dim_x ),
            'b':     (  self.dim_y,    self.dim_x ),
            'c':     (  self.dim_y,    self.dim_y ),
            'd':     (  self.dim_y,    self.dim_x ),
            'e':     (  self.dim_x,    self.dim_y ),
        }
        for attr, shape in expected_shapes.items():
            if hasattr(self, f"_{attr}"):
                actual = getattr(self, f"_{attr}")
                if actual.shape != shape:
                    raise ValueError(f"⚠️ Matrice {attr} a une forme {actual.shape}, attendue {shape}")

    # ------------------------------------------------------------------
    # Update derived matrices
    # ------------------------------------------------------------------
    def _update_A_mQ_from_Sigma(self) -> None:
        
        self._Q1    = np.block([[self._sxx, self._b.T], [self._b, self._syy]])
        self._Q2    = np.block([[self._a, self._e], [self._d, self._c]])
        self._Sigma = np.block([[self._Q1, self._Q2.T], [self._Q2, self._Q1]])
        
        check_consistency(_Q1=self._Q1, _Sigma=self._Sigma)

        # self._A = self._Q2 @ np.linalg.inv(self._Q1)
        c, low = cho_factor(self._Q1)
        self._A = self._Q2 @ cho_solve((c, low), np.eye(self.dim_xy))

        if __debug__:
            eigvals = np.linalg.eigvals(self._A)
            if np.any(np.abs(eigvals) >= 1.0):
                logger.warning(f"⚠️ Certaines valeurs propres de A ont un module >= 1 : {eigvals}")
        self._update_A_views()
        
        self._mQ = self._Q1 - self._A @ self._Q2.T
        check_consistency(_mQ=self._mQ)
        self._update_mQ_views()

        self._z00  = np.zeros((self.dim_xy, 1))
        self._Pz00 = self._Q1.copy()

        if __debug__:
            self._check_dimensions()

    def _update_Sigma_from_A_mQ(self) -> None:
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

        if __debug__:
            self._check_dimensions()

    # ------------------------------------------------------------------
    # Dynamic views on A and mQ
    # ------------------------------------------------------------------
    def _update_A_views(self) -> None:
        def _callback() -> None:
            self._update_Sigma_from_A_mQ()
            if __debug__:
                self._check_consistency()
                logger.info("[ActiveView] ✅ A, Sigma matrice updated")

        self._A_xx = ActiveView(self._A, slice(0, self.dim_x), slice(0, self.dim_x), _callback)
        self._A_xy = ActiveView(self._A, slice(0, self.dim_x), slice(self.dim_x, self.dim_xy), _callback)
        self._A_yx = ActiveView(self._A, slice(self.dim_x, self.dim_xy), slice(0, self.dim_x), _callback)
        self._A_yy = ActiveView(self._A, slice(self.dim_x, self.dim_xy), slice(self.dim_x, self.dim_xy), _callback)

    def _update_mQ_views(self) -> None:
        def _callback() -> None:
            self._update_Sigma_from_A_mQ()
            if __debug__:
                self._check_consistency()
                logger.debug("[ActiveView] ✅ mQ, Sigma matrices updated")

        self._mQ_xx = ActiveView(self._mQ, slice(0, self.dim_x), slice(0, self.dim_x), _callback)
        self._mQ_xy = ActiveView(self._mQ, slice(0, self.dim_x), slice(self.dim_x, self.dim_xy), _callback)
        self._mQ_yx = ActiveView(self._mQ, slice(self.dim_x, self.dim_xy), slice(0, self.dim_x), _callback)
        self._mQ_yy = ActiveView(self._mQ, slice(self.dim_x, self.dim_xy), slice(self.dim_x, self.dim_xy), _callback)

    # ------------------------------------------------------------------
    # Consistency checks
    # ------------------------------------------------------------------
    """Check internal matrices for symmetry and positive semi-definiteness."""
    def _check_consistency(self) -> None:
        for attr, name in [('_mQ', 'mQ'), ('_Q1', 'Q1'), ('_Sigma', 'Sigma'),
                           ('_sxx', 'sxx'), ('_syy', 'syy'), ('_Pz00', 'Pz00')]:
            if hasattr(self, attr):
                is_covariance(getattr(self, attr), name)

    # ------------------------------------------------------------------
    # Getters / Setters and Properties
    # ------------------------------------------------------------------
    # @property
    # def g(self): return self._g

    @property
    def Q1(self) -> np.ndarray: return self._Q1
    @property
    def Q2(self) -> np.ndarray: return self._Q2
    @property
    def Sigma(self) -> np.ndarray: return self._Sigma
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

    @property
    def A(self) -> np.ndarray: return self._A
    @A.setter
    def A(self, new_A: np.ndarray) -> None:
        new_A = np.array(new_A, dtype=float)
        if new_A.shape != (self.dim_xy, self.dim_xy):
            raise ValueError(f"⚠️ A doit être ({self.dim_xy},{self.dim_xy})")
        self._A = new_A
        self._update_A_views()
        self._update_Sigma_from_A_mQ()
        self._check_consistency()
        if __debug__:
            logger.info("[ParamLinear] ✅ A matrix updates")

    @property
    def A_xx(self) -> ActiveView: return self._A_xx
    @property
    def A_xy(self) -> ActiveView: return self._A_xy
    @property
    def A_yx(self) -> ActiveView: return self._A_yx
    @property
    def A_yy(self) -> ActiveView: return self._A_yy

    @property
    def mQ(self) -> np.ndarray: return self._mQ
    @mQ.setter
    def mQ(self, new_Q: np.ndarray) -> None:
        new_Q = np.array(new_Q, dtype=float)
        if new_Q.shape != (self.dim_xy, self.dim_xy):
            raise ValueError(f"⚠️ mQ doit être ({self.dim_xy},{self.dim_xy})")
        self._mQ = new_Q
        self._update_mQ_views()
        self._update_Sigma_from_A_mQ()
        self._check_consistency()
        if __debug__:
            logger.info("[ParamLinear] ✅ mQ matrix updated")

    @property
    def mQ_xx(self) -> ActiveView: return self._mQ_xx
    @property
    def mQ_xy(self) -> ActiveView: return self._mQ_xy
    @property
    def mQ_yx(self) -> ActiveView: return self._mQ_yx
    @property
    def mQ_yy(self) -> ActiveView: return self._mQ_yy

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    def summary(self) -> None:
        def fmt(M: Any) -> str:
            if hasattr(M, "_parent"):
                M = M._parent[M._rows, M._cols]
            return np.array2string(M, formatter={'float_kind': lambda x: f"{x:6.2f}"})

        print("=== ParamLinear Summary ===")
        print(f"dim_x={self.dim_x}, dim_y={self.dim_y}, verbose={self.verbose}\n")
        print("A:\n", fmt(self.A))
        print("mQ:\n", fmt(self.mQ))
        print("z00:\n", fmt(self.z00))
        print("Pz00:\n", fmt(self.Pz00))
        print("Sigma:\n", fmt(self._Sigma))
        
        if self.verbose > 0:
            print("========================")
            print("  Q1:\n  ", fmt(self._Q1))
            print("  Q2:\n  ", fmt(self._Q2))
            print("========================")
            print("  sxx:\n  ", fmt(self._sxx))
            print("  syy:\n  ", fmt(self._syy))
            print("  a:\n  ", fmt(self._a))
            print("  b:\n  ", fmt(self._b))
            print("  c:\n  ", fmt(self._c))
            print("  d:\n  ", fmt(self._d))
            print("  e:\n  ", fmt(self._e))
        print("========================\n")
        if self.verbose>1:  # Ready to copy in python code
            print("A  = np.array(", repr(self.A.tolist()), ')')
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
    
    # Available : ['A_mQ_x1_y1', 'A_mQ_x3_y1', 'Sigma_x1_y1', 'Sigma_x3_y1', 'A_mQ_x2_y2', 'Sigma_x2_y2', 'A_mQ_x1_y1_VPgreaterThan1']
    model = ModelFactoryLinear.create("A_mQ_x1_y1")
    print(f'model={model}')
    print(f'model.get_params()={model.get_params()}')
    
    params = model.get_params().copy()
    param  = ParamLinear(verbose, params.pop('dim_x'), params.pop('dim_y'), **params)
    if verbose > 0:
        param.summary()
