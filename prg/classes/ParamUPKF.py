#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import path, sys
directory = path.Path(__file__)
sys.path.append(directory.parent.parent)

import logging
import warnings
from typing import Callable, Any

import numpy as np

from classes.ParamPKF import ActiveView

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
    Manage UPKF parameters
    """
    
    def __init__(self, dim_x, dim_y, verbose, g, mQ, z00, Pz00, alpha, beta, kappa):
        
        if not isinstance(dim_y, int) or dim_y <= 0:
            raise ValueError("⚠️ dim_y doit être un entier > 0")
        if not isinstance(dim_x, int) or dim_x <= 0:
            raise ValueError("⚠️ dim_x doit être un entier > 0")
        if verbose not in [0, 1, 2]:
            raise ValueError("⚠️ verbose doit être 0, 1 ou 2")

        self.dim_y   = dim_y
        self.dim_x   = dim_x
        self.dim_xy  = dim_x + dim_y
        self.verbose = verbose
        
        # Configuration du logger selon verbose
        self._set_log_level()
        
        # Parameters
        self.g     = g
        self._z00  = z00
        self._Pz00 = Pz00
        self._mQ   = np.array(mQ, dtype=float)
        self._update_mQ_views()
        
        # Paramètres UKF
        self._alpha   = alpha
        self._beta    = beta
        self._kappa   = kappa
        self._lambda_ = alpha**2 * (self.dim_x + kappa) - self.dim_x
        self._gamma   = np.sqrt(self.dim_x + self._lambda_)

    def __repr__(self):
        return f"<ParamUPKF(  dim_y={self.dim_y}, dim_x={self.dim_x}, verbose={self.verbose}, \
                            \talpha={self.alpha}, beta ={self.beta},   kappa ={self.kappa})>"

    # ------------------------------------------------------------------
    # Gestion du logging selon le niveau de verbosité
    # ------------------------------------------------------------------
    def _set_log_level(self):
        if self.verbose == 0:
            logger.setLevel(logging.WARNING)
        elif self.verbose == 1:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.DEBUG)

    # ------------------------------------------------------------------
    # Vues dynamiques sur Q
    # ------------------------------------------------------------------

    def _update_mQ_views(self):
        def _callback():
            self._check_consistency()
            logger.debug("[ActiveView] ✅ mQ matrices updated")
        self._mQ_xx = ActiveView(self._mQ, slice(0, self.dim_x),           slice(0, self.dim_x),           _callback)
        self._mQ_xy = ActiveView(self._mQ, slice(0, self.dim_x),           slice(self.dim_x, self.dim_xy), _callback)
        self._mQ_yx = ActiveView(self._mQ, slice(self.dim_x, self.dim_xy), slice(0, self.dim_x),           _callback)
        self._mQ_yy = ActiveView(self._mQ, slice(self.dim_x, self.dim_xy), slice(self.dim_x, self.dim_xy), _callback)

    # ------------------------------------------------------------------
    # Vérification de cohérence
    # ------------------------------------------------------------------
    def _check_consistency(self):
        """Check the internal consistency of the matrices (symmetry, PSD)."""

        def _is_covariance(M: np.ndarray, name: str):
            if not np.allclose(M, M.T, atol=1e-12):
                logger.warning(f"⚠️ {name} matrix is not symmetrical")
            eigvals = np.linalg.eigvals(M)
            if np.any(eigvals < -1e-12):
                logger.warning(f"⚠️ {name} matrix is not positive semi-definite (min eig = {eigvals.min():.3e})")
            logger.debug(f"Eig of {name} matrix: {eigvals}")
        
        if hasattr(self, "_mQ"):   _is_covariance(self._mQ,   "mQ")
        # if hasattr(self, "_Pz00"): _is_covariance(self._Pz00, "Pz00")

    # ------------------------------------------------------------------
    # Setters/Getters
    # ------------------------------------------------------------------
    @property
    def alpha(self):   return self._alpha
    @property
    def lambda_(self): return self._lambda_
    @property
    def gamma(self):   return self._gamma
    @property
    def beta(self):    return self._beta
    @property
    def kappa(self):   return self._kappa
    @property
    def z00(self):     return self._z00
    @property
    def Pz00(self):    return self._Pz00
    
    @property
    def mQ(self): return self._mQ
    @mQ.setter
    def mQ(self, new_Q):
        new_Q = np.array(new_Q, dtype=float)
        if new_Q.shape != (self.dim_xy, self.dim_xy):
            raise ValueError(f"⚠️ mQ doit être ({self.dim_xy},{self.dim_xy})")
        self._mQ = new_Q
        self._update_mQ_views()
        self._check_consistency()
        logger.info("[ParamUPKF] ✅ mQ matrix updated")
    # --- Sous-blocs de mQ (lecture seule) ---
    @property
    def mQ_xx(self): return self._mQ_xx
    @property
    def mQ_xy(self): return self._mQ_xy
    @property
    def mQ_yx(self): return self._mQ_yx
    @property
    def mQ_yy(self): return self._mQ_yy

    # ------------------------------------------------------------------
    # Résumé
    # ------------------------------------------------------------------
    def summary(self):
        """Affiche un résumé complet des matrices."""
        def fmt(M: Any) -> str:
            if hasattr(M, "_parent"):
                M = M._parent[M._rows, M._cols]
            return np.array2string(M, formatter={'float_kind': lambda x: f"{x:6.2f}"})

        print("=== ParamUPKF Summary ===")
        print(f"dim_x={self.dim_x}, dim_y={self.dim_y}, verbose={self.verbose}\n")
        print("g:\n",     self.g)
        print("mQ:\n",    fmt(self.mQ))
        print("z00:\n",   fmt(self.z00))
        print("Pz00:\n",  fmt(self.Pz00))
        if self.verbose>0:
            print("========================")
            print("  Q_xx:\n  ",  fmt(self.mQ_xx))
            print("  Q_yy:\n  ",  fmt(self.mQ_yy))
            print("========================")
        if self.verbose>1: # Ready to copy in python code
            print("mQ = np.array(",   repr(self.mQ.tolist()),   ')')
            print("z00 = np.array(",  repr(self.z00.tolist()), ')')
            print("Pz00 = np.array(", repr(self.Pz00.tolist()), ')')
        self._check_consistency()


# ----------------------------------------------------------------------
# Main program
# ----------------------------------------------------------------------
if __name__ == "__main__":
    verbose = 1

    # ------------------------------------------------------------------
    # Test parameters
    # ------------------------------------------------------------------
    from models.nonLinear.nonLinear_x1_y1 import model_x1_y1_ext_saturant
    dim_x, dim_y, g, mQ, z00, Pz00, alpha, beta, kappa = model_x1_y1_ext_saturant()
    from models.nonLinear.nonLinear_x1_y1 import model_x1_y1_cubique
    dim_x, dim_y, g, mQ, z00, Pz00, alpha, beta, kappa = model_x1_y1_cubique()
    from models.nonLinear.nonLinear_x1_y1 import model_x1_y1_sinus
    dim_x, dim_y, g, mQ, z00, Pz00, alpha, beta, kappa = model_x1_y1_sinus()
    from models.nonLinear.nonLinear_x1_y1 import model_x1_y1_gordon
    dim_x, dim_y, g, mQ, z00, Pz00, alpha, beta, kappa = model_x1_y1_gordon()
    from models.nonLinear.nonLinear_x2_y1 import model_x2_y1
    dim_x, dim_y, g, mQ, z00, Pz00, alpha, beta, kappa = model_x2_y1()

    param = ParamUPKF(dim_x, dim_y, verbose, g, mQ, z00, Pz00, alpha, beta, kappa)
    if verbose > 0:
        param.summary()
    
