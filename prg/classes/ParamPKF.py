#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import warnings
from typing import Callable, Any

import numpy as np
from scipy.linalg import solve_discrete_lyapunov

import path, sys
directory = path.Path(__file__)
print(directory.parent)
sys.path.append(directory.parent.parent)

# ----------------------------------------------------------------------
# Configuration du logging global
# ----------------------------------------------------------------------
logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Classe ActiveView
# ----------------------------------------------------------------------
class ActiveView:
    """
    Vue sur une sous-matrice de `parent_matrix`.
    Appelle `callback()` à chaque modification.
    """

    def __init__(self, parent_matrix: np.ndarray, rows, cols, callback: Callable[[], None]):
        self._parent = parent_matrix
        self._rows = rows
        self._cols = cols
        self._callback = callback

    def __getitem__(self, key):
        return self._parent[self._rows, self._cols][key]

    def __setitem__(self, key, value):
        self._parent[self._rows, self._cols][key] = value
        self._callback()

    def __repr__(self):
        return repr(self._parent[self._rows, self._cols])

    @property
    def value(self):
        return self._parent[self._rows, self._cols]

# ----------------------------------------------------------------------
# Classe ParamPKF
# ----------------------------------------------------------------------
class ParamPKF:
    """
    Contient et gère les paramètres d’un filtre de Kalman couple.
    Met automatiquement à jour les matrices dérivées à chaque modification.
    """
    
    def __init__(self, dim_x, dim_y, verbose, **kwargs):
        
        if not isinstance(dim_y, int) or dim_y <= 0:
            raise ValueError("dim_y doit être un entier > 0")
        if not isinstance(dim_x, int) or dim_x <= 0:
            raise ValueError("dim_x doit être un entier > 0")
        if verbose not in [0, 1, 2]:
            raise ValueError("verbose doit être 0, 1 ou 2")

        self.dim_y   = dim_y
        self.dim_x   = dim_x
        self.dim_xy  = dim_x + dim_y
        self.verbose = verbose
        
        # Configuration du logger selon verbose
        self._set_log_level()
        
        # print(kwargs)
        # print(kwargs['A'])
        
        # Deux façons de construire un objet de cette classe
        if len(kwargs.keys()) == 2:                                # parametrization (A, mQ)
            self.constructorFrom_A_mQ(kwargs['A'], kwargs['mQ'])
        elif len(kwargs.keys()) == 7:                              # parametrization (sxx, syy, a, b, c, d, e) --> Sigma
            self.constructorFrom_Sigma(kwargs['sxx'], kwargs['syy'], kwargs['a'], kwargs['b'], kwargs['c'], kwargs['d'], kwargs['e'])

        self._mu0 = np.zeros(self.dim_xy)

    def constructorFrom_A_mQ(self, A, mQ):

        # --- Initialisation des matrices ---
        self._A = np.array(A, dtype=float)
        if self._A.shape != (self.dim_xy, self.dim_xy):
            raise ValueError(f"A doit être carrée de dimension ({self.dim_xy},{self.dim_xy})")
        self._update_A_views()

        self._mQ = np.array(mQ, dtype=float)
        if self._mQ.shape != (self.dim_xy, self.dim_xy):
            raise ValueError(f"mQ doit être carrée de dimension ({self.dim_xy},{self.dim_xy})")
        self._update_mQ_views()

        self._update_Sigma_from_A_mQ()
        self._check_consistency()

    def constructorFrom_Sigma(self, sxx, syy, a, b, c, d, e):
        # --- Initialisation des matrices ---
        self._sxx = np.array(sxx, dtype=float)
        if self._sxx.shape != (self.dim_x, self.dim_x):
            raise ValueError(f"sxx doit être carrée de dimension ({self.dim_x},{self.dim_x})")
        self._b = np.array(b, dtype=float)
        if self._b.shape != (self.dim_y, self.dim_x):
            raise ValueError(f"b doit être carrée de dimension ({self.dim_y},{self.dim_x})")
        self._syy = np.array(syy, dtype=float)
        if self._syy.shape != (self.dim_y, self.dim_y):
            raise ValueError(f"syy doit être carrée de dimension ({self.dim_y},{self.dim_y})")
        self._a = np.array(a, dtype=float)
        if self._a.shape != (self.dim_x, self.dim_x):
            raise ValueError(f"a doit être carrée de dimension ({self.dim_x},{self.dim_x})")
        self._d = np.array(d, dtype=float)
        if self._d.shape != (self.dim_y, self.dim_x):
            raise ValueError(f"d doit être carrée de dimension ({self.dim_y},{self.dim_x})")
        self._e = np.array(e, dtype=float)
        if self._e.shape != (self.dim_x, self.dim_y):
            raise ValueError(f"e doit être carrée de dimension ({self.dim_x},{self.dim_y})")
        self._c = np.array(c, dtype=float)
        if self._c.shape != (self.dim_y, self.dim_y):
            raise ValueError(f"c doit être carrée de dimension ({self.dim_y},{self.dim_y})")

        self._update_A_mQ_from_Sigma()
        


    def __repr__(self):
        return f"<ParamPKF(mode={self.mode}, dim_y={self.dim_y}, dim_x={self.dim_x}, verbose={self.verbose})>"
       

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
    # Mise à jour des matrices dérivées
    # ------------------------------------------------------------------
    def _update_A_mQ_from_Sigma(self):
        """Met à jour A, mQ (et Q1, Q2) à partir de Sigma."""
        
        # construction de Sigma
        self._Q1 = np.block([
            [self._sxx, self._b.T],
            [self._b, self._syy]
        ])
        self._Q2 = np.block([
            [self._a, self._e],
            [self._d, self._c]
        ])

        self._Sigma = np.block([
            [self._Q1, self._Q2.T],
            [self._Q2, self._Q1],
        ])
        
        # calcul de A, mQ
        self._A  = self._Q2 @ np.linalg.inv(self._Q1)
        self._update_A_views()
        self._mQ = self._Q1 - self._A @ self._Q2.T
        self._update_mQ_views()
    
    def _update_Sigma_from_A_mQ(self):
        """Met à jour Q1, Q2 et Sigma à partir de A et mQ."""
        
        self._Q1 = solve_discrete_lyapunov(self._A, self._mQ)
        self._Q2 = self._A @ self._Q1
        self._Sigma = np.block([
            [self._Q1, self._Q2.T],
            [self._Q2, self._Q1],
        ])

        # Vérifie la cohérence Q ≈ Q1 - A Q2^T
        Q_est = self._Q1 - self._A @ self._Q2.T
        diff = self._mQ - Q_est
        rel_error = np.linalg.norm(diff) / (np.linalg.norm(self._mQ) + 1e-12)

        if rel_error > 1e-8:
            logger.warning(f"Incohérence : Q ≉ Q1 - A Q2^T (erreur relative = {rel_error:.2e})")
            if self.verbose >= 2:
                logger.debug(f"Différence :\n{diff}")
        else:
            logger.debug(f"Vérification OK : ||Q - (Q1 - A Q2^T)||_rel = {rel_error:.2e}")

        # Vues dérivées
        self._a = self._Sigma[self.dim_xy:self.dim_xy+self.dim_x,   0:self.dim_x]
        self._b = self._Sigma[self.dim_x:self.dim_xy,               0:self.dim_x]
        self._c = self._Sigma[self.dim_xy+self.dim_x:2*self.dim_xy, self.dim_x:self.dim_xy]
        self._d = self._Sigma[self.dim_xy+self.dim_x:2*self.dim_xy, 0:self.dim_x]
        self._e = self._Sigma[self.dim_xy:self.dim_xy+self.dim_x,   self.dim_x:self.dim_xy]

        # Blocs diagonaux
        self._sxx = self._Sigma[0:self.dim_x,           0:self.dim_x]
        self._syy = self._Sigma[self.dim_x:self.dim_xy, self.dim_x:self.dim_xy]

    # ------------------------------------------------------------------
    # Vues dynamiques sur A et Q
    # ------------------------------------------------------------------
    def _update_A_views(self):
        def _callback():
            self._update_Sigma_from_A_mQ()
            self._check_consistency()
            logger.debug("[ActiveView] A mis à jour → recalcul Sigma")
        self._A_xx = ActiveView(self._A, slice(0, self.dim_x),           slice(0, self.dim_x),           _callback)
        self._A_xy = ActiveView(self._A, slice(0, self.dim_x),           slice(self.dim_x, self.dim_xy), _callback)
        self._A_yx = ActiveView(self._A, slice(self.dim_x, self.dim_xy), slice(0, self.dim_x),           _callback)
        self._A_yy = ActiveView(self._A, slice(self.dim_x, self.dim_xy), slice(self.dim_x, self.dim_xy), _callback)

    def _update_mQ_views(self):
        def _callback():
            self._update_Sigma_from_A_mQ()
            self._check_consistency()
            logger.debug("[ActiveView] mQ mis à jour → recalcul Sigma")
        self._mQ_xx = ActiveView(self._mQ, slice(0, self.dim_x),           slice(0, self.dim_x),           _callback)
        self._mQ_xy = ActiveView(self._mQ, slice(0, self.dim_x),           slice(self.dim_x, self.dim_xy), _callback)
        self._mQ_yx = ActiveView(self._mQ, slice(self.dim_x, self.dim_xy), slice(0, self.dim_x),           _callback)
        self._mQ_yy = ActiveView(self._mQ, slice(self.dim_x, self.dim_xy), slice(self.dim_x, self.dim_xy), _callback)

    # ------------------------------------------------------------------
    # Vérification de cohérence
    # ------------------------------------------------------------------
    def _check_consistency(self):
        """Vérifie la cohérence interne des matrices (symétrie, PSD)."""

        def _is_covariance(M: np.ndarray, name: str):
            if not np.allclose(M, M.T, atol=1e-12):
                logger.warning(f"{name} n'est pas symétrique")
            eigvals = np.linalg.eigvals(M)
            if np.any(eigvals < -1e-12):
                logger.warning(f"{name} n'est pas PSD (min eig = {eigvals.min():.3e})")
            logger.debug(f"Valeurs propres de {name} : {eigvals}")

        _is_covariance(self._mQ, "mQ")
        if hasattr(self, "_Q1"): _is_covariance(self._Q1, "Q1")
        if hasattr(self, "_Sigma"): _is_covariance(self._Sigma, "Sigma")

    # ------------------------------------------------------------------
    # Setters
    # ------------------------------------------------------------------
    # --- Matrices calculées automatiquement ---
    @property
    def Q1(self): return self._Q1
    @property
    def Q2(self): return self._Q2
    @property
    def Sigma(self): return self._Sigma
    @property
    def mu0(self): return self._mu0

    # --- Sous-blocs de Sigma (lecture seule) ---
    @property
    def sxx(self): return self._sxx
    @property
    def syy(self): return self._syy
    @property
    def a(self): return self._a
    @property
    def b(self): return self._b
    @property
    def c(self): return self._c
    @property
    def d(self): return self._d
    @property
    def e(self): return self._e
    
    @property
    def A(self): return self._A
    @A.setter
    def A(self, new_A):
        new_A = np.array(new_A, dtype=float)
        if new_A.shape != (self.dim_xy, self.dim_xy):
            raise ValueError(f"A doit être ({self.dim_xy},{self.dim_xy})")
        self._A = new_A
        self._update_A_views()
        self._update_Sigma_from_A_mQ()
        self._check_consistency()
        logger.info("[ParamPKF] A mis à jour")
    
    # --- Sous-blocs de A (lecture seule) ---
    @property
    def A_xx(self): return self._A_xx
    @property
    def A_xy(self): return self._A_xy
    @property
    def A_yx(self): return self._A_yx
    @property
    def A_yy(self): return self._A_yy

    @property
    def mQ(self): return self._mQ
    @mQ.setter
    def mQ(self, new_Q):
        new_Q = np.array(new_Q, dtype=float)
        if new_Q.shape != (self.dim_xy, self.dim_xy):
            raise ValueError(f"mQ doit être ({self.dim_xy},{self.dim_xy})")
        self._mQ = new_Q
        self._update_mQ_views()
        self._update_Sigma_from_A_mQ()
        self._check_consistency()
        logger.info("[ParamPKF] mQ mis à jour")
    # --- Sous-blocs de mQ (lecture seule) ---
    @property
    def Q_xx(self): return self._mQ_xx
    @property
    def Q_xy(self): return self._mQ_xy
    @property
    def Q_yx(self): return self._mQ_yx
    @property
    def Q_yy(self): return self._mQ_yy


    # ------------------------------------------------------------------
    # Résumé
    # ------------------------------------------------------------------
    def summary(self):
        """Affiche un résumé complet des matrices."""
        def fmt(M: Any) -> str:
            if hasattr(M, "_parent"):
                M = M._parent[M._rows, M._cols]
            return np.array2string(M, formatter={'float_kind': lambda x: f"{x:6.2f}"})

        print("=== ParamPKF Summary ===")
        print(f"dim_x={self.dim_x}, dim_y={self.dim_y}, verbose={self.verbose}\n")
        print("A:\n", fmt(self.A))
        print("mQ:\n", fmt(self.mQ))
        print("Sigma:\n", fmt(self._Sigma))
        if verbose>0:
            print("========================")
            print("Q1:\n", fmt(self._Q1))
            print("Q2:\n", fmt(self._Q2))
            print("========================")
            print("sxx:\n", fmt(self._sxx))
            print("syy:\n", fmt(self._syy))
            print("a:\n", fmt(self._a))
            print("b:\n", fmt(self._b))
            print("c:\n", fmt(self._c))
            print("d:\n", fmt(self._d))
            print("e:\n", fmt(self._e))
            print("========================")
        if verbose>1:
            print("A = np.array(",  repr(self.A.tolist()), ')')
            print("mQ = np.array(", repr(self.mQ.tolist()), ')')
        # self._check_consistency()
        


# ----------------------------------------------------------------------
# Exemple d'utilisation
# ----------------------------------------------------------------------
if __name__ == "__main__":
    verbose = 1

    # ------------------------------------------------------------------
    # Test parameters
    # ------------------------------------------------------------------
    # from models.model_dimx1_dimy1 import model_dimx1_dimy1_from_Sigma
    # dim_x, dim_y, sxx, syy, a, b, c, d, e = model_dimx1_dimy1_from_Sigma()
    # param = ParamPKF(dim_x, dim_y, verbose, sxx=sxx, syy=syy, a=a, b=b, c=c, d=d, e=e)
    # if verbose > 0:
    #     param.summary()
    
    # ------------------------------------------------------------------
    # dim_x = dim_y = 1 - Test parameters for (A, mQ) parametrization
    # ------------------------------------------------------------------
    from models.model_dimx1_dimy1 import model_dimx1_dimy1_from_A_mQ
    dim_x, dim_y, A, mQ = model_dimx1_dimy1_from_A_mQ()
    param_A_mQ = ParamPKF(dim_x, dim_y, verbose, A=A, mQ=mQ)
    if verbose > 0:
      param_A_mQ.summary()

    # ------------------------------------------------------------------
    # dim_x = dim_y = 2 - Test parameters for (Sigma = (sxx, syy, a, b, c, d, e)) parametrization
    # ------------------------------------------------------------------
    # from models.model_dimx2_dimy2 import model_dimx2_dimy2_from_Sigma
    # dim_x, dim_y, sxx, syy, a, b, c, d, e = model_dimx2_dimy2_from_Sigma()
    # param_Sigma = ParamPKF(dim_x, dim_y, verbose, sxx=sxx, syy=syy, a=a, b=b, c=c, d=d, e=e)
    # if verbose > 0:
    #   param_Sigma.summary()
    
    # ------------------------------------------------------------------
    # dim_x = dim_y = 2 - Test parameters for (A, mQ) parametrization
    # ------------------------------------------------------------------
    # from models.model_dimx2_dimy2 import model_dimx2_dimy2_from_A_mQ
    # dim_x, dim_y, A, mQ = model_dimx2_dimy2_from_A_mQ()
    # param_A_mQ = ParamPKF(dim_x, dim_y, verbose, A=A, mQ=mQ)
    # if verbose > 0:
    #   param_A_mQ.summary()

    # ------------------------------------------------------------------
    # dim_x = 3, dim_y = 1 - Test parameters for (Sigma = (sxx, syy, a, b, c, d, e)) parametrization
    # ------------------------------------------------------------------
    # from models.model_dimx3_dimy1 import model_dimx3_dimy1_from_Sigma
    # dim_x, dim_y, sxx, syy, a, b, c, d, e = model_dimx3_dimy1_from_Sigma()
    # param_Sigma = ParamPKF(dim_x, dim_y, verbose, sxx=sxx, syy=syy, a=a, b=b, c=c, d=d, e=e)
    # if verbose > 0:
    #     param_Sigma.summary()
    
    # ------------------------------------------------------------------
    # dim_x = 3, dim_y = 1 - Test parameters for (A, mQ) parametrization
    # ------------------------------------------------------------------
    # from models.model_dimx3_dimy1 import model_dimx3_dimy1_from_A_mQ
    # dim_x, dim_y, A, mQ = model_dimx3_dimy1_from_A_mQ()
    # param_A_mQ = ParamPKF(dim_x, dim_y, verbose, A=A, mQ=mQ)
    # if verbose > 0:
    #     param_A_mQ.summary()
    
