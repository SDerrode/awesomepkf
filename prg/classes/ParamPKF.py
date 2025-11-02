#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import warnings
from typing import Callable, Any

import numpy as np
from scipy.linalg import solve_discrete_lyapunov


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


# ----------------------------------------------------------------------
# Classe ParamPKF
# ----------------------------------------------------------------------
class ParamPKF:
    """
    Contient et gère les paramètres d’un filtre de Kalman couplé.
    Met automatiquement à jour les matrices dérivées à chaque modification.
    """

    def __init__(self, dim_y: int, dim_x: int, A: np.ndarray, mQ: np.ndarray, verbose: int = 1):
        # --- Vérifications des dimensions ---
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

        # --- Initialisation des matrices ---
        self._A = np.array(A, dtype=float)
        if self._A.shape != (self.dim_xy, self.dim_xy):
            raise ValueError(f"A doit être carrée de dimension ({self.dim_xy},{self.dim_xy})")
        self._update_A_views()

        self._mQ = np.array(mQ, dtype=float)
        if self._mQ.shape != (self.dim_xy, self.dim_xy):
            raise ValueError(f"mQ doit être carrée de dimension ({self.dim_xy},{self.dim_xy})")
        self._update_Q_views()

        self._mu0 = np.zeros(self.dim_xy)
        self._updateSigma()
        self._check_consistency()

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
    def _updateSigma(self):
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
        self._a = self._Sigma[self.dim_xy:self.dim_xy+self.dim_x, 0:self.dim_x]
        self._b = self._Sigma[self.dim_x:self.dim_xy, 0:self.dim_x]
        self._c = self._Sigma[self.dim_xy+self.dim_x:2*self.dim_xy, self.dim_x:self.dim_xy]
        self._d = self._Sigma[self.dim_xy+self.dim_x:2*self.dim_xy, 0:self.dim_x]
        self._e = self._Sigma[self.dim_xy:self.dim_xy+self.dim_x, self.dim_x:self.dim_xy]

        # Blocs diagonaux
        self._Sigma_X1 = self._Sigma[0:self.dim_x, 0:self.dim_x]
        self._Sigma_Y1 = self._Sigma[self.dim_x:self.dim_xy, self.dim_x:self.dim_xy]
        self._Sigma_X2 = self._Sigma[self.dim_xy:self.dim_xy+self.dim_x, self.dim_xy:self.dim_xy+self.dim_x]
        self._Sigma_Y2 = self._Sigma[self.dim_xy+self.dim_x:2*self.dim_xy, self.dim_xy+self.dim_x:2*self.dim_xy]

    # ------------------------------------------------------------------
    # Vues dynamiques sur A et Q
    # ------------------------------------------------------------------
    def _update_A_views(self):
        def _callback():
            self._updateSigma()
            self._check_consistency()
            logger.debug("[ActiveView] A mis à jour → recalcul Sigma")
        self._A_xx = ActiveView(self._A, slice(0, self.dim_x), slice(0, self.dim_x), _callback)
        self._A_xy = ActiveView(self._A, slice(0, self.dim_x), slice(self.dim_x, self.dim_xy), _callback)
        self._A_yx = ActiveView(self._A, slice(self.dim_x, self.dim_xy), slice(0, self.dim_x), _callback)
        self._A_yy = ActiveView(self._A, slice(self.dim_x, self.dim_xy), slice(self.dim_x, self.dim_xy), _callback)

    def _update_Q_views(self):
        def _callback():
            self._updateSigma()
            self._check_consistency()
            logger.debug("[ActiveView] mQ mis à jour → recalcul Sigma")
        self._Q_xx = ActiveView(self._mQ, slice(0, self.dim_x), slice(0, self.dim_x), _callback)
        self._Q_xy = ActiveView(self._mQ, slice(0, self.dim_x), slice(self.dim_x, self.dim_xy), _callback)
        self._Q_yx = ActiveView(self._mQ, slice(self.dim_x, self.dim_xy), slice(0, self.dim_x), _callback)
        self._Q_yy = ActiveView(self._mQ, slice(self.dim_x, self.dim_xy), slice(self.dim_x, self.dim_xy), _callback)

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
    def Sigma_X1(self): return self._Sigma_X1

    @property
    def Sigma_Y1(self): return self._Sigma_Y1

    @property
    def Sigma_X2(self): return self._Sigma_X2

    @property
    def Sigma_Y2(self): return self._Sigma_Y2

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
        self._updateSigma()
        self._check_consistency()
        logger.info("[ParamPKF] A mis à jour")

    @property
    def mQ(self): return self._mQ
    @mQ.setter
    def mQ(self, new_Q):
        new_Q = np.array(new_Q, dtype=float)
        if new_Q.shape != (self.dim_xy, self.dim_xy):
            raise ValueError(f"mQ doit être ({self.dim_xy},{self.dim_xy})")
        self._mQ = new_Q
        self._update_Q_views()
        self._updateSigma()
        self._check_consistency()
        logger.info("[ParamPKF] mQ mis à jour")

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
        print("========================")

    # ------------------------------------------------------------------
    def __repr__(self):
        return f"<ParamPKF(dim_y={self.dim_y}, dim_x={self.dim_x}, verbose={self.verbose})>"


# ----------------------------------------------------------------------
# Exemple d'utilisation
# ----------------------------------------------------------------------
if __name__ == "__main__":
    verbose=0
    
    dim_x, dim_y = 2, 2
    A = np.array([[5, 2, 1, 0],
                  [3, 8, 0, 2],
                  [2, 2, 10, 6],
                  [1, 1, 5, 9]], float)

    mQ = np.array([[1.0, 0.5, 0.1, 0.2],
                   [0.5, 1.0, 0.1, 0.1],
                   [0.1, 0.1, 1.0, 0.5],
                   [0.2, 0.1, 0.5, 1.0]], float)

    param = ParamPKF(dim_y=dim_y, dim_x=dim_x, A=A, mQ=mQ, verbose=verbose)
    param.summary()
