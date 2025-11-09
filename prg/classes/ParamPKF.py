#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import path, sys
directory = path.Path(__file__)
sys.path.append(directory.parent.parent)
print(directory.parent.parent)

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
        self._parent   = parent_matrix
        self._rows     = rows
        self._cols     = cols
        self._callback = callback

    def _submatrix(self):
        """Retourne la sous-matrice correspondante, en gérant tous les types d'index."""
        if isinstance(self._rows, (list, np.ndarray)) and isinstance(self._cols, (list, np.ndarray)):
            return self._parent[np.ix_(self._rows, self._cols)]
        else:
            return self._parent[self._rows, self._cols]

    def __getitem__(self, key):
        return self._submatrix()[key]

    def __setitem__(self, key, value):
        sub = self._submatrix()
        sub[key] = value
        # réécrire dans le parent
        if isinstance(self._rows, (list, np.ndarray)) and isinstance(self._cols, (list, np.ndarray)):
            self._parent[np.ix_(self._rows, self._cols)] = sub
        else:
            self._parent[self._rows, self._cols] = sub
        self._callback()

    def __sub__(self, other):
        if not isinstance(other, ActiveView):
            return NotImplemented

        A = self.value
        B = other.value

        if A.shape != B.shape:
            raise ValueError(f"Shape mismatch: {A.shape} vs {B.shape}")

        diff = A - B
        return ActiveView(diff, range(diff.shape[0]), range(diff.shape[1]), lambda: None)

    def __repr__(self):
        return f"ActiveView(\n{self.value}\n)"

    @property
    def value(self):
        """Retourne la sous-matrice correspondante."""
        return self._submatrix()

    # 🧩 AJOUTS POUR COMPATIBILITÉ NUMPY
    def __array__(self, dtype=None):
        """Permet à NumPy de convertir l'objet en tableau."""
        return np.asarray(self.value, dtype=dtype)

    def copy(self):
        """Retourne une copie indépendante de la sous-matrice."""
        return self.value.copy()

    def __neg__(self):
        """Support pour l’opérateur unaire -ActiveView"""
        return ActiveView(-self.value, range(self.value.shape[0]), range(self.value.shape[1]), lambda: None)

    def __add__(self, other):
        """Support pour + entre ActiveView ou ndarray"""
        if isinstance(other, ActiveView):
            other = other.value
        return ActiveView(self.value + other, range(self.value.shape[0]), range(self.value.shape[1]), lambda: None)

    def __radd__(self, other):
        """Support pour ndarray + ActiveView"""
        return self.__add__(other)

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
        
        # Deux façons de construire un objet de cette classe
        if len(kwargs.keys()) == 4:                                # parametrization (A, mQ, z00, Pz00)
            self.constructorFrom_A_mQ(kwargs['A'], kwargs['mQ'], kwargs['z00'], kwargs['Pz00'])
        elif len(kwargs.keys()) == 7:                              # parametrization (sxx, syy, a, b, c, d, e) --> Sigma
            self.constructorFrom_Sigma(kwargs['sxx'], kwargs['syy'], kwargs['a'], kwargs['b'], kwargs['c'], kwargs['d'], kwargs['e'])
        else:
            logger.warning(f"⚠️ Le modèle n'est pas bien paramétré : {kwargs.keys()}")


    def constructorFrom_A_mQ(self, A, mQ, z00, Pz00):

        # --- Initialisation des matrices ---
        self._A = np.array(A, dtype=float)
        if self._A.shape != (self.dim_xy, self.dim_xy):
            raise ValueError(f"⚠️ A doit être carrée de dimension ({self.dim_xy},{self.dim_xy})")
        eigvals = np.linalg.eigvals(self._A)
        if np.any(np.abs(eigvals) >= 1.0):
            logger.warning(f"⚠️ Certaines valeurs propres de A ont un module >= 1 : {eigvals}")
        self._update_A_views()

        self._mQ = np.array(mQ, dtype=float)
        if self._mQ.shape != (self.dim_xy, self.dim_xy):
            raise ValueError(f"⚠️ mQ doit être carrée de dimension ({self.dim_xy},{self.dim_xy})")
        self._update_mQ_views()

        self._z00 = np.array(z00, dtype=float)
        if self._z00.shape != (self.dim_xy, 1):
            raise ValueError(f"⚠️ z00 doit être un vecteur colonne ({self.dim_xy},1)")
        
        self._Pz00 = np.array(Pz00, dtype=float)
        if self._Pz00.shape != (self.dim_xy, self.dim_xy):
            raise ValueError(f"⚠️ Pz00 doit être carrée de dimension ({self.dim_xy},{self.dim_xy})")

        self._update_Sigma_from_A_mQ()
        self._check_consistency()

    def constructorFrom_Sigma(self, sxx, syy, a, b, c, d, e):
        
        # --- Initialisation des matrices ---
        self._sxx = np.array(sxx, dtype=float)
        if self._sxx.shape != (self.dim_x, self.dim_x):
            raise ValueError(f"⚠️ sxx doit être carrée de dimension ({self.dim_x},{self.dim_x})")
        self._b = np.array(b, dtype=float)
        if self._b.shape != (self.dim_y, self.dim_x):
            raise ValueError(f"⚠️ b doit être carrée de dimension ({self.dim_y},{self.dim_x})")
        self._syy = np.array(syy, dtype=float)
        if self._syy.shape != (self.dim_y, self.dim_y):
            raise ValueError(f"⚠️ syy doit être carrée de dimension ({self.dim_y},{self.dim_y})")
        self._a = np.array(a, dtype=float)
        if self._a.shape != (self.dim_x, self.dim_x):
            raise ValueError(f"⚠️ a doit être carrée de dimension ({self.dim_x},{self.dim_x})")
        self._d = np.array(d, dtype=float)
        if self._d.shape != (self.dim_y, self.dim_x):
            raise ValueError(f"⚠️ d doit être carrée de dimension ({self.dim_y},{self.dim_x})")
        self._e = np.array(e, dtype=float)
        if self._e.shape != (self.dim_x, self.dim_y):
            raise ValueError(f"⚠️ e doit être carrée de dimension ({self.dim_x},{self.dim_y})")
        self._c = np.array(c, dtype=float)
        if self._c.shape != (self.dim_y, self.dim_y):
            raise ValueError(f"⚠️ c doit être carrée de dimension ({self.dim_y},{self.dim_y})")

        self._update_A_mQ_from_Sigma()
        self._check_consistency()

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
        
        # calcul de A, mQ, z00, Pz00
        self._A  = self._Q2 @ np.linalg.inv(self._Q1)
        self._update_A_views()
        self._mQ = self._Q1 - self._A @ self._Q2.T
        self._update_mQ_views()
        
        self._z00  = np.zeros(shape=(self.dim_xy, 1))
        self._Pz00 = self._Q1.copy()

    
    def _update_Sigma_from_A_mQ(self):
        """Met à jour Q1, Q2 et Sigma à partir de A et mQ."""
        
        self._Q1    = solve_discrete_lyapunov(self._A, self._mQ)
        self._Q2    = self._A @ self._Q1
        self._Sigma = np.block([
            [self._Q1, self._Q2.T],
            [self._Q2, self._Q1],
        ])

        # Vérifie la cohérence Q ≈ Q1 - A Q2^T
        Q_est     = self._Q1 - self._A @ self._Q2.T
        diff      = self._mQ - Q_est
        rel_error = np.linalg.norm(diff) / (np.linalg.norm(self._mQ) + 1e-12)

        if rel_error > 1e-8:
            logger.warning(f"⚠️ Incohérence : Q ≉ Q1 - A Q2^T (erreur relative = {rel_error:.2e})")
            if self.verbose >= 2:
                logger.debug(f"Différence :\n{diff}")
        else:
            logger.debug(f"♻️ Vérification OK : ||Q - (Q1 - A Q2^T)||_rel = {rel_error:.2e}")

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
            logger.info("[ActiveView] ✅ A, Sigma matrice updated")
        self._A_xx = ActiveView(self._A, slice(0, self.dim_x),           slice(0, self.dim_x),           _callback)
        self._A_xy = ActiveView(self._A, slice(0, self.dim_x),           slice(self.dim_x, self.dim_xy), _callback)
        self._A_yx = ActiveView(self._A, slice(self.dim_x, self.dim_xy), slice(0, self.dim_x),           _callback)
        self._A_yy = ActiveView(self._A, slice(self.dim_x, self.dim_xy), slice(self.dim_x, self.dim_xy), _callback)

    def _update_mQ_views(self):
        def _callback():
            self._update_Sigma_from_A_mQ()
            self._check_consistency()
            logger.debug("[ActiveView] ✅ mQ, Sigma matrices updated")
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
        if hasattr(self, "_mQ"):    _is_covariance(self._mQ,    "mQ")
        if hasattr(self, "_Q1"):    _is_covariance(self._Q1,    "Q1")
        if hasattr(self, "_Sigma"): _is_covariance(self._Sigma, "Sigma")
        if hasattr(self, "_sxx"):   _is_covariance(self._sxx,   "sxx")
        if hasattr(self, "_syy"):   _is_covariance(self._syy,   "syy")
        if hasattr(self, "_Pz00"):  _is_covariance(self._Pz00,  "Pz00")

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
    def z00(self): return self._z00
    @property
    def Pz00(self): return self._Pz00

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
            raise ValueError(f"⚠️ A doit être ({self.dim_xy},{self.dim_xy})")
        self._A = new_A
        self._update_A_views()
        self._update_Sigma_from_A_mQ()
        self._check_consistency()
        logger.info("[ParamPKF] ✅ A matrix updates")
    
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
            raise ValueError(f"⚠️ mQ doit être ({self.dim_xy},{self.dim_xy})")
        self._mQ = new_Q
        self._update_mQ_views()
        self._update_Sigma_from_A_mQ()
        self._check_consistency()
        logger.info("[ParamPKF] ✅ mQ matrix updated")
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

        print("=== ParamPKF Summary ===")
        print(f"dim_x={self.dim_x}, dim_y={self.dim_y}, verbose={self.verbose}\n")
        print("A:\n",     fmt(self.A))
        print("mQ:\n",    fmt(self.mQ))
        print("z00:\n",   fmt(self.z00))
        print("Pz00:\n",   fmt(self.Pz00))
        print("Sigma:\n", fmt(self._Sigma))
        if self.verbose>0:
            print("========================")
            print("  Q1:\n  ",  fmt(self._Q1))
            print("  Q2:\n  ",  fmt(self._Q2))
            print("========================")
            print("  sxx:\n  ", fmt(self._sxx))
            print("  syy:\n  ", fmt(self._syy))
            print("  a:\n  ",   fmt(self._a))
            print("  b:\n  ",   fmt(self._b))
            print("  c:\n  ",   fmt(self._c))
            print("  d:\n  ",   fmt(self._d))
            print("  e:\n  ",   fmt(self._e))
            print("========================")
        if self.verbose>1:  # Ready to copy in python code
            print("A  = np.array(", repr(self.A.tolist()), ')')
            print("mQ = np.array(", repr(self.mQ.tolist()), ')')
            print("z00 = np.array(", repr(self.z00.tolist()), ')')
            print("Pz00 = np.array(", repr(self.Pz00.tolist()), ')')
        # self._check_consistency()


# ----------------------------------------------------------------------
# Exemples d'utilisation
# ----------------------------------------------------------------------
if __name__ == "__main__":
    verbose = 1

    # ------------------------------------------------------------------
    # Test parameters for (Sigma = (sxx, syy, a, b, c, d, e)) parametrization
    # ------------------------------------------------------------------
    
    from models.linear.linear_x1_y1 import model_x1_y1_from_Sigma # dim_x = dim_y = 1
    dim_x, dim_y, sxx, syy, a, b, c, d, e = model_x1_y1_from_Sigma()
    
    from models.linear.linear_x2_y2 import model_x2_y2_from_Sigma # dim_x = dim_y = 2
    dim_x, dim_y, sxx, syy, a, b, c, d, e = model_x2_y2_from_Sigma()
    
    from models.linear.linear_x3_y1 import model_x3_y1_from_Sigma # dim_x = 3, dim_y = 1
    dim_x, dim_y, sxx, syy, a, b, c, d, e = model_x3_y1_from_Sigma()
    
    param = ParamPKF(dim_x, dim_y, verbose, sxx=sxx, syy=syy, a=a, b=b, c=c, d=d, e=e)
    if verbose > 0:
        param.summary()

    # ------------------------------------------------------------------
    # Test parameters for (A, mQ) parametrization
    # ------------------------------------------------------------------
    
    from models.linear.linear_x1_y1 import model_x1_y1_from_A_mQ # dim_x = dim_y = 1
    dim_x, dim_y, A, mQ, z00, Pz00 = model_x1_y1_from_A_mQ()
    
    from models.linear.linear_x2_y2 import model_x2_y2_from_A_mQ # dim_x = dim_y = 2
    dim_x, dim_y, A, mQ, z00, Pz00 = model_x2_y2_from_A_mQ()
    
    from models.linear.linear_x3_y1 import model_x3_y1_from_A_mQ # dim_x = 3, dim_y = 1
    dim_x, dim_y, A, mQ, z00, Pz00 = model_x3_y1_from_A_mQ()

    param = ParamPKF(dim_x, dim_y, verbose, A=A, mQ=mQ, z00=z00, Pz00=Pz00)
    if verbose > 0:
        param.summary()