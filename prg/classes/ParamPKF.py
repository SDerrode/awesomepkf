#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import path, sys
directory = path.Path(__file__)
sys.path.append(directory.parent.parent)

import logging
from typing import Callable, Any, Union, Optional
import warnings

import numpy as np
from scipy.linalg import solve_discrete_lyapunov

from models.linear import BaseModel, all_models
from classes.ActiveView import ActiveView

# ----------------------------------------------------------------------
# Configuration du logging global
# ----------------------------------------------------------------------
logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Classe ParamPKF
# ----------------------------------------------------------------------
class ParamPKF:
    """
    Contient et gère les paramètres d’un filtre de Kalman couple.
    Met automatiquement à jour les matrices dérivées à chaque modification.
    """

    def __init__(self, verbose: int, dim_x: int, dim_y: int, **kwargs):
        if not isinstance(dim_y, int) or dim_y <= 0:
            raise ValueError("⚠️ dim_y doit être un entier > 0")
        if not isinstance(dim_x, int) or dim_x <= 0:
            raise ValueError("⚠️ dim_x doit être un entier > 0")
        if verbose not in [0, 1, 2]:
            raise ValueError("⚠️ verbose doit être 0, 1 ou 2")

        self.dim_y = dim_y
        self.dim_x = dim_x
        self.dim_xy = dim_x + dim_y
        self.verbose = verbose

        # Configuration du logger selon verbose
        self._set_log_level()

        # Deux façons de construire un objet de cette classe
        if len(kwargs.keys()) == 4:  # parametrization (A, mQ, z00, Pz00)
            self.constructorFrom_A_mQ(kwargs['A'], kwargs['mQ'], kwargs['z00'], kwargs['Pz00'])
        elif len(kwargs.keys()) == 7:  # parametrization (sxx, syy, a, b, c, d, e) --> Sigma
            self.constructorFrom_Sigma(kwargs['sxx'], kwargs['syy'], kwargs['a'], kwargs['b'], kwargs['c'], kwargs['d'], kwargs['e'])
        else:
            logger.warning(f"⚠️ Le modèle n'est pas bien paramétré : {kwargs.keys()}")

        # Vérification des dimensions dès la création
        if __debug__:
            self._check_dimensions()

    # ------------------------------------------------------------------
    # Constructeurs
    # ------------------------------------------------------------------
    def constructorFrom_A_mQ(self, A: np.ndarray, mQ: np.ndarray, z00: np.ndarray, Pz00: np.ndarray) -> None:
        self._A: np.ndarray = np.array(A, dtype=float)
        if self._A.shape != (self.dim_xy, self.dim_xy):
            raise ValueError(f"⚠️ A doit être carrée de dimension ({self.dim_xy},{self.dim_xy})")
        if __debug__:
            eigvals = np.linalg.eigvals(self._A)
            if np.any(np.abs(eigvals) >= 1.0):
                logger.warning(f"⚠️ Certaines valeurs propres de A ont un module >= 1 : {eigvals}")
        self._update_A_views()

        self._mQ: np.ndarray = np.array(mQ, dtype=float)
        if self._mQ.shape != (self.dim_xy, self.dim_xy):
            raise ValueError(f"⚠️ mQ doit être carrée de dimension ({self.dim_xy},{self.dim_xy})")
        self._update_mQ_views()

        self._z00: np.ndarray = np.array(z00, dtype=float)
        if self._z00.shape != (self.dim_xy, 1):
            raise ValueError(f"⚠️ z00 doit être un vecteur colonne ({self.dim_xy},1)")

        self._Pz00: np.ndarray = np.array(Pz00, dtype=float)
        if self._Pz00.shape != (self.dim_xy, self.dim_xy):
            raise ValueError(f"⚠️ Pz00 doit être carrée de dimension ({self.dim_xy},{self.dim_xy})")

        self._update_Sigma_from_A_mQ()
        self._check_consistency()

    def constructorFrom_Sigma(self, sxx: np.ndarray, syy: np.ndarray, a: np.ndarray, b: np.ndarray,
                              c: np.ndarray, d: np.ndarray, e: np.ndarray) -> None:
        self._sxx: np.ndarray = np.array(sxx, dtype=float)
        self._syy: np.ndarray = np.array(syy, dtype=float)
        self._a: np.ndarray = np.array(a, dtype=float)
        self._b: np.ndarray = np.array(b, dtype=float)
        self._c: np.ndarray = np.array(c, dtype=float)
        self._d: np.ndarray = np.array(d, dtype=float)
        self._e: np.ndarray = np.array(e, dtype=float)

        self._update_A_mQ_from_Sigma()
        self._check_consistency()

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
    # Vérification des dimensions
    # ------------------------------------------------------------------
    def _check_dimensions(self) -> None:
        """Vérifie que toutes les matrices ont les dimensions attendues."""
        expected_shapes = {
            'A': (self.dim_xy, self.dim_xy),
            'mQ': (self.dim_xy, self.dim_xy),
            'z00': (self.dim_xy, 1),
            'Pz00': (self.dim_xy, self.dim_xy),
            'Q1': (self.dim_xy, self.dim_xy),
            'Q2': (self.dim_xy, self.dim_xy),
            'Sigma': (2*self.dim_xy, 2*self.dim_xy),
            'sxx': (self.dim_x, self.dim_x),
            'syy': (self.dim_y, self.dim_y),
            'a': (self.dim_x, self.dim_x),
            'b': (self.dim_y, self.dim_x),
            'c': (self.dim_y, self.dim_y),
            'd': (self.dim_y, self.dim_x),
            'e': (self.dim_x, self.dim_y),
        }
        for attr, shape in expected_shapes.items():
            if hasattr(self, f"_{attr}"):
                actual = getattr(self, f"_{attr}")
                if actual.shape != shape:
                    raise ValueError(f"⚠️ Matrice {attr} a une forme {actual.shape}, attendue {shape}")


    # ------------------------------------------------------------------
    # Mise à jour des matrices dérivées
    # ------------------------------------------------------------------
    def _update_A_mQ_from_Sigma(self) -> None:
        self._Q1 = np.block([[self._sxx, self._b.T], [self._b, self._syy]])
        self._Q2 = np.block([[self._a, self._e], [self._d, self._c]])
        self._Sigma = np.block([[self._Q1, self._Q2.T], [self._Q2, self._Q1]])

        self._A = self._Q2 @ np.linalg.inv(self._Q1)
        self._update_A_views()
        self._mQ = self._Q1 - self._A @ self._Q2.T
        self._update_mQ_views()

        self._z00 = np.zeros((self.dim_xy, 1))
        self._Pz00 = self._Q1.copy()

        if __debug__:
            self._check_dimensions()

    def _update_Sigma_from_A_mQ(self) -> None:
        self._Q1 = solve_discrete_lyapunov(self._A, self._mQ)
        self._Q2 = self._A @ self._Q1
        self._Sigma = np.block([[self._Q1, self._Q2.T], [self._Q2, self._Q1]])

        # Vérification cohérence
        Q_est = self._Q1 - self._A @ self._Q2.T
        diff = self._mQ - Q_est
        rel_error = np.linalg.norm(diff) / (np.linalg.norm(self._mQ) + 1e-12)
        if rel_error > 1e-8 and __debug__:
            logger.warning(f"⚠️ Incohérence : Q ≉ Q1 - A Q2^T (erreur relative = {rel_error:.2e})")
            if self.verbose >= 2:
                logger.debug(f"Différence :\n{diff}")
        elif __debug__:
            logger.debug(f"♻️ Vérification OK : ||Q - (Q1 - A Q2^T)||_rel = {rel_error:.2e}")

        # Sous-blocs
        self._a = self._Sigma[self.dim_xy:self.dim_xy+self.dim_x, 0:self.dim_x]
        self._b = self._Sigma[self.dim_x:self.dim_xy, 0:self.dim_x]
        self._c = self._Sigma[self.dim_xy+self.dim_x:2*self.dim_xy, self.dim_x:self.dim_xy]
        self._d = self._Sigma[self.dim_xy+self.dim_x:2*self.dim_xy, 0:self.dim_x]
        self._e = self._Sigma[self.dim_xy:self.dim_xy+self.dim_x, self.dim_x:self.dim_xy]
        self._sxx = self._Sigma[0:self.dim_x, 0:self.dim_x]
        self._syy = self._Sigma[self.dim_x:self.dim_xy, self.dim_x:self.dim_xy]

        if __debug__:
            self._check_dimensions()

    # ------------------------------------------------------------------
    # Vues dynamiques
    # ------------------------------------------------------------------
    def _update_A_views(self) -> None:
        def _callback() -> None:
            if __debug__:
                self._update_Sigma_from_A_mQ()
                self._check_consistency()
                logger.info("[ActiveView] ✅ A, Sigma matrice updated")

        self._A_xx = ActiveView(self._A, slice(0, self.dim_x), slice(0, self.dim_x), _callback)
        self._A_xy = ActiveView(self._A, slice(0, self.dim_x), slice(self.dim_x, self.dim_xy), _callback)
        self._A_yx = ActiveView(self._A, slice(self.dim_x, self.dim_xy), slice(0, self.dim_x), _callback)
        self._A_yy = ActiveView(self._A, slice(self.dim_x, self.dim_xy), slice(self.dim_x, self.dim_xy), _callback)

    def _update_mQ_views(self) -> None:
        def _callback() -> None:
            if __debug__:
                self._update_Sigma_from_A_mQ()
                self._check_consistency()
                logger.debug("[ActiveView] ✅ mQ, Sigma matrices updated")

        self._mQ_xx = ActiveView(self._mQ, slice(0, self.dim_x), slice(0, self.dim_x), _callback)
        self._mQ_xy = ActiveView(self._mQ, slice(0, self.dim_x), slice(self.dim_x, self.dim_xy), _callback)
        self._mQ_yx = ActiveView(self._mQ, slice(self.dim_x, self.dim_xy), slice(0, self.dim_x), _callback)
        self._mQ_yy = ActiveView(self._mQ, slice(self.dim_x, self.dim_xy), slice(self.dim_x, self.dim_xy), _callback)

    # ------------------------------------------------------------------
    # Vérification de cohérence
    # ------------------------------------------------------------------
    def _check_consistency(self) -> None:
        def _is_covariance(M: np.ndarray, name: str) -> None:
            if not np.allclose(M, M.T, atol=1e-12):
                logger.warning(f"⚠️ {name} matrix is not symmetrical")
            eigvals = np.linalg.eigvals(M)
            if np.any(eigvals < -1e-12):
                logger.warning(f"⚠️ {name} matrix is not positive semi-definite (min eig = {eigvals.min():.3e})")
            logger.debug(f"Eig of {name} matrix: {eigvals}")

        for attr, name in [('_mQ', 'mQ'), ('_Q1', 'Q1'), ('_Sigma', 'Sigma'),
                           ('_sxx', 'sxx'), ('_syy', 'syy'), ('_Pz00', 'Pz00')]:
            if hasattr(self, attr):
                _is_covariance(getattr(self, attr), name)

    # ------------------------------------------------------------------
    # Getters / Setters
    # ------------------------------------------------------------------
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
            logger.info("[ParamPKF] ✅ A matrix updates")

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
            logger.info("[ParamPKF] ✅ mQ matrix updated")

    @property
    def mQ_xx(self) -> ActiveView: return self._mQ_xx
    @property
    def mQ_xy(self) -> ActiveView: return self._mQ_xy
    @property
    def mQ_yx(self) -> ActiveView: return self._mQ_yx
    @property
    def mQ_yy(self) -> ActiveView: return self._mQ_yy

    # ------------------------------------------------------------------
    # Résumé
    # ------------------------------------------------------------------
    def summary(self) -> None:
        def fmt(M: Any) -> str:
            if hasattr(M, "_parent"):
                M = M._parent[M._rows, M._cols]
            return np.array2string(M, formatter={'float_kind': lambda x: f"{x:6.2f}"})

        print("=== ParamPKF Summary ===")
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
        # self._check_consistency()


# ----------------------------------------------------------------------
# Exemples d'utilisation
# ----------------------------------------------------------------------
if __name__ == "__main__":
    verbose = 1
    
    # Lister tous les fichiers de modèles détectés
    # print("Modèles détectés :", list(all_models.keys()))

    # Importer un modèle spécifique et accéder à ses fonctions/classes
    # Available : ['A_mQ_x1_y1', 'A_mQ_x3_y1', 'Sigma_x1_y1', 'Sigma_x3_y1', 'A_mQ_x2_y2', 'Sigma_x2_y2', 'A_mQ_x1_y1_VPgreaterThan1']
    model_module = all_models['Sigma_x2_y2']
    model = model_module.create_model()
    print(f'model={model.info}')
    print(f'model={model.get_params()}')
    
    params = model.get_params().copy()
    param = ParamPKF(verbose, params.pop('dim_x'), params.pop('dim_y'), **params)
    if verbose > 0:
        param.summary()
