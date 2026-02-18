#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Standard library
from typing import Generator, Optional
from dataclasses import dataclass
import logging

# Third-party
import numpy as np
from scipy.linalg import cho_factor, cho_solve, LinAlgError

# Local imports
from .HistoryTracker import HistoryTracker
from .SeedGenerator import SeedGenerator
from others.utils import diagnose_covariance, rich_show_fields
from others.numerics import EPS_ABS, COND_FAIL

@dataclass(slots=True, frozen=True)
class PKFStep:
    """Container for one PKF iteration step."""
    k:              int
    xkp1:           Optional[np.ndarray]
    ykp1:           np.ndarray
    Xkp1_predict:   np.ndarray
    PXXkp1_predict: np.ndarray
    ikp1:           np.ndarray
    Skp1:           np.ndarray
    Kkp1:           np.ndarray
    Xkp1_update:    np.ndarray
    PXXkp1_update:  np.ndarray
    
    def __post_init__(self):
        # ------------------------
        # Vérification vecteurs colonnes
        # ------------------------
        for name in ["xkp1", "ykp1", "Xkp1_predict", "ikp1", "Xkp1_update"]:
            arr = getattr(self, name)
            if arr.ndim != 2 or arr.shape[1] != 1:
                raise ValueError(
                    f"{name} must be a column vector of shape (n,1), got {arr.shape}"
                )
        
        # ------------------------
        # Vérification matrices carrées et covariances
        # ------------------------
        for name in ["PXXkp1_predict", "Skp1", "PXXkp1_update"]:
            arr = getattr(self, name)
            if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
                raise ValueError(
                    f"{name} must be a square matrix, got shape {arr.shape}"
                )
            # Symétrie
            if not np.allclose(arr, arr.T, atol=EPS_ABS):
                raise ValueError(f"{name} must be symmetric")
            # Semi-définie positive
            eigvals = np.linalg.eigvalsh(arr)
            if np.any(eigvals < -EPS_ABS):
                raise ValueError(
                    f"{name} must be positive semi-definite, found negative eigenvalues: {eigvals[eigvals<0]}"
                )
        
        # ------------------------
        # Vérification Kkp1 (dx × dy)
        # ------------------------
        arr = getattr(self, "Kkp1")
        if arr.ndim != 2:
            raise ValueError(f"Kkp1 must be a 2D matrix, got shape {arr.shape}")

import inspect

class PKF:
    
    def __init__(self, sKey: Optional[int] = None, verbose: int = 0):

        if not ((isinstance(sKey, int) and sKey > 0) or sKey is None):
            raise ValueError("sKey must be None or a number>0")
        if verbose not in [0, 1, 2]:
            raise ValueError("verbose must be 0, 1 or 2")
        
        # stack = inspect.stack()
        # print("Pile d'appels :")
        # for frame in stack[:5]:  # limiter pour lisibilité
        #     print(f"Function: {frame.function}, File: {frame.filename}, Line: {frame.lineno}")
        # input('ATTENTE STACK APPEL')

        self.dt        = 1
        self.verbose   = verbose
        
        # Générateur de nombres aléatoires
        self._seed_gen = SeedGenerator(sKey)
        
        # Do we have a ground truth? Defult Yes
        self.ground_truth = True
        
        # Shortcuts
        self.dim_x, self.dim_y, self.dim_xy = self.param.dim_x, self.param.dim_y, self.param.dim_xy
        self.z00, self.Pz00, self.g, self.mQ, self.augmented = self.param._z00, self.param._Pz00, self.param.g, self.param.mQ, self.param.augmented

        # Matrices utiles pour accélérer les calculs
        self.eye_dim_y       = np.eye(self.dim_y)
        self.eye_dim_x       = np.eye(self.dim_x)
        self.zeros_dim_x_y   = np.zeros(shape=(self.dim_x, self.dim_y))
        self.zeros_dim_y_1   = np.zeros(shape=(self.dim_y, 1))
        self.zeros_dim_xy_1  = np.zeros(shape=(self.dim_xy, 1))
        self.zeros_dim_xy_xy = np.zeros(shape=(self.dim_xy, self.dim_xy))

        # History tracker
        self.history = HistoryTracker(self.verbose)

        # ----------------------------
        # Logger
        # ----------------------------
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)  # Always debug; handler filters by verbose

        if not self.logger.handlers:
            ch = logging.StreamHandler()
            if verbose == 0:
                ch.setLevel(logging.CRITICAL + 1)  # Rien n'est affiché
            elif verbose == 1:
                ch.setLevel(logging.WARNING)       # Warnings et erreurs seulement
            else:  # verbose == 2
                ch.setLevel(logging.DEBUG)         # Tout est affiché
            
            formatter = logging.Formatter(
                fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def seed_gen(self) -> int:
        """Return generator seed."""
        return self._seed_gen.seed

    # ------------------------------------------------------------------
    # Data simulation & processing
    # ------------------------------------------------------------------
    def simulate_N_data(self, N):
        return list(self._data_generation(N))
    
    def process_N_data(self, N: Optional[int], data_generator: Optional[Generator] = None) -> list:
        return list(self.process_filter(N=N, data_generator=data_generator))

    # ------------------------------------------------------------------
    # Generators
    # ------------------------------------------------------------------
    def _data_generation(self, N: Optional[int] = None) -> Generator[tuple[int, np.ndarray, np.ndarray], None, None]:
        Zkp1_simul = np.zeros(shape=(self.dim_xy, 1))

        # First step
        if self.augmented:
            Zkp1_simul[0:self.dim_x, 0] = self._seed_gen.rng.multivariate_normal(
                mean=self.z00[0:self.dim_x, 0], cov=self.Pz00[0:self.dim_x, 0:self.dim_x]
            )
            Zkp1_simul[self.dim_x:, 0] = Zkp1_simul[self.dim_x-self.dim_y:self.dim_x, 0]
        else:
            Zkp1_simul[:,0] = self._seed_gen.rng.multivariate_normal(mean=self.z00[:, 0], cov=self.Pz00)

        Xkp1_simul, Ykp1_simul = np.split(Zkp1_simul, [self.dim_x])
        k = 0
        yield k, Xkp1_simul, Ykp1_simul

        # Next steps
        zerosvector_xy = np.zeros(self.dim_xy)
        zerosvector_x  = np.zeros(self.dim_x)
        noise_z        = np.zeros(shape=(self.dim_xy, 1))
        while N is None or k<N:
            if self.augmented:
                noise_z[0:self.dim_x, 0] = self._seed_gen.rng.multivariate_normal(mean=zerosvector_x, cov=self.mQ[0:self.dim_x, 0:self.dim_x])
                noise_z[self.dim_x:, 0] = noise_z[self.dim_x-self.dim_y:self.dim_x, 0]
            else:
                noise_z[:, 0] = self._seed_gen.rng.multivariate_normal(mean=zerosvector_xy, cov=self.mQ)

            Zkp1_simul = self.g(Zkp1_simul, noise_z, self.dt)
            Xkp1_simul, Ykp1_simul = np.split(Zkp1_simul, [self.dim_x])
            k += 1
            yield k, Xkp1_simul, Ykp1_simul

    # ------------------------------------------------------------------
    # Covariance diagnostics
    # ------------------------------------------------------------------
    def _test_CovMatrix(self, Mat, k):
        """
        Vérifie si une matrice est une covariance valide :
        - symétrique
        - semi-définie positive

        Log les problèmes rencontrés et affiche le rapport via rich_show_fields si verbose > 1.
        """
        if self.augmented:
            # Pas de check sur models augmentés
            return

        verdict, report = diagnose_covariance(Mat)

        if not verdict:
            # il y a un problème
            # Récupération des valeurs propres négatives
            eigvals = np.linalg.eigvalsh(Mat)
            neg_eigvals = eigvals[eigvals < -EPS_ABS]

            self.logger.warning(
                f"Step {k}: Covariance matrix invalid. "
                f"Symmetric: {report['is_symmetric']}, Cholesky OK: {report['cholesky_ok']}, PSD: {report['is_psd']}, "
                f"Near singular: {report['near_singular']}, Ill conditioned: {report['ill_conditioned']}, "
                f"Numerically singular: {report['numerically_singular']}, "
                f"Negative eigenvalues: {neg_eigvals}"
            )

            if self.verbose > 1:
                rich_show_fields(report, ["is_symmetric", "cholesky_ok", "is_psd", 
                                        "near_singular", "ill_conditioned", "numerically_singular"], 
                                title=f"Covariance diagnostic - Step {k}")
    
    # ------------------------------------------------------------------
    # First estimate
    # ------------------------------------------------------------------
    def _firstEstimate(self, generator):
        k, xkp1, ykp1 = next(generator)
        Xkp1_update = xkp1.copy()
        PXXkp1_update = self.Pz00[0:self.dim_x, 0:self.dim_x].copy()
        self._test_CovMatrix(PXXkp1_update, k)

        Xkp1_predict = np.zeros((self.dim_x, 1))
        aStep = PKFStep(
            k=k,
            xkp1=xkp1.copy() if xkp1 is not None else None,
            ykp1=ykp1.copy(),
            Xkp1_predict=Xkp1_predict.copy(),
            PXXkp1_predict=self.eye_dim_x.copy(),
            ikp1=self.zeros_dim_y_1.copy(),
            Skp1=self.eye_dim_y.copy(),
            Kkp1=self.zeros_dim_x_y.copy(),
            Xkp1_update=Xkp1_update.copy(),
            PXXkp1_update=PXXkp1_update.copy(),
        )

        self.history.record(aStep)
        if self.verbose>1:
            rich_show_fields(aStep, title="First Estimate")
        return aStep

    # ------------------------------------------------------------------
    # Next update
    # ------------------------------------------------------------------
    def _nextUpdating(self, k, xkp1, ykp1, Zkp1_predict, Pkp1_predict, store=True):
        
        Xkp1_predict, Ykp1_predict = np.split(Zkp1_predict, [self.dim_x])
        PXXkp1_predict = Pkp1_predict[:self.dim_x, :self.dim_x]
        PXYkp1_predict = Pkp1_predict[:self.dim_x, self.dim_x:]
        PYXkp1_predict = Pkp1_predict[self.dim_x:, :self.dim_x]
        PYYkp1_predict = Pkp1_predict[self.dim_x:, self.dim_x:]

        ikp1 = ykp1 - Ykp1_predict
        Skp1 = PYYkp1_predict.copy()

        condS = np.linalg.cond(Skp1)
        if condS > COND_FAIL:
            self.logger.warning(f"Step {k}: Skp1 ill-conditioned (cond={condS:.2e})")
            Skp1 += EPS_ABS * self.eye_dim_y

        try:
            c, low = cho_factor(Skp1)
            Kkp1 = PXYkp1_predict @ cho_solve((c, low), self.eye_dim_y)
        except LinAlgError as e:
            self.logger.error(f"Step {k}: LinAlgError in cho_factor/solve: {e}")
            raise
        except ValueError as e:
            self.logger.error(f"Step {k}: ValueError in cho_factor/solve: {e}")
            raise

        Xkp1_update   = Xkp1_predict   + Kkp1 @ ikp1
        PXXkp1_update = PXXkp1_predict - Kkp1 @ PYXkp1_predict
        self._test_CovMatrix(PXXkp1_update, k)

        # Joseph form
        temp = np.vstack((self.eye_dim_x, -Kkp1.T))
        PXXkp1_update_Joseph = temp.T @ Pkp1_predict @ temp
        self._test_CovMatrix(PXXkp1_update_Joseph, k)

        aStep = PKFStep(
            k              = k,
            xkp1           = xkp1.copy() if xkp1 is not None else None,
            ykp1           = ykp1.copy(),
            Xkp1_predict   = Xkp1_predict.copy(),
            PXXkp1_predict = PXXkp1_predict.copy(),
            ikp1           = ikp1.copy(),
            Skp1           = Skp1.copy(),
            Kkp1           = Kkp1.copy(),
            Xkp1_update    = Xkp1_update.copy(),
            # PXXkp1_update  = PXXkp1_update.copy(),
            PXXkp1_update  = PXXkp1_update_Joseph.copy(),
        )

        if store:
            self.history.record(aStep)

        if self.verbose>1:
            rich_show_fields(aStep, title=f"Step {k} Update")

        return aStep
