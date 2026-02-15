#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Standard library
from typing import Generator, Optional

from dataclasses import dataclass

# Third-party
import numpy as np
from scipy.linalg import cho_factor, cho_solve

# Local imports
from .HistoryTracker import HistoryTracker
from .SeedGenerator import SeedGenerator
from others.utils import diagnose_covariance, rich_show_fields


"""
Module PKF ########################################################
####################################################################
Classe mère de toutes les classes portant sur filtre de Kalman couple (PKF),
quelle soit linéaire ou non linéaire
####################################################################
"""

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
            if not np.allclose(arr, arr.T, atol=1e-12):
                raise ValueError(f"{name} must be symmetric")
            # Semi-définie positive (toutes les valeurs propres >= 0)
            eigvals = np.linalg.eigvalsh(arr)
            if np.any(eigvals < -1e-12):
                raise ValueError(
                    f"{name} must be positive semi-definite, found negative eigenvalues: {eigvals[eigvals<0]}"
                )
        
        # ------------------------
        # Vérification Kkp1 (dx × dy)
        # ------------------------
        arr = getattr(self, "Kkp1")
        if arr.ndim != 2:
            raise ValueError(f"Kkp1 must be a 2D matrix, got shape {arr.shape}")


class PKF:
    
    def __init__(self, sKey: Optional[int] = None, verbose: int = 0):

        if not ((isinstance(sKey, int) and sKey > 0) or sKey is None):
            raise ValueError("sKey must be None or a number>0")
        if verbose not in [0, 1, 2]:
            raise ValueError("verbose must be 0, 1 or 2")

        self.dt        = 1
        self.verbose   = verbose
        
        # Genrateur de nombre aléatoires
        self._seed_gen = SeedGenerator(sKey)

        # Shortcuts
        self.dim_x, self.dim_y, self.dim_xy = self.param.dim_x, self.param.dim_y, self.param.dim_xy
        self.z00, self.Pz00, self.g, self.mQ, self.augmented = self.param._z00, self.param._Pz00, self.param.g, self.param.mQ, self.param.augmented

        # for speeding process
        self.eye_dim_y       = np.eye(self.dim_y)
        self.eye_dim_x       = np.eye(self.dim_x)
        self.zeros_dim_x_y   = np.zeros(shape=(self.dim_x, self.dim_y))
        self.zeros_dim_y_1   = np.zeros(shape=(self.dim_y, 1))
        self.zeros_dim_xy_1  = np.zeros(shape=(self.dim_xy, 1))
        self.zeros_dim_xy_xy = np.zeros(shape=(self.dim_xy, self.dim_xy))

        # Create HistoryTracker
        self._history = HistoryTracker(self.verbose)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def seed_gen(self) -> int:
        """Return generator seed."""
        return self._seed_gen.seed

    @property
    def history(self) -> Optional[HistoryTracker]:
        return self._history

    def simulate_N_data(self, N):
        return list(self._data_generation(N))
    
    def process_N_data(self, N: Optional[int], data_generator: Optional[Generator] = None) -> list:
        return list(self.process_filter(N=N, data_generator=data_generator))

    # ------------------------------------------------------------------
    # Generators
    # ------------------------------------------------------------------
    def _data_generation(self, N: Optional[int] = None) -> Generator[tuple[int, np.ndarray, np.ndarray], None, None]:
        """
        Generator for the simulation of Z_{k+1} = A * Z_k + W_{k+1},
        with W_{k+1} ~ N(0, mQ) and Z_1 ~ N(0, Q1).
        """

        Zkp1_simul = np.zeros(shape=(self.dim_xy, 1))

        # The first
        if self.augmented:
            Zkp1_simul[0:self.dim_x, 0] = self._seed_gen.rng.multivariate_normal(mean=self.z00[0:self.dim_x, 0], cov=self.Pz00[0:self.dim_x, 0:self.dim_x])
            Zkp1_simul[self.dim_x:,  0] = Zkp1_simul[self.dim_x-self.dim_y:self.dim_x, 0]
        else:
            Zkp1_simul[:,0] = self._seed_gen.rng.multivariate_normal(mean=self.z00[:, 0], cov=self.Pz00)
        Xkp1_simul, Ykp1_simul = np.split(Zkp1_simul, [self.dim_x])

        k = 0
        yield k, Xkp1_simul, Ykp1_simul

        # The next ones...
        zerosvector_xy = np.zeros(shape=(self.dim_xy))
        zerosvector_x  = np.zeros(shape=(self.dim_x))
        noise_z        = np.zeros(shape=(self.dim_xy, 1))
        while N is None or k<N:
            
            if self.augmented:
                noise_z[0:self.dim_x, 0] = self._seed_gen.rng.multivariate_normal(mean=zerosvector_x, cov=self.mQ[0:self.dim_x, 0:self.dim_x])
                noise_z[self.dim_x:,  0] = noise_z[self.dim_x-self.dim_y:self.dim_x, 0]
            else:
                noise_z[:, 0] = self._seed_gen.rng.multivariate_normal(mean=zerosvector_xy, cov=self.mQ)

            Zkp1_simul = self.g(Zkp1_simul, noise_z, self.dt)
            Xkp1_simul, Ykp1_simul = np.split(Zkp1_simul, [self.dim_x])

            k += 1
            yield k, Xkp1_simul, Ykp1_simul

    # ------------------------------------------------------------------
    # Factorizations
    # ------------------------------------------------------------------
    
    def _test_CovMatrix(self, Mat, k):
        if not self.augmented:
            verdict, report = diagnose_covariance(Mat)
            if not verdict:
                print(f'Matrix={Mat}\nReport - iteration k={k}:')
                rich_show_fields(report, ["is_symmetric", "cholesky_ok", "is_psd", "near_singular", "ill_conditioned", "numerically_singular"], title="")
                input('attente')
    
    def _firstEstimate(self, generator):
        
        # First sample
        k, xkp1, ykp1 = next(generator)
        
        # temp            = self.Pz00[0:self.dim_x, self.dim_x:] @ inv(self.Pz00[self.dim_x:, self.dim_x:])
        # Xkp1_update     = temp @ ykp1
        # PXXkp1_update   = self.Pz00[0:self.dim_x, 0:self.dim_x] - temp @ self.Pz00[self.dim_x:, 0:self.dim_x]
        Xkp1_update     = xkp1.copy() #z00[0:self.dim_x]
        PXXkp1_update   = self.Pz00[0:self.dim_x, 0:self.dim_x].copy()
        self._test_CovMatrix(PXXkp1_update, k)
        
        Xkp1_predict = np.zeros(shape=(self.dim_x, 1))
        aStep = PKFStep(
            k              = k,
            xkp1           = xkp1.copy() if xkp1 is not None else None,
            ykp1           = ykp1.copy(),
            Xkp1_predict   = Xkp1_predict.copy(),
            PXXkp1_predict = self.eye_dim_x.copy(),
            ikp1           = self.zeros_dim_y_1.copy(),
            Skp1           = self.eye_dim_y.copy(),
            Kkp1           = self.zeros_dim_x_y.copy(),
            Xkp1_update    = Xkp1_update.copy(),
            PXXkp1_update  = PXXkp1_update.copy(),
        )

        # Record data in the tracker
        self._history.record(aStep)
        # Affichage
        if self.verbose>1:
            rich_show_fields(aStep, title="toto")

        return aStep


    def _nextUpdating(self, k, xkp1, ykp1, Zkp1_predict, Pkp1_predict):
        
        # Cutting Zkp1_predict into 2 blocks
        Xkp1_predict, Ykp1_predict = np.split(Zkp1_predict, [self.dim_x])
        
        # Cutting Pkp1 into 4 blocks
        PXXkp1_predict = Pkp1_predict[:self.dim_x, :self.dim_x]
        PXYkp1_predict = Pkp1_predict[:self.dim_x, self.dim_x:]
        PYXkp1_predict = Pkp1_predict[self.dim_x:, :self.dim_x]
        PYYkp1_predict = Pkp1_predict[self.dim_x:, self.dim_x:]

        # Updating
        ###############################################
        ikp1 = ykp1 - Ykp1_predict
        Skp1 = PYYkp1_predict.copy()

        condS = np.linalg.cond(Skp1)
        if condS > 1e12:
            if self.verbose >= 2:
                logger.warning(f"Skp1 ill-conditioned (cond={condS:.2e})")
            Skp1 += 1e-10 * self.eye_dim_y
        try:
            c, low = cho_factor(Skp1)
            Kkp1   = PXYkp1_predict @ cho_solve((c, low), self.eye_dim_y)
        except LinAlgError as e:
            print(f'Skp1={Skp1}')
            input('ATTENTE')
        except ValueError as e:
            print("Erreur de valeur :", e)
            input('ATTENTE')
        # print(f'Kkp1={Kkp1}')
        
        Xkp1_update   = Xkp1_predict   + Kkp1 @ ikp1
        PXXkp1_update = PXXkp1_predict - Kkp1 @ PYXkp1_predict
        self._test_CovMatrix(PXXkp1_update, k)
        
        # Forme de Joseph
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
            PXXkp1_update  = PXXkp1_update_Joseph.copy(), #PXXkp1_update.copy(),
        )
        
        # Record data in the tracker
        self._history.record(aStep)
        # Affichage
        if self.verbose>1:
            rich_show_fields(aStep, title="toto")

        return aStep
