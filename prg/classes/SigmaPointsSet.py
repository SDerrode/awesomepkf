#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from typing import Dict, Type
import warnings
import numpy as np
from itertools import product

from others.numerics import EPS_ABS

class SigmaPointsSet(ABC):
    """
    Base class for all sigma-point sets.
    """

    registry: Dict[str, Type["SigmaPointsSet"]] = {}

    def __init_subclass__(cls, *, key: str, **kwargs):
        super().__init_subclass__(**kwargs)

        if key in SigmaPointsSet.registry:
            raise RuntimeError(f"Key already registered: {key}")

        SigmaPointsSet.registry[key] = cls

    def __init__(self, dim: int):
        self.dim = dim

    @abstractmethod
    def _sigma_point(self):
        pass
    
    def _chol(self, P):
        try:
            return np.linalg.cholesky(P)
        except np.linalg.LinAlgError:
            try:
                return np.linalg.cholesky(P + EPS_ABS*np.eye(self.dim))
            except np.linalg.LinAlgError:
                raise("Méthode _chol : décomposition Choleski impossible")


class SetWAN2000(SigmaPointsSet, key="wan2000"):
    """
    The most commonly used set, defined in the paper:
    E. A. Wan and R. V. D. Merwe, 
    “The unscented Kalman filter for nonlinear estimation,” 
    in Proc. IEEE Adaptive Syst. Signal Process. Commun. Control Symp. (ASSPCCS’00), 2000, pp. 153–158.
    """
    
    def __init__(self, dim, param):
        super().__init__(dim)
        
        self.nbSigmaPoint = 2*self.dim+1
        
        self.Wm     = np.full(self.nbSigmaPoint, 1. / (2. * (self.dim + param.lambda_)))
        self.Wm[0]  = param.lambda_ / (self.dim + param.lambda_)
        if not np.isclose(self.Wm.sum(), 1.0, atol=EPS_ABS):
            raise ValueError(f"Wm weights do not sum to 1 (sum={self.Wm.sum()})")
        self.Wm /= self.Wm.sum() # normalisation au cas ou il reste un résidu
        self.Wc     = np.copy(self.Wm)
        self.Wc[0] += 1. - param.alpha**2 + param.beta  # corrective term

        self.gamma  = np.sqrt(self.dim + param.lambda_)
        

    def _sigma_point(self, x, P):
        
        x = np.atleast_2d(x).reshape(-1,1)  # (dim,1)
        
        # Compute Cholesky factor
        sqrt_P = self._chol(P)
        # print(f'sqrt_P=\n{sqrt_P}, \n{sqrt_P@sqrt_P}, \n{P}')
        
        sigma = [x]
        for i in range(self.dim):
            delta = self.gamma * sqrt_P[:, i].reshape(-1,1)
            sigma.append(x + delta)
            sigma.append(x - delta)
        
        return np.array(sigma)


class SetCPKF(SigmaPointsSet, key="cpkf"):
    """
    NOTE: This is not strictly a UKF, but it can be implemented similarly. 
    This "Cubature Kalman Filter" is defined in:
    I. Arasaratnam, S. Haykin, and T. R. Hurd, 
    “Cubature Kalman Filtering for Continuous-Discrete Systems: Theory and Simulations,” 
    IEEE Trans. Signal Process., vol. 58, no. 10, pp. 4977–4993, 2010.
    """
    def __init__(self, dim, param):
        super().__init__(dim)
        
        self.nbSigmaPoint = 2 * self.dim
        self.Wm = np.full(self.nbSigmaPoint, 1. / (2. * self.dim))
        if not np.isclose(self.Wm.sum(), 1.0, atol=EPS_ABS):
            raise ValueError(f"Wm weights do not sum to 1 (sum={self.Wm.sum()})")
        self.Wm /= self.Wm.sum() # normalisation au cas ou il reste un résidu
        self.Wc = np.copy(self.Wm)
        
        self.gamma = np.sqrt(self.dim)

    def _sigma_point(self, x, P):
        
        x = np.atleast_2d(x).reshape(-1,1)  # (dim,1)
        
        # Compute Cholesky factor
        sqrt_P = self._chol(P)
        
        sigma = []
        for i in range(self.dim):
            delta = self.gamma * sqrt_P[:, i].reshape(-1,1)
            sigma.append(x + delta)
            sigma.append(x - delta)
        
        return np.array(sigma)

class SetLERNER2002(SigmaPointsSet, key="lerner2002"):
    """
    Set defined in the paper:
    U. N. Lerner, “Hybrid Bayesian networks for reasoning about complex systems,” 
    Ph.D., Stanford University, 2002.
    There are more points. The unscented transform is exact up to order 4.
    """
    def __init__(self, dim, param):
        super().__init__(dim)
        
        self.nbSigmaPoint = 2 * self.dim**2 + 1
        
        self.Wm = np.zeros(shape=(self.nbSigmaPoint))
        self.Wm[0]                            = (self.dim**2 - 7.*self.dim)/18 + 1.
        self.Wm[1:2*self.dim+1]               = (4-self.dim) / 18.
        self.Wm[2*self.dim+1:2*self.dim**2+1] = 1./36.
        if not np.isclose(self.Wm.sum(), 1.0, atol=EPS_ABS):
            raise ValueError(f"Wm weights do not sum to 1 (sum={self.Wm.sum()})")
        self.Wm /= self.Wm.sum() # normalisation au cas ou il reste un résidu
        self.Wc     = np.copy(self.Wm)
        self.Wc[0] += 1. - param.alpha**2 + param.beta  # same corrective term as for WAN2000
        
        self.gamma  = np.sqrt(3.)

    def _sigma_point(self, x, P):
        
        x = np.atleast_2d(x).reshape(-1,1)  # (dim,1)

        # Compute Cholesky factor
        sqrt_P = self._chol(P)

        # Initialize list of sigma points
        sigma = [x.copy()]  # central point

        # Axial points
        for i in range(self.dim):
            delta = self.gamma * sqrt_P[:, i].reshape(self.dim, 1)
            sigma.append(x + delta)
            sigma.append(x - delta)
        
        delta_plus  = np.empty((self.dim, 1))
        delta_minus = np.empty((self.dim, 1))

        for i in range(self.dim):
            # print(f'i={i}')
            col_i = sqrt_P[:, i]  # vue, pas de copie

            for j in range(i+1, self.dim):
                # print(f'  j={j}')
                col_j = sqrt_P[:, j]  # vue

                # delta_plus = gamma * (col_i + col_j)
                np.add(col_i, col_j, out=delta_plus[:, 0])
                delta_plus *= self.gamma

                # delta_minus = gamma * (col_i - col_j)
                np.subtract(col_i, col_j, out=delta_minus[:, 0])
                delta_minus *= self.gamma

                sigma.append(x + delta_plus)
                sigma.append(x - delta_plus)
                sigma.append(x + delta_minus)
                sigma.append(x - delta_minus)

        return np.array(sigma)

class SetIto2000(SigmaPointsSet, key="ito2000"):
    """
    Set defined in the paper:
    K. Ito and K. Xiong, “Gaussian filters for nonlinear filtering problems,” IEEE Trans. Autom. Control, vol. 45, no. 5, pp. 910–927, May 2000.
    Attention, il y a un paramètre p et le nombre de points explose en p^dim. Donc pour dim = 1,2, rester sur p<=4.
    """
    def __init__(self, dim, param):
        super().__init__(dim)
        
        self.p            = 4
        self.nbSigmaPoint = self.p**self.dim
        
        self.Wm = np.zeros(shape=(self.nbSigmaPoint))
        xi_1d, w_1d = np.polynomial.hermite.hermgauss(self.p)

        # Produit tensoriel
        self.Xi = np.array(list(product(xi_1d, repeat=self.dim)))
        self.Wm = np.prod(np.array(list(product(w_1d, repeat=self.dim))), axis=1)

        # Normalisation (important)
        self.Wm /= np.pi ** (self.dim / 2)
        if not np.isclose(self.Wm.sum(), 1.0, atol=EPS_ABS):
            raise ValueError(f"Wm weights do not sum to 1 (sum={self.Wm.sum()})")
        self.Wm /= self.Wm.sum() # normalisation au cas ou il reste un résidu

        self.Wc     = np.copy(self.Wm)


    def _sigma_point(self, x, P): # self.x + S @ xi
        
        x = np.atleast_2d(x).reshape(-1,1)  # (dim,1)

        # Compute Cholesky factor
        sqrt_P = self._chol(P)
        
        sigma = []
        for xi in self.Xi:
            sigma.append( x + (sqrt_P @ xi).reshape(self.dim, 1))


        return np.array(sigma)
