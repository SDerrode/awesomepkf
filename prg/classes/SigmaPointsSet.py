#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from typing import Dict, Type
import warnings
import numpy as np
from itertools import product

eps = 1E-12

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

    def __init__(self, dim_x: int):
        self.dim_x = dim_x

    @abstractmethod
    def _sigma_point(self):
        pass
    
    def _chol(self, P):
        try:
            return np.linalg.cholesky(P)
        except np.linalg.LinAlgError:
            warnings.warn("P is not positive definite, regularization applied")
            return np.linalg.cholesky(P + eps*np.eye(self.dim_x))


class SetJULIER1995(SigmaPointsSet, key="julier1995"):
    """
    Original set defined in the paper:
    S. J. Julier, J. K. Uhlmann, and H. F. Durrant-Whyte, 
    “A new approach for filtering nonlinear systems,” 
    in Proc. IEEE American Control Conf. (ACC’95), 1995, pp. 1628–1632.
    """
    
    def __init__(self, dim_x, param):
        super().__init__(dim_x)
        
        self.nbSigmaPoint = 2 * self.dim_x + 1
        self.kappaJulier  = param.kappaJulier
        
        self.Wm = np.full(self.nbSigmaPoint, self.kappaJulier / (self.kappaJulier + self.dim_x))
        self.Wc = np.copy(self.Wm)
        
        self.gamma = np.sqrt(self.dim_x + self.kappaJulier)

    def _sigma_point(self, x, P):
        
        x = np.atleast_2d(x).reshape(-1,1)  # (dim_x,1)
        
        # Compute Cholesky factor
        sqrt_P = self._chol(P)
        sigma  = [x]
        for i in range(self.dim_x):
            delta = self.gamma * sqrt_P[:, i].reshape(-1,1)
            sigma.append(x + delta)
            sigma.append(x - delta)
        
        return np.array(sigma)

class SetWAN2000(SigmaPointsSet, key="wan2000"):
    """
    The most commonly used set, defined in the paper:
    E. A. Wan and R. V. D. Merwe, 
    “The unscented Kalman filter for nonlinear estimation,” 
    in Proc. IEEE Adaptive Syst. Signal Process. Commun. Control Symp. (ASSPCCS’00), 2000, pp. 153–158.
    """
    
    def __init__(self, dim_x, param):
        super().__init__(dim_x)
        
        self.nbSigmaPoint = 2 * self.dim_x + 1
        
        self.Wm     = np.full(self.nbSigmaPoint, 1. / (2. * (self.dim_x + param.lambda_)))
        self.Wm[0]  = param.lambda_ / (self.dim_x + param.lambda_)
        if not np.isclose(self.Wm.sum(), 1.0, atol=eps):
            raise ValueError(f"Wm weights do not sum to 1 (sum={self.Wm.sum()})")
        
        self.Wc     = np.copy(self.Wm)
        self.Wc[0] += 1. - param.alpha**2 + param.beta  # corrective term

        self.gamma  = np.sqrt(self.dim_x + param.lambda_)
        

    def _sigma_point(self, x, P):
        
        x = np.atleast_2d(x).reshape(-1,1)  # (dim_x,1)
        
        # Compute Cholesky factor
        sqrt_P = self._chol(P)
            
        sigma = [x]
        for i in range(self.dim_x):
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
    def __init__(self, dim_x, param):
        super().__init__(dim_x)
        
        self.nbSigmaPoint = 2 * self.dim_x
        self.Wm = np.full(self.nbSigmaPoint, 1. / (2. * self.dim_x))
        if not np.isclose(self.Wm.sum(), 1.0, atol=eps):
            raise ValueError(f"Wm weights do not sum to 1 (sum={self.Wm.sum()})")
        
        self.Wc = np.copy(self.Wm)
        
        self.gamma = np.sqrt(self.dim_x)

    def _sigma_point(self, x, P):
        
        x = np.atleast_2d(x).reshape(-1,1)  # (dim_x,1)
        
        # Compute Cholesky factor
        sqrt_P = self._chol(P)
        
        sigma = []
        for i in range(self.dim_x):
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
    def __init__(self, dim_x, param):
        super().__init__(dim_x)
        
        self.nbSigmaPoint = 2 * self.dim_x**2 + 1
        
        self.Wm = np.zeros(shape=(self.nbSigmaPoint))
        self.Wm[0]                                = (self.dim_x**2 - 7.*self.dim_x)/18 + 1.
        self.Wm[1:2*self.dim_x+1]                 = (4-self.dim_x) / 18.
        self.Wm[2*self.dim_x+1:2*self.dim_x**2+1] = 1./36.
        if not np.isclose(self.Wm.sum(), 1.0, atol=eps):
            raise ValueError(f"Wm weights do not sum to 1 (sum={self.Wm.sum()})")

        self.Wc     = np.copy(self.Wm)
        self.Wc[0] += 1. - param.alpha**2 + param.beta  # same corrective term as for WAN2000
        
        self.gamma  = np.sqrt(3.)

    def _sigma_point(self, x, P):
        
        x = np.atleast_2d(x).reshape(-1,1)  # (dim_x,1)

        # Compute Cholesky factor
        sqrt_P = self._chol(P)

        # Initialize list of sigma points
        sigma = [x.copy()]  # central point

        # Axial points
        for i in range(self.dim_x):
            delta = self.gamma * sqrt_P[:, i].reshape(self.dim_x, 1)
            sigma.append(x + delta)
            sigma.append(x - delta)

        # Cross points
        for i in range(self.dim_x):
            for j in range(i+1, self.dim_x):
                delta_plus  = self.gamma * (sqrt_P[:, i] + sqrt_P[:, j]).reshape(self.dim_x, 1)
                delta_minus = self.gamma * (sqrt_P[:, i] - sqrt_P[:, j]).reshape(self.dim_x, 1)
                sigma.append(x + delta_plus)
                sigma.append(x - delta_plus)
                sigma.append(x + delta_minus)
                sigma.append(x - delta_minus)

        return np.array(sigma)



class SetIto2000(SigmaPointsSet, key="ito2000"):
    """
    Set defined in the paper:
    K. Ito and K. Xiong, “Gaussian filters for nonlinear filtering problems,” IEEE Trans. Autom. Control, vol. 45, no. 5, pp. 910–927, May 2000.
    Attention, il y a un paramètre p et le nombre de points explose en p^dim_x. Donc pour dim_x = 1,2, rester sur p<=4.
    """
    def __init__(self, dim_x, param):
        super().__init__(dim_x)
        
        self.p            = 4
        self.nbSigmaPoint = self.p**self.dim_x
        
        self.Wm = np.zeros(shape=(self.nbSigmaPoint))
        xi_1d, w_1d = np.polynomial.hermite.hermgauss(self.p)

        # Produit tensoriel
        self.Xi = np.array(list(product(xi_1d, repeat=self.dim_x)))
        self.Wm = np.prod(np.array(list(product(w_1d, repeat=self.dim_x))), axis=1)

        # Normalisation (important)
        self.Wm /= np.pi ** (self.dim_x / 2)
        # print(f'self.Wm={self.Wm}')
        if not np.isclose(self.Wm.sum(), 1.0, atol=eps):
            raise ValueError(f"Wm weights do not sum to 1 (sum={self.Wm.sum()})")

        self.Wc     = np.copy(self.Wm)


    def _sigma_point(self, x, P): # self.x + S @ xi
        
        x = np.atleast_2d(x).reshape(-1,1)  # (dim_x,1)

        # Compute Cholesky factor
        sqrt_P = self._chol(P)
        
        sigma = []
        for xi in self.Xi:
            # print(f'xi=', xi)
            # print(f'x=', x)
            # print((sqrt_P @ xi).reshape(self.dim_x, 1))
            sigma.append( x + (sqrt_P @ xi).reshape(self.dim_x, 1))
        #     input('apuse')
        
        # print(len(sigma))
        # print(sigma[0])
        # print(sigma[1])
        # exit(1)

        return np.array(sigma)
