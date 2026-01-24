from abc import ABC, abstractmethod
from typing import Dict, Type

import numpy as np


class SigmaPointsSet(ABC):
    """
    Classe de base pour tous les jeux de sigma-points.
    """

    registry: Dict[str, Type["SigmaPointsSet"]] = {}

    def __init_subclass__(cls, *, key: str, **kwargs):
        super().__init_subclass__(**kwargs)

        if key in SigmaPointsSet.registry:
            raise RuntimeError(f"Clé déjà enregistrée : {key}")

        SigmaPointsSet.registry[key] = cls

    def __init__(self, dim_x: int):
        self.dim_x = dim_x

    @abstractmethod
    def _sigma_point(self):
        pass


class SetWAN2000(SigmaPointsSet, key="wan2000"):
    def __init__(self, dim_x, param):
        super().__init__(dim_x)
        
        self.nbSigmaPoint = 2 * self.dim_x + 1
        self.Wm = np.full(self.nbSigmaPoint, 1. / (2. * (self.dim_x + param.lambda_)))
        self.Wc = np.copy(self.Wm)
        self.Wm[0] = param.lambda_ / (self.dim_x + param.lambda_)
        self.Wc[0] = param.lambda_ / (self.dim_x + param.lambda_) + (1. - param.alpha**2 + param.beta)
        # print(f'self.Wm={self.Wm}')
        self.gamma = param.gamma
        

    def _sigma_point(self, x, P):
        
        """Generate the 2*dim_x+1 sigma points around x"""
        A = np.linalg.cholesky(P)
        sigma = [x]
        for i in range(self.dim_x):
            sigma.append(x + self.gamma * A[:, i].reshape(-1,1))
            sigma.append(x - self.gamma * A[:, i].reshape(-1,1))
        return np.array(sigma)


class SetCPKF(SigmaPointsSet, key="cpkf"):
    def __init__(self, dim_x, param):
        super().__init__(dim_x)
        
        self.nbSigmaPoint = 2 * self.dim_x
        self.Wm = np.full(self.nbSigmaPoint, 1. / (2. * self.dim_x))
        self.Wc = np.copy(self.Wm)

    def _sigma_point(self, x, P):
        
        A = np.linalg.cholesky(P)
        sigma = []
        for i in range(self.dim_x):
            sigma.append(x + np.sqrt(self.dim_x) * A[:, i].reshape(-1,1))
            sigma.append(x - np.sqrt(self.dim_x) * A[:, i].reshape(-1,1))
        return np.array(sigma)

# class SetLERNER2002(SigmaPointsSet, key="lerner2002"):
#     def __init__(self, dim_x, param):
#         super().__init__(dim_x)

#     def _sigma_point(self, x, P):
#         return f"Sigma-points LERNER2002 (dim_x={self.dim_x})"
