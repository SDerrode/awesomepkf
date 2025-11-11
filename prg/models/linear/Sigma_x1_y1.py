import numpy as np
from .base_model_linear import BaseModelLinear

# A few utils functions that are used several times
from others.Utils import check_consistency

class Model_Sigma_x1_y1(BaseModelLinear):
    
    # Nom du modèle
    MODEL_NAME = "Sigma_x1_y1"

    def __init__(self) -> None:
        super().__init__(dim_x=1, dim_y=1, model_type="linear_Sigma")

        self.sxx = np.array([[1]])
        self.b   = np.array([[0.3]])
        self.syy = np.array([[1]])
        self.a   = np.array([[0.5]])
        self.d   = np.array([[0.05]])
        self.e   = np.array([[0.05]])
        self.c   = np.array([[0.04]])
        
        Q1     = np.block([[self.sxx, self.b.T], [self.b, self.syy]])
        Q2     = np.block([[self.a, self.e], [self.d, self.c]])
        self.A = Q2 @ np.linalg.inv(Q1)
        eigvals = np.linalg.eigvals(self.A)
        if np.any(np.abs(eigvals) >= 1.0):
            raise ValueError(f"⚠️ The modulus of one Eigen value of A is >= 1 : {eigvals}")

        if __debug__:
            check_consistency(sxx=self.sxx, syy=self.syy)
