import numpy as np

from .base_model_linear import BaseModelLinear


class Model_Sigma_x1_y1(BaseModelLinear):
    
    # Nom du modèle
    MODEL_NAME = "Sigma_x1_y1"

    def __init__(self) -> None:
        super().__init__(dim_x=1, dim_y=1, model_type="linear_AmQ")

        sxx = np.array([[1]])
        b   = np.array([[0.3]])
        syy = np.array([[1]])
        a   = np.array([[0.5]])
        d   = np.array([[0.05]])
        e   = np.array([[0.05]])
        c   = np.array([[0.04]])
        
        Q1  = np.block([[sxx, b.T], [b, syy]])
        Q2  = np.block([[a,   e],   [d, c]])
        
        self._compute_A_mq_z00_Pz00(Q1, Q2)
