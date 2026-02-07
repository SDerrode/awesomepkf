import numpy as np

from .base_model_linear import BaseModelLinear


class Model_Sigma_x3_y1(BaseModelLinear):

    # Nom du modèle
    MODEL_NAME = "Sigma_x3_y1"

    def __init__(self) -> None:
        super().__init__(dim_x=3, dim_y=1, model_type="linear_AmQ")
    
        sxx = np.array([[1.0, 0.4, 0.4],
                        [0.4, 1.0, 0.4],
                        [0.4, 0.4, 1.0]])
        b   = np.array([[0.6, 0.2, 0.4]])
        syy = np.array([[1.0]])
        a   = np.array([[0.5, 0.1, 0.2],
                        [0.4, 0.6, 0.2],
                        [0.4, 0.4, 0.5]])
        d   = np.array([[0.0, 0.0, 0.0]])
        e   = np.array([[0.20],
                        [0.15],
                        [0.25]])
        c   = np.array([[0.30]])

        Q1     = np.block([[sxx, b.T], [b, syy]])
        Q2     = np.block([[a,   e],   [d, c]])

        self._compute_A_mq_z00_Pz00(Q1, Q2)
