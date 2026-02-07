import numpy as np
from .base_model_linear import LinearSigma  # On utilise directement la sous-classe LinearSigma


class Model_Sigma_x2_y2(LinearSigma):
    """
    Modèle linéaire Sigma avec dim_x=2 et dim_y=1.

    Paramétrisation : A, sxx, syy, a, b, c, d, e
    Calcul robuste de la matrice de transition A à partir de Q1 et Q2.
    """
    
    MODEL_NAME = "Sigma_x2_y2"
    
    def __init__(self) -> None:
        super().__init__(dim_x=2, dim_y=2, model_type="linear_AmQ")

        sxx = np.array([[1.0, 0.4],
                        [0.4, 1.0]])
        b   = np.array([[0.6, 0.2],
                        [0.2, 0.6]])
        syy = np.array([[1.0, 0.3],
                        [0.3, 1.0]])
        a   = np.array([[0.5, 0.2],
                        [0.2, 0.5]])
        d   = np.array([[0.1, 0.05],
                        [0.05,0.1]])
        e   = np.array([[0.2, 0.15],
                        [0.15,0.2]])
        c   = np.array([[0.2, 0.1],
                        [0.1, 0.2]])

        Q1  = np.block([[sxx, b.T], [b, syy]])
        Q2  = np.block([[a,   e],   [d, c]])
        
        self._compute_A_mq_z00_Pz00(Q1, Q2)
