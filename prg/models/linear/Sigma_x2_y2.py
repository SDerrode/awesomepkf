import numpy as np
from .base_model_linear import BaseModelLinear

# A few utils functions that are used several times
from others.Utils import check_consistency

class Model_Sigma_x2_y2(BaseModelLinear):
    
    # Nom du modèle
    MODEL_NAME = "Sigma_x2_y2"
    
    def __init__(self) -> None:
        super().__init__(dim_x=2, dim_y=2, model_type="Sigma")

        self.sxx = np.matrix([[1.0, 0.4],
                        [0.4, 1.0]])
        self.b   = np.matrix([[0.6, 0.2],
                        [0.2, 0.6]])
        self.syy = np.matrix([[1.0, 0.3],
                        [0.3, 1.0]])
        self.a   = np.matrix([[0.5, 0.2],
                        [0.2, 0.5]])
        self.d   = np.matrix([[0.1, 0.05],
                        [0.05,0.1]])
        self.e   = np.matrix([[0.2, 0.15],
                        [0.15,0.2]])
        self.c   = np.matrix([[0.2, 0.1],
                     [0.1, 0.2]])

        if __debug__:
            check_consistency(sxx=self.sxx, syy=self.syy)
