import numpy as np
from .base_model_linear import BaseModelLinear

# A few utils functions that are used several times
from others.Utils import check_consistency

class Model_Sigma_x3_y1(BaseModelLinear):

    # Nom du modèle
    MODEL_NAME = "Sigma_x3_y1"

    def __init__(self) -> None:
        super().__init__(dim_x=3, dim_y=1, model_type="Sigma")
    
        self.sxx = np.matrix([[1.0, 0.4, 0.4],
                        [0.4, 1.0, 0.4],
                        [0.4, 0.4, 1.0]])
        self.b   = np.matrix([[0.6, 0.2, 0.4]])
        self.syy = np.matrix([[1.0]])
        self.a   = np.matrix([[0.5, 0.1, 0.2],
                        [0.4, 0.6, 0.2],
                        [0.4, 0.4, 0.5]])
        self.d   = np.matrix([[0.0, 0.0, 0.0]])
        self.e   = np.matrix([[0.2],
                        [0.15],
                        [0.25]])
        self.c   = np.matrix([[0.3]])

        if __debug__:
            check_consistency(sxx=self.sxx, syy=self.syy)

