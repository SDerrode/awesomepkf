import numpy as np
from .base_model_linear import BaseModelLinear

# A few utils functions that are used several times
from others.Utils import check_consistency

class Model_Sigma_x1_y1(BaseModelLinear):
    
    # Nom du modèle
    MODEL_NAME = "Sigma_x1_y1"

    def __init__(self) -> None:
        super().__init__(dim_x=1, dim_y=1, model_type="Sigma")

        self.sxx = np.matrix([[1]])
        self.b   = np.matrix([[0.3]])
        self.syy = np.matrix([[1]])
        self.a   = np.matrix([[0.5]])
        self.d   = np.matrix([[0.05]])
        self.e   = np.matrix([[0.05]])
        self.c   = np.matrix([[0.04]])
       
        if __debug__:
            check_consistency(sxx=self.sxx, syy=self.syy)
