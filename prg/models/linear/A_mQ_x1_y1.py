import numpy as np
from .base_model_linear import BaseModelLinear

# A few utils functions that are used several times
from others.utils import check_consistency

class Model_A_mQ_x1_y1(BaseModelLinear):
    
    # Nom du modèle
    MODEL_NAME = "A_mQ_x1_y1"

    def __init__(self) -> None:
        super().__init__(dim_x=1, dim_y=1, model_type="linear_AmQ")

        a, b, c, d = 0.95, -0.3, 0.2, 0.85
        # a, b, c, d =  0.533, -0.1099, 0.0418, 0.0275
        self.A    = np.array( [[a, b],
                               [c, d]] )
        # self.mQ   = np.array( [[0.739,  0.2777], 
        #                        [0.2777, 0.9968]])
        self.mQ   = np.diag( [0.4, 0.2])
        self.z00  = np.zeros((self.dim_xy, 1))
        self.Pz00 = np.eye(self.dim_xy)

        if __debug__:
            check_consistency(mQ=self.mQ, Pz00=self.Pz00)
