import numpy as np
from .base_model_linear import BaseModelLinear

# A few utils functions that are used several times
from others.utils import check_consistency

class Model_A_mQ_x2_y1_augmented(BaseModelLinear):
    
    # Nom du modèle
    MODEL_NAME = "A_mQ_x2_y1_augmented"

    def __init__(self) -> None:
        super().__init__(dim_x=2, dim_y=1, model_type="linear_AmQ")

        self.a, self.b, self.c, self.d = 0.95, -0.3, 0.2, 0.85
        self.A    = np.array( [[self.a, self.b, 0.], 
                               [self.c, self.d, 0.], 
                               [self.c, self.d, 0.]] )
        self.mQ   = np.diag([0.1, 0.2, 0.])
        self.z00  = np.array([[0.], [0.], [0.]])
        self.Pz00 = np.eye(self.dim_x+self.dim_y)

        if __debug__:
            check_consistency(mQ=self.mQ, Pz00=self.Pz00)
