import numpy as np
from .base_model_linear import BaseModelLinear

# A few utils functions that are used several times
from others.Utils import check_consistency

class Model_A_mQ_x1_y1_VPgreaterThan1(BaseModelLinear):
    """The A matrix got a VP>1.
       A is not ergodic, with no stationary distribution.
    """
    
    # Nom du modèle
    MODEL_NAME = "A_mQ_x1_y1_VPgreaterThan1"

    def __init__(self) -> None:
        super().__init__(dim_x=3, dim_y=1, model_type="AmQ")

        self.A = np.array( [ [0.5813651,  0.22435528],
                        [0.22435528, 1.1186349 ]] )
        self.mQ = np.array( [[0.739010989010989, 0.27774725274725276], 
                        [0.27774725274725276, 0.9968131868131868]] )
        self.z00 = np.zeros(shape=(self.dim_x+self.dim_y, 1))
        self.Pz00 = np.eye(self.dim_x+self.dim_y)

        if __debug__:
            check_consistency(mQ=self.mQ, Pz00=self.Pz00)
