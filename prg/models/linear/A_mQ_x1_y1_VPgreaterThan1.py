import numpy as np
from .base_model_linear import LinearAmQ  # On utilise directement la sous-classe LinearAmQ


class Model_A_mQ_x1_y1_VPgreaterThan1(LinearAmQ):
    """The A matrix got a VP>1.
       A is not ergodic, with no stationary distribution.
    """

    MODEL_NAME = "A_mQ_x1_y1_VPgreaterThan1"

    def __init__(self) -> None:
        
        # Dimensions x=1, y=1
        dim_x = 1
        dim_y = 1

        A = np.array( [ [0.5813651,  0.22435528],
                             [0.22435528, 1.1186349 ]] )
        mQ = np.array( [[0.739010989010989,   0.27774725274725276], 
                             [0.27774725274725276, 0.9968131868131868]] )
        
        self.z00 = np.zeros(shape=(dim_x+dim_y, 1))
        self.Pz00 = np.eye(dim_x+dim_y)

        super().__init__(dim_x=dim_x, dim_y=dim_y, A=A, mQ=mQ, z00=z00, Pz00=Pz00)

        