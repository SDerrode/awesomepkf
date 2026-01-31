import numpy as np
from .base_model_linear import LinearAmQ  # On utilise directement la sous-classe LinearAmQ


class Model_A_mQ_x1_y1(LinearAmQ):
    
    MODEL_NAME = "A_mQ_x1_y1"

    def __init__(self) -> None:
        
        # Dimensions x=1, y=1
        dim_x = 1
        dim_y = 1
        
        a, b, c, d = 0.95, -0.3, 0.2, 0.85
        A = np.array( [[a, b], 
                       [c, d]] )
        mQ   = np.diag([0.1, 0.2])
        z00  = np.array([[0.], [0.]])
        Pz00 = np.eye(dim_x+dim_y)
        
        super().__init__(dim_x=dim_x, dim_y=dim_y, A=A, mQ=mQ, z00=z00, Pz00=Pz00)

