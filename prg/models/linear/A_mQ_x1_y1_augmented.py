import numpy as np
from .base_model_linear import LinearAmQ  # On utilise directement la sous-classe LinearAmQ


class Model_A_mQ_x1_y1_augmented(LinearAmQ):
    
    # Nom du modèle
    MODEL_NAME = "A_mQ_x1_y1_augmented"

    def __init__(self) -> None:
        
        # Dimensions x=2, y=1
        dim_x = 2
        dim_y = 1

        a, b, c, d = 0.95, -0.3, 0.2, 0.85
        # a, b, c, d =  0.533, -0.1099, 0.0418, 0.0275
        A    = np.array( [[a, b, 0.], 
                               [c, d, 0.], 
                               [c, d, 0.]] )
        mQ   = np.diag([0.4, 0.2, 0.])
        # mQ   = np.array( [[0.739,  0.2777, 0.0], 
        #                        [0.2777, 0.9968, 0.0],
        #                        [.0    , 0.0   , 0.0 ]])
        
        z00 = np.zeros(shape=(dim_x+dim_y, 1))
        Pz00 = np.eye(dim_x+dim_y)

        super().__init__(dim_x=dim_x, dim_y=dim_y, A=A, mQ=mQ, z00=z00, Pz00=Pz00, augmented=True)

