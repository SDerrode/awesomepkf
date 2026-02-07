import numpy as np
from .base_model_linear import LinearAmQ  # On utilise directement la sous-classe LinearAmQ


class Model_A_mQ_x1_y1_augmented(BaseModelLinear):
    
    # Nom du modèle
    MODEL_NAME = "A_mQ_x1_y1_augmented"

    def __init__(self) -> None:
        
        # Dimensions x=2, y=1
        dim_x = 2
        dim_y = 1

        a, b, c, d = 0.95, -0.3, 0.2, 0.85
        # a, b, c, d =  0.533, -0.1099, 0.0418, 0.0275
        self.A    = np.array( [[a, b, 0.], 
                               [c, d, 0.], 
                               [c, d, 0.]] )
        self.mQ   = np.diag([0.4, 0.2, 0.])
        # self.mQ   = np.array( [[0.739,  0.2777, 0.0], 
        #                        [0.2777, 0.9968, 0.0],
        #                        [.0    , 0.0   , 0.0 ]])
        self.z00  = np.zeros((self.dim_xy, 1))
        self.Pz00 = np.eye(self.dim_xy)
        
        # print(f'self.A={self.A}')
        # print(f'self.mQ={self.mQ}')

        super().__init__(dim_x=dim_x, dim_y=dim_y, A=A, mQ=mQ, z00=z00, Pz00=Pz00)
        
