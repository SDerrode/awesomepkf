import numpy as np
from .base_model import BaseModel

# Nom du modèle
MODEL_NAME = "A_mQ_x1_y1_VPgreaterThan1"

def create_model():
    """The A matrix got a VP>1.
       A is not ergodic, with no stationary distribution.
    """
    
    dim_x, dim_y = 1, 1
    
    A = np.array( [ [0.5813651,  0.22435528],
                    [0.22435528, 1.1186349 ]] )
    mQ = np.array( [[0.739010989010989, 0.27774725274725276], 
                    [0.27774725274725276, 0.9968131868131868]] )
    z00 = np.zeros(shape=(dim_x+dim_y, 1))
    Pz00 = np.eye(dim_x+dim_y)

    model = BaseModel(dim_x=dim_x, dim_y=dim_y, A=A, mQ=mQ, z00=z00, Pz00=Pz00)
    return model

# Optionnel : permettre le test rapide du fichier
if __name__ == "__main__":
    m = create_model()
    m.info()
