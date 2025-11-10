import numpy as np
from .base_model_linear import BaseModelLinear

# Nom du modèle
MODEL_NAME = "A_mQ_x1_y1"

def create_model():
    
    dim_x, dim_y = 1, 1
    
    A = np.array( [[0.532967032967033, -0.10989010989010986], 
                   [0.04175824175824177, 0.027472527472527472]] )
    mQ = np.array( [[0.739010989010989, 0.27774725274725276], 
                    [0.27774725274725276, 0.9968131868131868]] )
    z00 = np.zeros(shape=(dim_x+dim_y, 1))
    Pz00 = np.eye(dim_x+dim_y)

    model = BaseModelLinear(dim_x=dim_x, dim_y=dim_y, A=A, mQ=mQ, z00=z00, Pz00=Pz00)
    return model

# Optionnel : permettre le test rapide du fichier
if __name__ == "__main__":
    m = create_model()
    m.info()
