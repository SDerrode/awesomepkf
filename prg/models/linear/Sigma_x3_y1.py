import numpy as np
from .base_model import BaseModel

# Nom du modèle
MODEL_NAME = "Sigma_x3_y1"

def create_model():
    
    dim_x, dim_y = 3, 1
    
    sxx = np.matrix([[1.0, 0.4, 0.4],
                     [0.4, 1.0, 0.4],
                     [0.4, 0.4, 1.0]])
    b   = np.matrix([[0.6, 0.2, 0.4]])
    syy = np.matrix([[1.0]])
    a   = np.matrix([[0.5, 0.1, 0.2],
                     [0.4, 0.6, 0.2],
                     [0.4, 0.4, 0.5]])
    d   = np.matrix([[0.0, 0.0, 0.0]])
    e   = np.matrix([[0.2],
                     [0.15],
                     [0.25]])
    c   = np.matrix([[0.3]])

    model = BaseModel(dim_x=dim_x, dim_y=dim_y,
                      sxx=sxx, syy=syy, a=a, b=b, c=c, d=d, e=e)
    return model

if __name__ == "__main__":
    m = create_model()
    m.info()
