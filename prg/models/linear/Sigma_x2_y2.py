import numpy as np
from .base_model import BaseModel

# Nom du modèle
MODEL_NAME = "Sigma_x2_y2"

def create_model():
    
    dim_x, dim_y = 2, 2
    
    sxx = np.matrix([[1.0, 0.4],
                     [0.4, 1.0]])
    b   = np.matrix([[0.6, 0.2],
                     [0.2, 0.6]])
    syy = np.matrix([[1.0, 0.3],
                     [0.3, 1.0]])
    a   = np.matrix([[0.5, 0.2],
                     [0.2, 0.5]])
    d   = np.matrix([[0.1, 0.05],
                     [0.05,0.1]])
    e   = np.matrix([[0.2, 0.15],
                     [0.15,0.2]])
    c   = np.matrix([[0.2, 0.1],
                     [0.1, 0.2]])

    model = BaseModel(dim_x=dim_x, dim_y=dim_y,
                      sxx=sxx, syy=syy, a=a, b=b, c=c, d=d, e=e)
    return model

if __name__ == "__main__":
    m = create_model()
    m.info()
