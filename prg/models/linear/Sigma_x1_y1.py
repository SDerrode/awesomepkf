import numpy as np
from .base_model_linear import BaseModelLinear

# Nom du modèle
MODEL_NAME = "Sigma_x1_y1"

def create_model():
    
    dim_x, dim_y = 1, 1
    
    sxx = np.matrix([[1]])
    b   = np.matrix([[0.3]])
    syy = np.matrix([[1]])
    a   = np.matrix([[0.5]])
    d   = np.matrix([[0.05]])
    e   = np.matrix([[0.05]])
    c   = np.matrix([[0.04]])

    model = BaseModelLinear(dim_x=dim_x, dim_y=dim_y,
                      sxx=sxx, syy=syy, a=a, b=b, c=c, d=d, e=e)
    return model

if __name__ == "__main__":
    m = create_model()
    m.info()
