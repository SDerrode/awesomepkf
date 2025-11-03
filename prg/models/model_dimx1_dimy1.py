import numpy as np

# Equivalent paramétrizations for dim_x, dim_y = 1, 1


def model_dimx1_dimy1_from_A_mQ():
    
    dim_x, dim_y = 1, 1
    
    A = np.array([
        [ 0.53, -0.11],
        [-0.08,  0.42]
    ])

    mQ = np.array([
        [0.74, 0.32],
        [0.32, 0.83]
    ])
    
    return dim_x, dim_y, A, mQ

def model_dimx1_dimy1_from_Sigma():
    
    dim_x, dim_y = 1, 1
    
    sxx = np.matrix([[1]])
    b   = np.matrix([[0.3]])
    syy = np.matrix([[1]])
    a   = np.matrix([[0.5]])
    d   = np.matrix([[0.05]])
    e   = np.matrix([[0.05]])
    c   = np.matrix([[0.04]])
    
    return dim_x, dim_y, sxx, syy, a, b, c, d, e
