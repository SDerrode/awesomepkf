import numpy as np

# Equivalent paramétrizations for dim_x, dim_y = 1, 1


def model_dimx1_dimy1_from_A_mQ():
    
    dim_x, dim_y = 1, 1
    
    A = np.array( [[0.532967032967033, -0.10989010989010986], 
                   [0.04175824175824177, 0.027472527472527472]] )
    mQ = np.array( [[0.739010989010989, 0.27774725274725276], 
                    [0.27774725274725276, 0.9968131868131868]] )

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
