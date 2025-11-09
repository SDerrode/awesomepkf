import numpy as np

# Equivalent paramétrizations for dim_x, dim_y = 1, 1


# Homogeneous model
def model_dim_x1_dim_y1_from_A_mQ_bis(): 
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

    return dim_x, dim_y, A, mQ, z00, Pz00


# Homogeneous model
def model_dim_x1_dim_y1_from_A_mQ():
    
    dim_x, dim_y = 1, 1
    
    A = np.array( [[0.532967032967033, -0.10989010989010986], 
                   [0.04175824175824177, 0.027472527472527472]] )
    mQ = np.array( [[0.739010989010989, 0.27774725274725276], 
                    [0.27774725274725276, 0.9968131868131868]] )
    z00 = np.zeros(shape=(dim_x+dim_y, 1))
    Pz00 = np.eye(dim_x+dim_y)

    return dim_x, dim_y, A, mQ, z00, Pz00

# Stationary model (equivalent to previous)
def model_dim_x1_dim_y1_from_Sigma():
    
    dim_x, dim_y = 1, 1
    
    sxx = np.matrix([[1]])
    b   = np.matrix([[0.3]])
    syy = np.matrix([[1]])
    a   = np.matrix([[0.5]])
    d   = np.matrix([[0.05]])
    e   = np.matrix([[0.05]])
    c   = np.matrix([[0.04]])
    
    return dim_x, dim_y, sxx, syy, a, b, c, d, e
