import numpy as np

def model_dim_x2_dim_y1():
    """This an UPKF model that simulate a UKF one"""
    
    dim_x, dim_y = 2, 1
    
    def _f(x, noise, dt):
        x1,  x2  = x
        nx1, nx2 = noise
        return np.array([
            x1 + 0.05 * x2 + 0.5 * np.sin(0.1 * x2) + nx1,
            0.9 * x2 + 0.2 * np.cos(0.3 * x1)       + nx2
        ])

    def _h(x, noise, dt):
        x1, x2 = x
        return np.sqrt(x1**2 + x2**2) + noise
    
    def g(z, noise, dt):
        fx12   = _f(z[0:dim_x], noise[0:dim_x], dt)
        h      = _h(fx12, noise[dim_x:], dt)
        gvalue = np.vstack((fx12, h))
        return gvalue

    # Paramètres
    mQ = np.array( [
        [1e-2, 0.,   0.],
        [0.,   1e-2, 0.],
        [0,    0.,   1e-1] ] )
    
    z00  = np.zeros(shape=(dim_x+dim_y, 1))
    Pz00 = np.eye(dim_x+dim_y)
    
    alpha = 1e-3
    beta  = 2.   # optimal pour les gaussiennes
    kappa = 0. 
    
    return dim_x, dim_y, g, mQ, z00, Pz00, alpha, beta, kappa

