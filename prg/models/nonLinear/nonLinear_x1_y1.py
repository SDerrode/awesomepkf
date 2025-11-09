import numpy as np


def model_x1_y1_cubique():
    """This an UPKF model that simulates a UKF one."""
    
    dim_x, dim_y = 1, 1
    
    def fx(x, noise, dt):
        return 0.9*x - 0.2*x**3 + noise

    def hx(x, noise, dt):
        return x + noise

    def g(z, noise, dt):
        fx12   = fx(z[0:dim_x], noise[0:dim_x], dt)
        h      = hx(fx12, noise[dim_x:], dt)
        gvalue = np.vstack((fx12, h))
        return gvalue

    # Paramètres
    mQ = np.array( [
        [1e-2, 0.],
        [0,    1e-1] ] )
    
    z00  = np.zeros(shape=(dim_x+dim_y, 1))
    Pz00 = np.eye(dim_x+dim_y)
    
    alpha = 0.01
    beta  = 2.   # optimal pour les gaussiennes
    kappa = 0. 
    
    return dim_x, dim_y, g, mQ, z00, Pz00, alpha, beta, kappa


def model_x1_y1_ext_saturant():
    """This an UPKF model that simulates a UKF one."""
    
    dim_x, dim_y = 1, 1
    
    def fx(x, noise, dt):
        return 0.5*x + 2*(1 - np.exp(-0.1*x)) + noise

    def hx(x, noise, dt):
        return np.log(1 + max(np.abs(x), 1E-8) ) + noise

    def g(z, noise, dt):
        fx12   = fx(z[0:dim_x], noise[0:dim_x], dt)
        h      = hx(fx12, noise[dim_x:], dt)
        gvalue = np.vstack((fx12, h))
        return gvalue

    # Paramètres
    mQ = np.array( [
        [1e-4, 0.],
        [0,    1e-3] ] )
    
    z00  = np.zeros(shape=(dim_x+dim_y, 1)) +0.5
    Pz00 = np.eye(dim_x+dim_y) * 2
    
    alpha = 0.01
    beta  = 2.   # optimal pour les gaussiennes
    kappa = 0 
    
    return dim_x, dim_y, g, mQ, z00, Pz00, alpha, beta, kappa


def model_x1_y1_sinus():
    """This an UPKF model that simulates a UKF one."""
    
    dim_x, dim_y = 1, 1
    
    def fx(x, noise, dt):
        return 0.8*x + 0.3*np.sin(x) + noise

    def hx(x, noise, dt):
        return x**2 + noise

    def g(z, noise, dt):
        fx12   = fx(z[0:dim_x], noise[0:dim_x], dt)
        h      = hx(fx12, noise[dim_x:], dt)
        gvalue = np.vstack((fx12, h))
        return gvalue

    # Paramètres
    mQ = np.array( [
        [1e-3, 0.],
        [0,    1e-2] ] )
    
    z00  = np.zeros(shape=(dim_x+dim_y, 1)) -1
    Pz00 = np.eye(dim_x+dim_y)
    
    alpha = 0.01
    beta  = 2.   # optimal pour les gaussiennes
    kappa = 0. 
    
    return dim_x, dim_y, g, mQ, z00, Pz00, alpha, beta, kappa


def model_x1_y1_gordon():
    """This an UPKF model that simulates a UKF one.
        Ce modèle a été introduit par Gordon et al. (1993) dans l’article fondateur sur les particle filters :
        Gordon, N., Salmond, D., & Smith, A. (1993). Novel approach to nonlinear/non-Gaussian Bayesian state estimation. 
        IEE Proceedings F, 140(2), 107–113.DOI: 10.1049/ip-f-2.1993.0015"""
    
    dim_x, dim_y = 1, 1
    
    def fx(x, noise, dt):
        return 0.5*x + 25*x/(1 + x**2) + 8*np.cos(1.2*dt) + noise

    def hx(x, noise, dt):
        return 0.05 * x**2 + noise

    def g(z, noise, dt):
        fx12   = fx(z[0:dim_x], noise[0:dim_x], dt)
        h      = hx(fx12, noise[dim_x:], dt)
        gvalue = np.vstack((fx12, h))
        return gvalue

    # Paramètres
    mQ = np.array( [
        [1e-2, 0.],
        [0,    1e-1] ] )
    
    z00  = np.zeros(shape=(dim_x+dim_y, 1))
    Pz00 = np.eye(dim_x+dim_y)
    
    alpha = 0.01
    beta  = 2.   # optimal pour les gaussiennes
    kappa = 0. 
    
    return dim_x, dim_y, g, mQ, z00, Pz00, alpha, beta, kappa



