import numpy as np

def model_dimx2_dimy1():
    
    dim_x, dim_y = 2, 1
    
    def f(x):
        print(f'x={x}')
        input('model_dimx2_dimy1')
        x1, x2 = x
        return np.array([
            x1 + 0.05 * x2 + 0.5 * np.sin(0.1 * x2),
            0.9 * x2 + 0.2 * np.cos(0.3 * x1)
        ])

    def h(x):
        print(f'x={x}')
        input('model_dimx2_dimy1')
        x1, x2 = x
        return np.sqrt(x1**2 + x2**2)
    
    
    # Paramètres
    mQ = np.array( [
        [1e-2, 0.,   0.],
        [0.,   1e-2, 0.],
        [0,    0.,   1e-1] ] )
    x0 = np.array([0.5, 0.2])
    P0 = np.eye(2)
    
    return dim_x, dim_y, f, h, mQ, x0, P0

