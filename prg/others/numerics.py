import numpy as np

FLOAT_DTYPE = np.float64
EPS = np.finfo(FLOAT_DTYPE).eps  # ≈ 2.22e-16
SQRT_EPS = np.sqrt(EPS)

# Seuils numériques pour le calcul ±2σ
EPS_ABS = 1e-12  # seuil absolu (bruit machine)
EPS_REL = 1e-6  # seuil relatif à l'échelle

COND_FAIL = 1e12
COND_WARN = 1e8
