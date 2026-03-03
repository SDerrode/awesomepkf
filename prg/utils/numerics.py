#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Numerical constants and tolerances used throughout the PKF codebase.

All floating-point computations use FLOAT_DTYPE (float64 by default).
Thresholds are defined here to ensure consistency across all modules.

Constants
---------
FLOAT_DTYPE : np.dtype
    Reference floating-point type (float64).
EPS : float
    Machine epsilon for FLOAT_DTYPE (~2.22e-16).
SQRT_EPS : float
    Square root of machine epsilon (~1.49e-8).
EPS_ABS : float
    Absolute tolerance for symmetry checks, weight validation, and PSD tests.
EPS_REL : float
    Relative tolerance for scale-dependent comparisons.
EIG_TOL_WARN : float
    Eigenvalue threshold below which a warning is issued.
    Negative eigenvalues above this level are considered numerical noise.
EIG_TOL_FAIL : float
    Eigenvalue threshold below which an error is raised.
    Negative eigenvalues below this level indicate a physically invalid matrix.
COND_WARN : float
    Condition number threshold above which a warning is issued.
COND_FAIL : float
    Condition number threshold above which regularisation is applied.
"""

import numpy as np

__all__ = [
    "FLOAT_DTYPE",
    "EPS",
    "SQRT_EPS",
    "EPS_ABS",
    "EPS_REL",
    "EIG_TOL_WARN",
    "EIG_TOL_FAIL",
    "COND_WARN",
    "COND_FAIL",
]

# ---------------------------------------------------------------------------
# Floating-point type
# ---------------------------------------------------------------------------
FLOAT_DTYPE = np.float64

# ---------------------------------------------------------------------------
# Machine precision
# ---------------------------------------------------------------------------
EPS: float = np.finfo(FLOAT_DTYPE).eps  # ~2.22e-16
SQRT_EPS: float = np.sqrt(EPS)  # ~1.49e-08

# ---------------------------------------------------------------------------
# Numerical tolerances
# ---------------------------------------------------------------------------

# Absolute tolerance — symmetry checks, weight validation, PSD tests
EPS_ABS: float = 1e-12

# Relative tolerance — scale-dependent comparisons
EPS_REL: float = 1e-6

# ---------------------------------------------------------------------------
# Eigenvalue tolerances
# ---------------------------------------------------------------------------

# Below EIG_TOL_WARN: eigenvalue is suspicious but likely numerical noise
# (covers observed values as small as ~1e-48 in strongly converged filters)
EIG_TOL_WARN: float = -EPS  # ~ -2.22e-16

# Below EIG_TOL_FAIL: eigenvalue is unambiguously negative — invalid matrix
EIG_TOL_FAIL: float = -EPS_ABS  # ~ -1e-12

# ---------------------------------------------------------------------------
# Condition number thresholds
# ---------------------------------------------------------------------------

# Above COND_WARN: matrix is becoming ill-conditioned — log a warning
COND_WARN: float = 1e12

# Above COND_FAIL: matrix is critically ill-conditioned — apply regularisation
COND_FAIL: float = 1e30
