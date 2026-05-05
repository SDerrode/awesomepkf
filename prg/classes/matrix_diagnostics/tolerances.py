"""Per-matrix-type tolerance dataclasses."""

from dataclasses import dataclass

__all__ = [
    "CovarianceTolerances",
    "InvertibleTolerances",
    "StabilityTolerances",
]


@dataclass
class CovarianceTolerances:
    """Thresholds for a covariance matrix."""

    # Symmetry: max|M - M^T| / max|M|
    symmetry_warn: float = 1e-8
    symmetry_fail: float = 1e-5

    # Minimum eigenvalues
    eigenvalue_warn: float = 1e-10  # near-singular
    eigenvalue_fail: float = 0.0  # negative or zero eigenvalue

    # Condition number
    condition_warn: float = 1e6
    condition_fail: float = 1e12

    # Diagonal (variance > 0)
    diagonal_fail: float = 0.0


@dataclass
class InvertibleTolerances:
    """Thresholds for an arbitrary invertible matrix."""

    # Condition number (rcond)
    condition_warn: float = 1e6
    condition_fail: float = 1e12

    # |det| minimum
    det_warn: float = 1e-10
    det_fail: float = 1e-15

    # Expected rank (None = n)
    expected_rank: int | None = None
    # rank_tol : None = automatic relative threshold from numpy (recommended)
    #            float = minimum absolute threshold (floor only)
    rank_tol: float = None  # ← was 1e-10, too restrictive

    # Residual after inversion: ||I - M @ M_inv||_F
    residual_warn: float = 1e-8
    residual_fail: float = 1e-5


@dataclass
class StabilityTolerances:
    """Thresholds for a stability matrix (eigenvalues in [0, 1])."""

    # Strictly negative eigenvalue
    eigenvalue_min: float = 0.0

    # Eigenvalue strictly larger than 1
    eigenvalue_max: float = 1.0

    # Numerical tolerance margin around bounds
    # (avoids false positives due to rounding errors)
    tol_boundary: float = 1e-10

    # Eigenvalue close to 1 (near-unstable, warning)
    near_unit_warn: float = 0.99

    # Eigenvalue close to 0 (near-zero, warning)
    near_zero_warn: float = 1e-8

    # Target spectral radius (warns if close to 1)
    spectral_radius_warn: float = 0.99
