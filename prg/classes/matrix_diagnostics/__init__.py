"""
Matrix diagnostics package.

Three diagnostic classes are exposed:

- :class:`CovarianceMatrix` — checks symmetry, positive
  semi-definiteness, condition number; offers Tikhonov regularisation.
- :class:`InvertibleMatrix` — checks rank, determinant, condition
  number, post-inversion residual; caches the inverse.
- :class:`StabilityMatrix` — checks all eigenvalue moduli lie in
  [0, 1]; exposes :meth:`StabilityMatrix.spectral_radius`.

Each class returns a :class:`DiagnosticReport` aggregating
:class:`CheckResult` items with a :class:`Status` (OK / WARNING / FAIL).
"""

from prg.classes.matrix_diagnostics.covariance import (
    CovarianceMatrix,
    RegularizationResult,
)
from prg.classes.matrix_diagnostics.invertible import InvertibleMatrix
from prg.classes.matrix_diagnostics.results import CheckResult, DiagnosticReport
from prg.classes.matrix_diagnostics.stability import StabilityMatrix
from prg.classes.matrix_diagnostics.status import Status
from prg.classes.matrix_diagnostics.tolerances import (
    CovarianceTolerances,
    InvertibleTolerances,
    StabilityTolerances,
)

__all__ = [
    "CheckResult",
    "CovarianceMatrix",
    "CovarianceTolerances",
    "DiagnosticReport",
    "InvertibleMatrix",
    "InvertibleTolerances",
    "RegularizationResult",
    "StabilityMatrix",
    "StabilityTolerances",
    "Status",
]
