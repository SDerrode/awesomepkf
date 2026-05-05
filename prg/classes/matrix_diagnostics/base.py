"""Common abstract base for the diagnostic classes."""

import numpy as np

from prg.classes.matrix_diagnostics.results import CheckResult, DiagnosticReport
from prg.classes.matrix_diagnostics.status import Status

__all__ = ["_BaseMatrixDiagnostic"]


class _BaseMatrixDiagnostic:
    """Common abstract base class."""

    def __init__(self, matrix: np.ndarray):
        M = np.asarray(matrix, dtype=float)
        if M.ndim != 2 or M.shape[0] != M.shape[1]:
            raise ValueError(f"Expected a square 2-D array, got shape {M.shape}.")
        self._M = M
        self._n = M.shape[0]

    # --- helpers ---

    @staticmethod
    def _ok(
        name: str, value: float | None, threshold: float | None, msg: str
    ) -> CheckResult:
        return CheckResult(name, Status.OK, value, threshold, msg)

    @staticmethod
    def _warn(
        name: str, value: float | None, threshold: float | None, msg: str
    ) -> CheckResult:
        return CheckResult(name, Status.WARNING, value, threshold, msg)

    @staticmethod
    def _fail(
        name: str, value: float | None, threshold: float | None, msg: str
    ) -> CheckResult:
        return CheckResult(name, Status.FAIL, value, threshold, msg)

    def _check_nan_inf(self) -> CheckResult:
        name = "NaN / Inf"
        if not np.all(np.isfinite(self._M)):
            return self._fail(name, None, None, "Matrix contains NaN or Inf values.")
        return self._ok(name, None, None, "All entries are finite.")

    def _check_shape(self) -> CheckResult:
        return self._ok(
            "Shape",
            None,
            None,
            f"Square matrix of size {self._n}×{self._n}.",
        )

    # --- public ---

    def check(self) -> DiagnosticReport:
        raise NotImplementedError

    def is_ok(self) -> bool:
        """True only if all checks are OK (no warnings, no failures)."""
        return self.check().is_ok

    def is_valid(self) -> bool:
        """True if no check is FAIL (warnings tolerated)."""
        return self.check().is_valid

    def summary(self) -> None:
        print(self.check())
