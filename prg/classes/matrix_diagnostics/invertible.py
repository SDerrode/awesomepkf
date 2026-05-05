"""InvertibleMatrix diagnostic + cached inverse."""

from __future__ import annotations

import warnings

import numpy as np

from prg.classes.matrix_diagnostics.base import _BaseMatrixDiagnostic
from prg.classes.matrix_diagnostics.results import CheckResult, DiagnosticReport
from prg.classes.matrix_diagnostics.status import Status
from prg.classes.matrix_diagnostics.tolerances import InvertibleTolerances

__all__ = ["InvertibleMatrix"]


class InvertibleMatrix(_BaseMatrixDiagnostic):
    """
    Diagnostic for an arbitrary invertible matrix.

    Checks: NaN/Inf, rank, determinant, condition number,
            post-inversion residual.
    Exposes inverse() which returns the inverse after verification.
    """

    def __init__(
        self,
        matrix: np.ndarray,
        tol: InvertibleTolerances | None = None,
    ):
        super().__init__(matrix)
        self.tol = tol or InvertibleTolerances()
        self._inverse_cache: np.ndarray | None = None

    # -------------------------------------------------------

    def check(self) -> DiagnosticReport:
        report = DiagnosticReport(
            matrix_type="InvertibleMatrix",
            shape=self._M.shape,
            dtype=str(self._M.dtype),
        )

        # 1 — NaN / Inf
        c = self._check_nan_inf()
        report.checks.append(c)
        if c.status == Status.FAIL:
            return report

        # 2 — Shape
        report.checks.append(self._check_shape())

        # 3 — Rank
        report.checks.append(self._check_rank())

        # 4 — Determinant
        report.checks.append(self._check_determinant())

        # 5 — Condition number
        report.checks.append(self._check_condition())

        # 6 — Residual (computes and caches the inverse)
        report.checks.append(self._check_residual())

        return report

    # -------------------------------------------------------

    def inverse(self) -> np.ndarray:
        """
        Returns the inverse of the matrix.
        Runs a complete diagnostic beforehand.
        Raises RuntimeError if the matrix is FAIL.
        """
        report = self.check()

        if report.overall_status == Status.FAIL:
            raise RuntimeError(f"Cannot invert matrix — diagnostic FAILED:\n{report}")

        if report.overall_status == Status.WARNING:
            warnings.warn(
                "Inverting a matrix with WARNING status — "
                "numerical accuracy may be reduced.",
                stacklevel=2,
            )

        if self._inverse_cache is None:
            self._inverse_cache = np.linalg.inv(self._M)

        return self._inverse_cache

    # -------------------------------------------------------
    # Specific checks
    # -------------------------------------------------------

    def _check_rank(self) -> CheckResult:
        name = "Rank"
        tol = self.tol
        expected = tol.expected_rank if tol.expected_rank is not None else self._n

        # Relative threshold: avoids false positives on matrices with small values
        # np.linalg.matrix_rank without explicit tol already uses a relative threshold
        # (max(M.shape) * eps_machine * sigma_max) — this is the correct behaviour.
        if tol.rank_tol is not None:
            # Convert the absolute threshold to a threshold relative to the spectral norm
            sigma_max = float(np.linalg.norm(self._M, ord=2))
            adaptive_tol = max(tol.rank_tol, sigma_max * self._n * np.finfo(float).eps)
        else:
            adaptive_tol = None  # let numpy choose (default relative threshold)

        rank = int(np.linalg.matrix_rank(self._M, tol=adaptive_tol))

        if rank < expected:
            return self._fail(
                name,
                float(rank),
                float(expected),
                f"Rank {rank} < {expected}. Matrix is rank-deficient and not invertible.",
            )
        return self._ok(
            name,
            float(rank),
            float(expected),
            f"Full rank ({rank} = {expected}).",
        )

    def _check_determinant(self) -> CheckResult:
        name = "Determinant |det|"
        tol = self.tol
        det = float(np.linalg.det(self._M))
        abs_det = abs(det)

        if abs_det <= tol.det_fail:
            return self._fail(
                name,
                abs_det,
                tol.det_fail,
                f"|det| = {abs_det:.4g} ≈ 0 — matrix is singular.",
            )
        if abs_det <= tol.det_warn:
            return self._warn(
                name,
                abs_det,
                tol.det_warn,
                f"|det| = {abs_det:.4g} is very small — near-singular.",
            )
        return self._ok(
            name,
            abs_det,
            tol.det_warn,
            f"|det| = {abs_det:.4g}.",
        )

    def _check_condition(self) -> CheckResult:
        name = "Condition number"
        tol = self.tol
        cond = float(np.linalg.cond(self._M))

        if cond >= tol.condition_fail:
            return self._fail(
                name,
                cond,
                tol.condition_fail,
                f"Condition number {cond:.4g} is extremely large — matrix is ill-conditioned.",
            )
        if cond >= tol.condition_warn:
            return self._warn(
                name,
                cond,
                tol.condition_warn,
                f"Condition number {cond:.4g} is large — inversion may lose precision.",
            )
        return self._ok(
            name,
            cond,
            tol.condition_warn,
            f"Condition number {cond:.4g} is acceptable.",
        )

    def _check_residual(self) -> CheckResult:
        name = "Inversion residual ||I - M @ M⁻¹||_F"
        tol = self.tol

        try:
            M_inv = np.linalg.inv(self._M)
            self._inverse_cache = M_inv
            residual = float(np.linalg.norm(np.eye(self._n) - self._M @ M_inv, "fro"))
        except np.linalg.LinAlgError:
            self._inverse_cache = None
            return self._fail(
                name,
                None,
                None,
                "np.linalg.inv raised LinAlgError — matrix is singular.",
            )

        if residual >= tol.residual_fail:
            return self._fail(
                name,
                residual,
                tol.residual_fail,
                f"Residual {residual:.4g} is too large — inversion is numerically unreliable.",
            )
        if residual >= tol.residual_warn:
            return self._warn(
                name,
                residual,
                tol.residual_warn,
                f"Residual {residual:.4g} is non-negligible.",
            )
        return self._ok(
            name,
            residual,
            tol.residual_warn,
            f"Residual {residual:.4g} — inversion is numerically accurate.",
        )
