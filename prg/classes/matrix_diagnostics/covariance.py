"""CovarianceMatrix diagnostic + Tikhonov regularisation."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from prg.classes.matrix_diagnostics.base import _BaseMatrixDiagnostic
from prg.classes.matrix_diagnostics.results import CheckResult, DiagnosticReport
from prg.classes.matrix_diagnostics.status import Status
from prg.classes.matrix_diagnostics.tolerances import CovarianceTolerances

__all__ = ["CovarianceMatrix", "RegularizationResult"]


class CovarianceMatrix(_BaseMatrixDiagnostic):
    """
    Diagnostic for a covariance matrix.

    Checks: NaN/Inf, symmetry, positive diagonal,
            eigenvalues, condition number.
    """

    def __init__(
        self,
        matrix: np.ndarray,
        tol: CovarianceTolerances | None = None,
    ):
        super().__init__(matrix)
        self.tol = tol or CovarianceTolerances()

    # -------------------------------------------------------

    def check(self) -> DiagnosticReport:
        report = DiagnosticReport(
            matrix_type="CovarianceMatrix",
            shape=self._M.shape,
            dtype=str(self._M.dtype),
        )

        # 1 — NaN / Inf
        c = self._check_nan_inf()
        report.checks.append(c)
        if c.status == Status.FAIL:
            return report  # no point in continuing

        # 2 — Shape
        report.checks.append(self._check_shape())

        # 3 — Symmetry
        report.checks.append(self._check_symmetry())

        # 4 — Diagonale positive
        report.checks.append(self._check_diagonal())

        # 5 — Eigenvalues
        report.checks.append(self._check_eigenvalues())

        # 6 — Condition number
        report.checks.append(self._check_condition())

        return report

    # -------------------------------------------------------
    # Specific checks
    # -------------------------------------------------------

    def _check_symmetry(self) -> CheckResult:
        name = "Symmetry"
        tol = self.tol
        norm_M = np.max(np.abs(self._M))
        if norm_M == 0:
            return self._ok(name, 0.0, None, "Zero matrix — trivially symmetric.")

        asymmetry = np.max(np.abs(self._M - self._M.T)) / norm_M

        if asymmetry > tol.symmetry_fail:
            return self._fail(
                name,
                asymmetry,
                tol.symmetry_fail,
                "Matrix is not symmetric. Covariance matrices must be symmetric.",
            )
        if asymmetry > tol.symmetry_warn:
            return self._warn(
                name,
                asymmetry,
                tol.symmetry_warn,
                "Minor asymmetry detected — consider symmetrizing: M = (M + M.T) / 2.",
            )
        return self._ok(name, asymmetry, tol.symmetry_warn, "Matrix is symmetric.")

    def _check_diagonal(self) -> CheckResult:
        name = "Diagonal (variances > 0)"
        diag = np.diag(self._M)
        bad = np.where(diag <= self.tol.diagonal_fail)[0]
        if len(bad) > 0:
            return self._fail(
                name,
                float(diag.min()),
                self.tol.diagonal_fail,
                f"Non-positive diagonal entries at indices: {bad.tolist()}. "
                "Variances must be strictly positive.",
            )
        return self._ok(
            name, float(diag.min()), None, "All diagonal entries are positive."
        )

    def _check_eigenvalues(self) -> CheckResult:
        name = "Eigenvalues (positive semi-definite)"
        tol = self.tol
        # Force symmetry for eigvalsh
        M_sym = (self._M + self._M.T) / 2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            eigvals = np.linalg.eigvalsh(M_sym)

        min_eig = float(eigvals.min())

        if min_eig <= tol.eigenvalue_fail:
            return self._fail(
                name,
                min_eig,
                tol.eigenvalue_fail,
                f"Negative or zero eigenvalue ({min_eig:.4g}). "
                "Matrix is not positive semi-definite.",
            )
        if min_eig < tol.eigenvalue_warn:
            return self._warn(
                name,
                min_eig,
                tol.eigenvalue_warn,
                f"Near-zero eigenvalue ({min_eig:.4g}). "
                "Matrix may be numerically singular.",
            )
        return self._ok(
            name,
            min_eig,
            tol.eigenvalue_warn,
            f"All eigenvalues positive (min = {min_eig:.4g}).",
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
                f"Condition number {cond:.4g} is large — numerical precision may be affected.",
            )
        return self._ok(
            name,
            cond,
            tol.condition_warn,
            f"Condition number {cond:.4g} is acceptable.",
        )

    # -------------------------------------------------------
    # Regularisation
    # -------------------------------------------------------

    def regularize(self, eps: float | None = None) -> RegularizationResult:
        """
        Tikhonov regularisation: M_reg = (M + M.T)/2 + ε * I.

        Corrects an invalid covariance matrix by adding a diagonal perturbation
        ε · I. Two failure cases are handled:

        1. **Zero/negative eigenvalues**: ε is chosen to make λ_min
        strictly positive with a reasonable margin.
        2. **Poor conditioning** (cond ≥ condition_fail): ε is chosen to
        bring the condition number below the ``condition_fail`` threshold.

        If the matrix is already healthy (λ_min > eigenvalue_warn **and**
        cond < condition_fail), no perturbation is applied (ε = 0).

        If ``eps`` is provided explicitly, it is used directly without
        automatic computation.

        Parameters
        ----------
        eps : float, optional
            Regularisation value. If ``None``, computed automatically.
            If provided, must be strictly positive.

        Returns
        -------
        RegularizationResult
            Contains the regularised matrix, the ``eps`` value used,
            and the before/after diagnostic reports.

        Raises
        ------
        ValueError
            If ``eps`` is provided but not strictly positive.
        RuntimeError
            If the matrix contains NaN/Inf (not regularisable), or if it
            remains invalid after regularisation (deep structural problem).
        """
        if eps is not None and eps <= 0:
            raise ValueError(f"eps must be strictly positive, got {eps}.")

        # --- Guard: NaN/Inf → not regularisable ---
        if not np.all(np.isfinite(self._M)):
            raise RuntimeError(
                "Regularization failed ― matrix remains invalid after adding ε=nan.\n"
                f"{self.check()}"
            )

        # Symmetrise before eigvalsh (good numerical practice)
        M_sym = (self._M + self._M.T) / 2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            eigvals = np.linalg.eigvalsh(M_sym)
        min_eig = float(eigvals.min())
        max_eig = float(eigvals.max())

        # --- Automatic computation of ε ---
        if eps is None:
            # Evaluates both failure criteria
            eigenvalue_ok = min_eig > self.tol.eigenvalue_warn
            # Avoids cond(NaN) if max_eig ~ 0; also protects division
            if max_eig > 0.0 and min_eig > 0.0:
                cond = max_eig / min_eig  # equivalent to np.linalg.cond for SPD
            else:
                cond = np.inf
            condition_ok = cond < self.tol.condition_fail

            if eigenvalue_ok and condition_ok:
                # Matrix already healthy — no regularisation needed
                eps = 0.0
            else:
                # ε must simultaneously:
                #   1. make λ_min strictly positive (if needed)
                #   2. lower the conditioning below condition_fail
                #      by raising λ_min up to λ_max / condition_fail
                eps_for_eigenvalue = (
                    abs(min_eig) + self.tol.eigenvalue_warn * 10
                    if not eigenvalue_ok
                    else 0.0
                )
                eps_for_condition = (
                    max_eig / self.tol.condition_fail - min_eig
                    if not condition_ok
                    else 0.0
                )
                # Facteur ×10 pour garantir une marge confortable
                eps = max(eps_for_eigenvalue, eps_for_condition) * 10

        M_reg = M_sym + eps * np.eye(self._n)

        # --- Before / after diagnostics ---
        report_before = self.check()
        report_after = CovarianceMatrix(M_reg, tol=self.tol).check()

        result = RegularizationResult(
            matrix_original=self._M.copy(),
            matrix_regularized=M_reg,
            eps_applied=eps,
            min_eigenvalue_before=min_eig,
            min_eigenvalue_after=float(np.linalg.eigvalsh(M_reg).min()),
            report_before=report_before,
            report_after=report_after,
        )

        if not result.is_success:
            raise RuntimeError(
                f"Regularization failed ― matrix remains invalid after adding ε={eps:.4g}.\n"
                f"{report_after}"
            )

        return result

    def regularized(self, eps: float | None = None) -> np.ndarray:
        """
        Shortcut to :meth:`regularize` — returns the corrected matrix directly.

        Parameters
        ----------
        eps : float, optional
            Regularisation value. If ``None``, computed automatically.

        Returns
        -------
        np.ndarray
            Regularised matrix ``(M + Mᵀ)/2 + ε * I``.
            If the matrix is already healthy (``is_ok``), returns a
            symmetrised copy without modification (ε = 0).
        """
        return self.regularize(eps=eps).matrix_regularized


@dataclass
class RegularizationResult:
    """
    Regularisation result of a Tikhonov regularisation on a covariance matrix.

    Attributes
    ----------
    matrix_original : np.ndarray
        Original matrix before regularisation.
    matrix_regularized : np.ndarray
        Matrix after regularisation: ``(M + M.T)/2 + ε * I``.
    eps_applied : float
        Value of ε actually added to the diagonal.
    min_eigenvalue_before : float
        Smallest eigenvalue before regularisation.
    min_eigenvalue_after : float
        Smallest eigenvalue after regularisation.
    report_before : DiagnosticReport
        Diagnostic report before regularisation.
    report_after : DiagnosticReport
        Diagnostic report after regularisation.
    """

    matrix_original: np.ndarray
    matrix_regularized: np.ndarray
    eps_applied: float
    min_eigenvalue_before: float
    min_eigenvalue_after: float
    report_before: DiagnosticReport
    report_after: DiagnosticReport

    @property
    def is_success(self) -> bool:
        """True if the regularised matrix passes the diagnostic (is_valid)."""
        return self.report_after.is_valid

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "  Regularization Report (Tikhonov  M_reg = (M+Mᵀ)/2 + ε·I)",
            "-" * 60,
            f"  ε applied          : {self.eps_applied:.6g}",
            f"  λ_min before       : {self.min_eigenvalue_before:.6g}",
            f"  λ_min after        : {self.min_eigenvalue_after:.6g}",
            f"  Success            : {'✅ Yes' if self.is_success else '❌ No'}",
            "=" * 60,
        ]
        return "\n".join(lines)
