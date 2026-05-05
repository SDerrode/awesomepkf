"""StabilityMatrix diagnostic — checks eigenvalues lie in [0, 1]."""

from __future__ import annotations

import warnings

import numpy as np

from prg.classes.matrix_diagnostics.base import _BaseMatrixDiagnostic
from prg.classes.matrix_diagnostics.results import CheckResult, DiagnosticReport
from prg.classes.matrix_diagnostics.status import Status
from prg.classes.matrix_diagnostics.tolerances import StabilityTolerances

__all__ = ["StabilityMatrix"]


class StabilityMatrix(_BaseMatrixDiagnostic):
    """
    Stability diagnostic for a square matrix.

    Checks that all eigenvalues (in modulus for complex eigenvalues)
    lie between 0 and 1 inclusive:

        ∀ λ ∈ spec(M) : 0 ≤ |λ| ≤ 1

    This corresponds to the (asymptotic) stability condition of a
    discrete dynamical system  x_{t+1} = M · x_t.

    Checks performed:
        1. NaN / Inf
        2. Shape (square matrix)
        3. Eigenvalues in [0, 1]  — FAIL if |λ| > 1 or |λ| < 0
        4. Spectral radius        — WARNING if ρ(M) ≥ near_unit_warn
        5. Near-zero eigenvalues  — WARNING if |λ_min| ≈ 0
        6. Imaginary part         — INFO if complex eigenvalues

    Parameters
    ----------
    matrix : np.ndarray
        Square matrix to analyse.
    tol : StabilityTolerances, optional
        Custom tolerance thresholds.

    Examples
    --------
    >>> M = np.array([[0.5, 0.1], [0.0, 0.8]])
    >>> stab = StabilityMatrix(M)
    >>> stab.summary()
    >>> stab.is_valid()        # True si aucun FAIL
    >>> stab.spectral_radius() # float — rayon spectral ρ(M)
    """

    def __init__(
        self,
        matrix: np.ndarray,
        tol: StabilityTolerances | None = None,
    ):
        super().__init__(matrix)
        self.tol = tol or StabilityTolerances()
        self._eigenvalues: np.ndarray | None = None  # cache

    # -------------------------------------------------------

    def spectral_radius(self) -> float:
        """Returns the spectral radius ρ(M) = max |λ_i|."""
        return float(np.max(np.abs(self._get_eigenvalues())))

    # -------------------------------------------------------

    def check(self) -> DiagnosticReport:
        report = DiagnosticReport(
            matrix_type="StabilityMatrix",
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

        # 3 — Eigenvalues in [0, 1]
        report.checks.append(self._check_eigenvalue_bounds())

        # 4 — Spectral radius (near-instability)
        report.checks.append(self._check_spectral_radius())

        # 5 — Near-zero eigenvalues
        report.checks.append(self._check_near_zero_eigenvalues())

        # 6 — Partie imaginaire (informatif)
        report.checks.append(self._check_complex_eigenvalues())

        return report

    # -------------------------------------------------------
    # Helpers internes
    # -------------------------------------------------------

    def _get_eigenvalues(self) -> np.ndarray:
        """Computes (and caches) the eigenvalues."""
        if self._eigenvalues is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._eigenvalues = np.linalg.eigvals(self._M)
        return self._eigenvalues

    # -------------------------------------------------------
    # Specific checks
    # -------------------------------------------------------

    def _check_eigenvalue_bounds(self) -> CheckResult:
        """Checks that all |λ_i| ∈ [0, 1]."""
        name = "Eigenvalues in [0, 1]"
        tol = self.tol
        eigvals = self._get_eigenvalues()
        moduli = np.abs(eigvals)

        # Violations: |λ| > 1 + tol_boundary → instability
        unstable_mask = moduli > tol.eigenvalue_max + tol.tol_boundary
        unstable_vals = eigvals[unstable_mask]

        if len(unstable_vals) > 0:
            worst = float(np.max(np.abs(unstable_vals)))
            details = ", ".join(
                (
                    f"{v.real:.4g}{'+' if v.imag >= 0 else ''}{v.imag:.4g}j"
                    if v.imag != 0
                    else f"{v.real:.4g}"
                )
                for v in unstable_vals
            )
            return self._fail(
                name,
                worst,
                tol.eigenvalue_max,
                f"{len(unstable_vals)} eigenvalue(s) with |λ| > 1 (unstable): [{details}]. "
                "The system is not stable.",
            )

        # Violations: |λ| < 0 → mathematically impossible,
        # but we check for safety the moduli outside lower bounds
        # (always ≥ 0 by definition, redundant but explicit check)
        max_mod = float(moduli.max())
        min_mod = float(moduli.min())
        return self._ok(
            name,
            max_mod,
            tol.eigenvalue_max,
            f"All {len(eigvals)} eigenvalue(s) have |λ| ∈ [{min_mod:.4g}, {max_mod:.4g}] ⊆ [0, 1].",
        )

    def _check_spectral_radius(self) -> CheckResult:
        """Warns if the spectral radius is close to 1 (near-instability)."""
        name = "Spectral radius ρ(M)"
        tol = self.tol
        rho = self.spectral_radius()

        if rho > tol.eigenvalue_max + tol.tol_boundary:
            # Already reported by _check_eigenvalue_bounds — raise consistent FAIL
            return self._fail(
                name,
                rho,
                tol.eigenvalue_max,
                f"ρ(M) = {rho:.6g} > 1 — system is unstable.",
            )
        if rho >= tol.spectral_radius_warn:
            return self._warn(
                name,
                rho,
                tol.spectral_radius_warn,
                f"ρ(M) = {rho:.6g} is close to 1 — system is marginally stable. "
                "Small perturbations could lead to instability.",
            )
        return self._ok(
            name,
            rho,
            tol.spectral_radius_warn,
            f"ρ(M) = {rho:.6g} — system has comfortable stability margin.",
        )

    def _check_near_zero_eigenvalues(self) -> CheckResult:
        """Warns if some |λ_i| are near-zero (very heavily damped mode)."""
        name = "Near-zero eigenvalues"
        tol = self.tol
        eigvals = self._get_eigenvalues()
        moduli = np.abs(eigvals)

        near_zero_mask = moduli < tol.near_zero_warn
        count = int(near_zero_mask.sum())

        if count > 0:
            vals_str = ", ".join(
                f"{moduli[i]:.2e}" for i in np.where(near_zero_mask)[0]
            )
            return self._warn(
                name,
                float(moduli[near_zero_mask].min()),
                tol.near_zero_warn,
                f"{count} eigenvalue(s) with |λ| ≈ 0 ({vals_str}). "
                "Corresponding modes decay extremely fast — may indicate numerical issues.",
            )
        return self._ok(
            name,
            float(moduli.min()),
            tol.near_zero_warn,
            "No near-zero eigenvalues detected.",
        )

    def _check_complex_eigenvalues(self) -> CheckResult:
        """Informs if any eigenvalues are complex (oscillatory modes)."""
        name = "Complex eigenvalues (oscillatory modes)"
        eigvals = self._get_eigenvalues()
        complex_mask = np.abs(eigvals.imag) > 1e-12
        count = int(complex_mask.sum())

        if count > 0:
            # Present conjugate pairs (approximate deduplication)
            complex_vals = eigvals[complex_mask]
            # Garde uniquement ceux avec imag > 0 (une valeur par paire)
            positive_imag = complex_vals[complex_vals.imag > 0]
            pairs_str = ", ".join(
                f"{v.real:.4g}±{abs(v.imag):.4g}j" for v in positive_imag
            )
            return self._warn(
                name,
                None,
                None,
                f"{count} complex eigenvalue(s) detected ({pairs_str}). "
                "The system exhibits oscillatory dynamics. "
                "Stability is determined by |λ|, not Re(λ) alone.",
            )
        return self._ok(
            name,
            None,
            None,
            "All eigenvalues are real — no oscillatory modes.",
        )
