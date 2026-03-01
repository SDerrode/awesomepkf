#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
matrix_diagnostics.py
=====================
Diagnostic complet pour deux types de matrices :
    - CovarianceMatrix  : matrice de covariance (symétrique définie positive)
    - InvertibleMatrix  : matrice inversible quelconque

Chaque classe expose :
    - check()    → diagnostic complet, retourne un DiagnosticReport
    - summary()  → affiche le rapport en console

InvertibleMatrix expose en plus :
    - inverse()  → retourne l'inverse (après vérification)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np


# ==========================================================
# Statut
# ==========================================================


class Status(Enum):
    OK = auto()
    WARNING = auto()
    FAIL = auto()

    def __str__(self) -> str:
        return {
            Status.OK: "✅ OK",
            Status.WARNING: "⚠️  WARNING",
            Status.FAIL: "❌ FAIL",
        }[self]


# ==========================================================
# Résultat d'un test individuel
# ==========================================================


@dataclass
class CheckResult:
    name: str
    status: Status
    value: float | None
    threshold: float | None
    message: str

    def __str__(self) -> str:
        thr = f"  (threshold: {self.threshold})" if self.threshold is not None else ""
        val = f"  [value: {self.value:.6g}]" if self.value is not None else ""
        return f"  {str(self.status):<14}  {self.name}{val}{thr}\n    → {self.message}"


# ==========================================================
# Rapport complet
# ==========================================================


@dataclass
class DiagnosticReport:
    matrix_type: str
    shape: tuple
    dtype: str
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def overall_status(self) -> Status:
        if any(c.status == Status.FAIL for c in self.checks):
            return Status.FAIL
        if any(c.status == Status.WARNING for c in self.checks):
            return Status.WARNING
        return Status.OK

    @property
    def is_ok(self) -> bool:
        """True uniquement si tous les checks sont OK (aucun warning, aucun fail)."""
        return self.overall_status == Status.OK

    @property
    def is_valid(self) -> bool:
        """True si aucun check n'est FAIL (warnings tolérés)."""
        return self.overall_status != Status.FAIL

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            f"  Diagnostic — {self.matrix_type}",
            f"  Shape : {self.shape}    dtype : {self.dtype}",
            f"  Overall : {self.overall_status}",
            "-" * 60,
        ]
        for check in self.checks:
            lines.append(str(check))
        lines.append("=" * 60)
        return "\n".join(lines)


# ==========================================================
# Seuils de tolérance
# ==========================================================


@dataclass
class CovarianceTolerances:
    """Seuils pour une matrice de covariance."""

    # Symétrie  : max|M - M^T| / max|M|
    symmetry_warn: float = 1e-8
    symmetry_fail: float = 1e-5

    # Valeurs propres minimales
    eigenvalue_warn: float = 1e-10  # quasi-singulière
    eigenvalue_fail: float = 0.0  # valeur propre négative ou nulle

    # Numéro de condition
    condition_warn: float = 1e6
    condition_fail: float = 1e12

    # Diagonal (variance > 0)
    diagonal_fail: float = 0.0


@dataclass
class InvertibleTolerances:
    """Seuils pour une matrice inversible quelconque."""

    # Numéro de condition (rcond)
    condition_warn: float = 1e6
    condition_fail: float = 1e12

    # |det| minimum
    det_warn: float = 1e-10
    det_fail: float = 1e-15

    # Rang attendu (None = n)
    expected_rank: Optional[int] = None
    rank_tol: float = 1e-10

    # Résidu après inversion : ||I - M @ M_inv||_F
    residual_warn: float = 1e-8
    residual_fail: float = 1e-5


@dataclass
class StabilityTolerances:
    """Seuils pour une matrice de stabilité (valeurs propres dans [0, 1])."""

    # Valeur propre strictement négative
    eigenvalue_min: float = 0.0

    # Valeur propre strictement supérieure à 1
    eigenvalue_max: float = 1.0

    # Marge de tolérance numérique autour des bornes
    # (évite les faux positifs dus aux erreurs d'arrondi)
    tol_boundary: float = 1e-10

    # Valeur propre proche de 1 (quasi-instable, warning)
    near_unit_warn: float = 0.99

    # Valeur propre proche de 0 (quasi-nulle, warning)
    near_zero_warn: float = 1e-8

    # Rayon spectral cible (avertit si proche de 1)
    spectral_radius_warn: float = 0.99


# ==========================================================
# Classe de base
# ==========================================================


class _BaseMatrixDiagnostic:
    """Classe abstraite commune."""

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
        """True uniquement si tous les checks sont OK (aucun warning, aucun fail)."""
        return self.check().is_ok

    def is_valid(self) -> bool:
        """True si aucun check n'est FAIL (warnings tolérés)."""
        return self.check().is_valid

    def summary(self) -> None:
        print(self.check())


# ==========================================================
# Matrice de covariance
# ==========================================================


class CovarianceMatrix(_BaseMatrixDiagnostic):
    """
    Diagnostic pour une matrice de covariance.

    Vérifie : NaN/Inf, symétrie, diagonale positive,
              valeurs propres, numéro de condition.
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
            return report  # inutile de continuer

        # 2 — Shape
        report.checks.append(self._check_shape())

        # 3 — Symétrie
        report.checks.append(self._check_symmetry())

        # 4 — Diagonale positive
        report.checks.append(self._check_diagonal())

        # 5 — Valeurs propres
        report.checks.append(self._check_eigenvalues())

        # 6 — Numéro de condition
        report.checks.append(self._check_condition())

        return report

    # -------------------------------------------------------
    # Checks spécifiques
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

    def regularize(self, eps: float | None = None) -> "RegularizationResult":
        """
        Régularisation de Tikhonov : M_reg = (M + M.T)/2 + ε * I.

        Corrige une matrice dont les valeurs propres sont nulles ou
        légèrement négatives (dues à des erreurs numériques d'arrondi)
        en ajoutant une petite perturbation sur la diagonale.

        Si ``eps`` n'est pas fourni, il est calculé automatiquement :

            ε = |λ_min| + δ

        où ``δ = max(|λ_min|, n * eps_machine) * 10`` assure que la plus
        petite valeur propre de ``M_reg`` est strictement positive avec
        une marge raisonnable.

        Parameters
        ----------
        eps : float, optional
            Valeur de régularisation. Si ``None``, calculée automatiquement.

        Returns
        -------
        RegularizationResult
            Contient la matrice régularisée, la valeur ``eps`` utilisée,
            et les rapports de diagnostic avant/après.

        Raises
        ------
        ValueError
            Si ``eps`` est fourni mais non strictement positif.
        RuntimeError
            Si la matrice reste invalide après régularisation
            (indique un problème structural plus profond).

        Examples
        --------
        >>> result = CovarianceMatrix(M).regularize()
        >>> result.is_success           # True si la régularisation a réussi
        >>> result.matrix_regularized   # matrice corrigée
        >>> result.eps_applied          # ε effectivement ajouté
        """
        if eps is not None and eps <= 0:
            raise ValueError(f"eps must be strictly positive, got {eps}.")

        # Symétrise d'abord (bonne pratique avant eigvalsh)
        M_sym = (self._M + self._M.T) / 2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            eigvals = np.linalg.eigvalsh(M_sym)
        min_eig = float(eigvals.min())

        # Calcul automatique de ε si non fourni
        if eps is None:
            if min_eig > self.tol.eigenvalue_warn:
                # Matrice déjà saine — aucune régularisation nécessaire
                eps = 0.0
            else:
                # ε doit à la fois :
                #   1. rendre λ_min strictement positif
                #   2. abaisser le numéro de condition sous condition_fail
                #      en augmentant λ_min jusqu'à λ_max / condition_fail
                max_eig = float(eigvals.max())
                eps_for_eigenvalue = abs(min_eig) + self.tol.eigenvalue_warn * 10
                eps_for_condition = max_eig / self.tol.condition_fail
                eps = max(eps_for_eigenvalue, eps_for_condition) * 10

        M_reg = M_sym + eps * np.eye(self._n)

        # Diagnostic avant/après
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
                f"Regularization failed — matrix remains invalid after adding ε={eps:.4g}.\n"
                f"{report_after}"
            )

        return result

    def regularized(self, eps: float | None = None) -> np.ndarray:
        """
        Raccourci vers :meth:`regularize` — retourne directement la matrice corrigée.

        Parameters
        ----------
        eps : float, optional
            Valeur de régularisation. Si ``None``, calculée automatiquement.

        Returns
        -------
        np.ndarray
            Matrice régularisée ``(M + Mᵀ)/2 + ε * I``.
            Si la matrice est déjà saine (``is_ok``), retourne une copie
            symétrisée sans modification (ε = 0).
        """
        return self.regularize(eps=eps).matrix_regularized


# ==========================================================
# Résultat de régularisation
# ==========================================================


@dataclass
class RegularizationResult:
    """
    Résultat d'une régularisation de Tikhonov sur une matrice de covariance.

    Attributes
    ----------
    matrix_original : np.ndarray
        Matrice originale avant régularisation.
    matrix_regularized : np.ndarray
        Matrice après régularisation : ``(M + M.T)/2 + ε * I``.
    eps_applied : float
        Valeur de ε effectivement ajoutée sur la diagonale.
    min_eigenvalue_before : float
        Plus petite valeur propre avant régularisation.
    min_eigenvalue_after : float
        Plus petite valeur propre après régularisation.
    report_before : DiagnosticReport
        Rapport de diagnostic avant régularisation.
    report_after : DiagnosticReport
        Rapport de diagnostic après régularisation.
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
        """True si la matrice régularisée passe le diagnostic (is_valid)."""
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


# ==========================================================
# Matrice inversible
# ==========================================================


class InvertibleMatrix(_BaseMatrixDiagnostic):
    """
    Diagnostic pour une matrice inversible quelconque.

    Vérifie : NaN/Inf, rang, déterminant, numéro de condition,
              résidu post-inversion.
    Expose inverse() qui retourne l'inverse après vérification.
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

        # 3 — Rang
        report.checks.append(self._check_rank())

        # 4 — Déterminant
        report.checks.append(self._check_determinant())

        # 5 — Numéro de condition
        report.checks.append(self._check_condition())

        # 6 — Résidu (calcule et met en cache l'inverse)
        report.checks.append(self._check_residual())

        return report

    # -------------------------------------------------------

    def inverse(self) -> np.ndarray:
        """
        Retourne l'inverse de la matrice.
        Lance un diagnostic complet au préalable.
        Lève une RuntimeError si la matrice est FAIL.
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
    # Checks spécifiques
    # -------------------------------------------------------

    def _check_rank(self) -> CheckResult:
        name = "Rank"
        tol = self.tol
        rank = int(np.linalg.matrix_rank(self._M, tol=tol.rank_tol))
        expected = tol.expected_rank if tol.expected_rank is not None else self._n

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


# ==========================================================
# Matrice de stabilité
# ==========================================================


class StabilityMatrix(_BaseMatrixDiagnostic):
    """
    Diagnostic de stabilité pour une matrice carrée.

    Vérifie que toutes les valeurs propres (en module pour les valeurs
    propres complexes) sont comprises entre 0 et 1 inclus :

        ∀ λ ∈ spec(M) : 0 ≤ |λ| ≤ 1

    Cela correspond à la condition de stabilité (asymptotique) d'un
    système dynamique discret  x_{t+1} = M · x_t.

    Checks effectués :
        1. NaN / Inf
        2. Shape (matrice carrée)
        3. Valeurs propres dans [0, 1]  — FAIL si |λ| > 1 ou |λ| < 0
        4. Rayon spectral               — WARNING si ρ(M) ≥ near_unit_warn
        5. Valeurs propres quasi-nulles — WARNING si |λ_min| ≈ 0
        6. Partie imaginaire            — INFO si valeurs propres complexes

    Parameters
    ----------
    matrix : np.ndarray
        Matrice carrée à analyser.
    tol : StabilityTolerances, optional
        Seuils de tolérance personnalisés.

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
        """Retourne le rayon spectral ρ(M) = max |λ_i|."""
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

        # 3 — Valeurs propres dans [0, 1]
        report.checks.append(self._check_eigenvalue_bounds())

        # 4 — Rayon spectral (quasi-instabilité)
        report.checks.append(self._check_spectral_radius())

        # 5 — Valeurs propres quasi-nulles
        report.checks.append(self._check_near_zero_eigenvalues())

        # 6 — Partie imaginaire (informatif)
        report.checks.append(self._check_complex_eigenvalues())

        return report

    # -------------------------------------------------------
    # Helpers internes
    # -------------------------------------------------------

    def _get_eigenvalues(self) -> np.ndarray:
        """Calcule (et met en cache) les valeurs propres."""
        if self._eigenvalues is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._eigenvalues = np.linalg.eigvals(self._M)
        return self._eigenvalues

    # -------------------------------------------------------
    # Checks spécifiques
    # -------------------------------------------------------

    def _check_eigenvalue_bounds(self) -> CheckResult:
        """Vérifie que tous les |λ_i| ∈ [0, 1]."""
        name = "Eigenvalues in [0, 1]"
        tol = self.tol
        eigvals = self._get_eigenvalues()
        moduli = np.abs(eigvals)

        # Violations : |λ| > 1 + tol_boundary  →  instabilité
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

        # Violations : |λ| < 0  →  impossible mathématiquement,
        # mais on vérifie par sécurité les moduli hors bornes basses
        # (ici toujours ≥ 0 par définition, vérification redondante mais explicite)
        max_mod = float(moduli.max())
        min_mod = float(moduli.min())
        return self._ok(
            name,
            max_mod,
            tol.eigenvalue_max,
            f"All {len(eigvals)} eigenvalue(s) have |λ| ∈ [{min_mod:.4g}, {max_mod:.4g}] ⊆ [0, 1].",
        )

    def _check_spectral_radius(self) -> CheckResult:
        """Avertit si le rayon spectral est proche de 1 (quasi-instabilité)."""
        name = "Spectral radius ρ(M)"
        tol = self.tol
        rho = self.spectral_radius()

        if rho > tol.eigenvalue_max + tol.tol_boundary:
            # Déjà signalé par _check_eigenvalue_bounds — on remonte FAIL cohérent
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
        """Avertit si certains |λ_i| sont quasi-nuls (mode très amorti)."""
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
        """Informe si des valeurs propres sont complexes (oscillations)."""
        name = "Complex eigenvalues (oscillatory modes)"
        eigvals = self._get_eigenvalues()
        complex_mask = np.abs(eigvals.imag) > 1e-12
        count = int(complex_mask.sum())

        if count > 0:
            # Présenter les paires conjuguées (dédupliquer approximativement)
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


# ==========================================================
# Demo
# ==========================================================

if __name__ == "__main__":

    print("\n--- CovarianceMatrix : cas sain ---")
    A = np.array([[4.0, 2.0], [2.0, 3.0]])
    cov_a = CovarianceMatrix(A)
    cov_a.summary()
    print(f"  is_ok    = {cov_a.is_ok()}")
    print(f"  is_valid = {cov_a.is_valid()}")

    print("\n--- CovarianceMatrix : quasi-singulière ---")
    B = np.array([[1.0, 1.0], [1.0, 1.0 + 1e-11]])
    cov_b = CovarianceMatrix(B)
    cov_b.summary()
    print(f"  is_ok    = {cov_b.is_ok()}")
    print(f"  is_valid = {cov_b.is_valid()}")

    print("\n--- CovarianceMatrix : non-symétrique ---")
    C = np.array([[4.0, 1.0], [3.0, 3.0]])
    cov_c = CovarianceMatrix(C)
    cov_c.summary()
    print(f"  is_ok    = {cov_c.is_ok()}")
    print(f"  is_valid = {cov_c.is_valid()}")

    print(
        "\n--- CovarianceMatrix : valeur propre nulle → régularisation automatique ---"
    )
    # Matrice avec λ_min = 0 (singulière mais symétrique)
    v = np.array([[1.0], [1.0]])
    D_sing = v @ v.T  # [[1, 1], [1, 1]] — rang 1, λ_min = 0
    cov_d = CovarianceMatrix(D_sing)
    cov_d.summary()
    print(f"  is_valid avant = {cov_d.is_valid()}")
    result = cov_d.regularize()
    print(result)
    print(f"  is_success     = {result.is_success}")
    print(f"  ε appliqué     = {result.eps_applied:.6g}")
    print(f"  Matrice régularisée :\n{result.matrix_regularized}")

    print(
        "\n--- CovarianceMatrix : valeur propre nulle → régularisation manuelle (ε=1e-6) ---"
    )
    result_manual = cov_d.regularize(eps=1e-6)
    print(result_manual)

    print("\n--- InvertibleMatrix : cas sain ---")
    E = np.array([[3.0, 1.0], [1.0, 2.0]])
    inv_e = InvertibleMatrix(E)
    inv_e.summary()
    print(f"  is_ok    = {inv_e.is_ok()}")
    print(f"  is_valid = {inv_e.is_valid()}")
    print("  Inverse :\n", inv_e.inverse())

    print("\n--- InvertibleMatrix : singulière ---")
    F = np.array([[1.0, 2.0], [2.0, 4.0]])
    inv_f = InvertibleMatrix(F)
    print(f"  is_ok    = {inv_f.is_ok()}")
    print(f"  is_valid = {inv_f.is_valid()}")
    try:
        inv_f.inverse()
    except RuntimeError as e:
        print(e)
