#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
prg/exceptions.py
-----------------
Centralised exception hierarchy for the PKF project.

Tree
----
PKFError
├── ParamError
├── NumericalError
│   ├── CovarianceError
│   └── InvertibilityError
└── FilterError
    └── StepValidationError
"""

__all__ = [
    "PKFError",
    "ParamError",
    "NumericalError",
    "CovarianceError",
    "InvertibilityError",
    "FilterError",
    "StepValidationError",
]


# ---------------------------------------------------------------------------
# Mixin: shared step attribute
# ---------------------------------------------------------------------------


class _StepMixin:
    """
    Internal mixin providing the ``step`` attribute to an exception.

    Avoids logic duplication between ``NumericalError`` and
    ``StepValidationError``.  Not exported (``_`` prefix).
    """

    step: int  # declared here for static analysis tools

    def _step_repr(self) -> str:
        """Retourne la partie step du repr, factorisée."""
        return f"step={self.step}"


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------


class PKFError(Exception):
    """
    Root of all PKF project exceptions.

    All top-level ``except`` blocks can catch this class to
    intercept any project error in a single block.
    """

    def __repr__(self) -> str:
        # FIX: self.args can be empty if raised without a message → fallback to ""
        msg = self.args[0] if self.args else ""
        return f"{self.__class__.__name__}({msg!r})"


# ---------------------------------------------------------------------------
# Parameter errors
# ---------------------------------------------------------------------------


class ParamError(PKFError):
    """
    Invalid parameter supplied to a project class or method.

    Raised for example if ``sKey`` is not a strictly positive integer,
    or if ``verbose`` does not belong to ``{0, 1, 2}``.
    """


# ---------------------------------------------------------------------------
# Numerical errors
# ---------------------------------------------------------------------------


class NumericalError(_StepMixin, PKFError):
    """
    Generic numerical error.

    Base class for all errors related to matrix computations.
    Carries the structured context (step, matrix_name) shared by its
    subclasses, avoiding the need to parse text messages at the call site.

    Parameters
    ----------
    message : str
        Human-readable description of the error.
    matrix_name : str, optional
        Name of the matrix concerned (default ``""``).
    step : int, optional
        Time step index where the error occurred (default ``-1``).

    Attributes
    ----------
    matrix_name : str
        Name of the matrix concerned.
    step : int
        Time step index where the error occurred. ``-1`` if unknown.

    Examples
    --------

    >>> try:
    ...     raise CovarianceError("not PSD", matrix_name="PXX", step=42)
    ... except NumericalError as e:
    ...     print(e.step, e.matrix_name)
    42 PXX
    """

    def __init__(self, message: str, matrix_name: str = "", step: int = -1) -> None:
        super().__init__(message)
        self.matrix_name = matrix_name
        self.step = step

    def __str__(self) -> str:
        parts = [self.args[0] if self.args else ""]
        if self.step != -1:
            parts.append(f"step={self.step}")
        if self.matrix_name:
            parts.append(f"matrix={self.matrix_name!r}")
        return " | ".join(parts)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"step={self.step}, "
            f"matrix={self.matrix_name!r}, "
            f"msg={self.args[0]!r})"
        )


class CovarianceError(NumericalError):
    """
    Invalid covariance matrix.

    Raised when a matrix is not symmetric positive definite,
    or when the regularisation attempt has failed.
    """


class InvertibilityError(NumericalError):
    """
    Non-invertible matrix.

    Raised when a matrix expected to be invertible (e.g. ``Skp1``, ``Sigma22``)
    fails the invertibility diagnostic.
    """


# ---------------------------------------------------------------------------
# Filter errors
# ---------------------------------------------------------------------------


class FilterError(PKFError):
    """
    Generic PKF filter error.

    Base class for errors occurring during filter execution,
    independent of internal matrix computations.
    """


class StepValidationError(_StepMixin, FilterError):
    """
    Failure to construct a ``PKFStep``.

    Raised when data passed to the ``PKFStep`` constructor
    is invalid (wrong shapes, field inconsistencies, etc.).

    Parameters
    ----------
    message : str
        Human-readable description of the error.
    step : int, optional
        Time step index where the error occurred (default ``-1``).

    Attributes
    ----------
    step : int
        Time step index where the error occurred. ``-1`` if unknown.
    """

    def __init__(self, message: str, step: int = -1) -> None:
        super().__init__(message)
        self.step = step

    def __str__(self) -> str:
        parts = [self.args[0] if self.args else ""]
        if self.step != -1:
            parts.append(f"step={self.step}")
        return " | ".join(parts)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(" f"step={self.step}, " f"msg={self.args[0]!r})"
        )
