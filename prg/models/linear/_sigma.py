"""LinearSigma — linear model parametrised by (sxx, syy, a, b, c, d, e)."""

import numpy as np
from scipy.linalg import LinAlgError, cho_factor, cho_solve

from prg.classes.matrix_diagnostics import CovarianceMatrix, StabilityMatrix
from prg.models.linear._base import BaseModelLinear
from prg.utils.exceptions import NumericalError

__all__ = ["LinearSigma"]


class LinearSigma(BaseModelLinear):
    """
    Linear model with variances sxx, syy and coefficients a, b, c, d, e.
    """

    def __init__(
        self, dim_x, dim_y, sxx, syy, a, b, c, d, e, augmented=False, pairwiseModel=True
    ):
        super().__init__(
            dim_x,
            dim_y,
            model_type="linear_Sigma",
            augmented=augmented,
            pairwiseModel=pairwiseModel,
        )

        self.sxx = sxx
        self.syy = syy
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e

        self._initSigma()

    def _initSigma(self):
        try:
            Q1 = np.block([[self.sxx, self.b.T], [self.b, self.syy]])
            Q2 = np.block([[self.a, self.e], [self.d, self.c]])
        except (ValueError, np.exceptions.AxisError) as e:
            raise NumericalError(
                f"[{self.__class__.__name__}] _initSigma: block matrix construction error: {e}"
            ) from e

        try:
            c_factor, lower = cho_factor(Q1)
            self.A = Q2 @ cho_solve((c_factor, lower), np.eye(self.dim_xy))
        except LinAlgError as e:
            raise NumericalError(
                f"[{self.__class__.__name__}] _initSigma: Cholesky decomposition failed"
                f" (Q1 is not positive definite): {e}"
            ) from e
        except (ValueError, np.exceptions.AxisError) as e:
            raise NumericalError(
                f"[{self.__class__.__name__}] _initSigma: matrix solve error: {e}"
            ) from e

        stab = StabilityMatrix(self.A)
        if not stab.is_valid():
            stab.summary()
            raise ValueError(
                f"[{self.__class__.__name__}] _initSigma: "
                f"the computed matrix A is not stable (spectral radius >= 1). "
                f"Check parameters sxx, syy, a, b, c, d, e."
            )

        self.B = np.eye(self.A.shape[0])

        if __debug__:
            for arr in [self.sxx, self.syy, Q1]:
                report = CovarianceMatrix(arr).check()
                if not report.is_valid:
                    raise ValueError("Matrix is not positive semi-definite.")

        self._build_symbolic_model()

    def get_params(self):
        return {
            "dim_x": self.dim_x,
            "dim_y": self.dim_y,
            "augmented": self.augmented,
            "pairwiseModel": self.pairwiseModel,
            "g": self.g,
            "f": getattr(self, "_fx", None),
            "h": getattr(self, "_hx", None),
            "jacobiens_g": self.jacobiens_g,
            "sxx": self.sxx,
            "syy": self.syy,
            "a": self.a,
            "b": self.b,
            "c": self.c,
            "d": self.d,
            "e": self.e,
            "alpha": self.alpha,
            "beta": self.beta,
            "kappa": self.kappa,
            "lambda_": self.lambda_,
        }
