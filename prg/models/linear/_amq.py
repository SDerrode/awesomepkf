"""LinearAmQ — linear model parametrised by (A, mQ)."""

import numpy as np

from prg.classes.matrix_diagnostics import CovarianceMatrix
from prg.classes.SeedGenerator import SeedGenerator
from prg.models.linear._base import BaseModelLinear
from prg.utils.exceptions import NumericalError
from prg.utils.generate_matrix_cov import generate_block_matrix

__all__ = ["LinearAmQ"]


class LinearAmQ(BaseModelLinear):
    """
    Linear model with transition matrix A and covariance Q.
    B defaults to the identity if not provided.
    """

    def __init__(
        self, dim_x, dim_y, A, mQ, mz0, Pz0, B=None, augmented=False, pairwiseModel=True
    ):
        super().__init__(
            dim_x,
            dim_y,
            model_type="linear_AmQ",
            augmented=augmented,
            pairwiseModel=pairwiseModel,
        )

        try:
            self.A = A
            self.B = B if B is not None else np.eye(A.shape[0])
            self.mQ = mQ
            self.mz0 = mz0
            self.Pz0 = Pz0
        except Exception as e:
            raise NumericalError(
                f"[{self.__class__.__name__}] LinearAmQ: parameter assignment error: {e}"
            ) from e

        if __debug__ and not self.augmented:
            for arr in [self.mQ, self.Pz0, self.B @ self.B.transpose()]:
                report = CovarianceMatrix(arr).check()
                if not report.is_valid:
                    raise ValueError("Matrix is not positive semi-definite.")

        self._build_symbolic_model()

    @staticmethod
    def _init_random_params(dim_x, dim_y, val_max, seed=None):
        """Generates mQ, mz0, Pz0 in a standard way via SeedGenerator."""
        seed = 9
        rng = SeedGenerator(seed).rng
        try:
            mQ = generate_block_matrix(rng, dim_x, dim_y, val_max)
            mz0 = rng.standard_normal((dim_x + dim_y, 1))
            Pz0 = generate_block_matrix(rng, dim_x, dim_y, val_max)
        except (ValueError, np.exceptions.AxisError) as e:
            raise NumericalError(f"_init_random_params failed: {e}") from e
        return mQ, mz0, Pz0

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
            "A": self.A,
            "B": self.B,
            "mQ": self.mQ,
            "mz0": self.mz0,
            "Pz0": self.Pz0,
            "alpha": self.alpha,
            "beta": self.beta,
            "kappa": self.kappa,
            "lambda_": self.lambda_,
        }

    def classic2pairwise(self, mod):

        try:
            dim_x = mod.dim_x
            dim_y = mod.dim_y
            dim_xy = mod.dim_xy

            F = mod.A
            C = mod.B

            H = np.zeros((dim_y, dim_xy))
            H[:, dim_x:] = np.eye(dim_y)
            D = np.zeros((dim_y, dim_y))

        except (ValueError, IndexError, np.exceptions.AxisError) as e:
            raise NumericalError(
                f"[{mod.__class__.__name__}] Initialization failed: {e}"
            ) from e

        A = np.block(
            [
                [F, np.zeros((dim_xy, dim_y))],
                [H @ F, np.zeros((dim_y, dim_y))],
            ]
        )
        B = np.block(
            [
                [C, np.zeros((dim_xy, dim_y))],
                [H @ C, D],
            ]
        )

        mQ = np.zeros((dim_xy + dim_y, dim_xy + dim_y))
        mQ[0:dim_xy, 0:dim_xy] = mod.mQ

        mz0 = np.zeros((dim_xy + dim_y, 1))
        mz0[0:dim_xy] = mod.mz0
        mz0[dim_xy : dim_xy + dim_y] = mz0[dim_xy - dim_y : dim_xy]

        Pz0 = np.zeros((dim_xy + dim_y, dim_xy + dim_y))
        Pz0[0:dim_xy, 0:dim_xy] = mod.Pz0
        Pz0[dim_xy : dim_xy + dim_y, :] = Pz0[dim_xy - dim_y : dim_xy, :]
        Pz0[:, dim_xy : dim_xy + dim_y] = Pz0[:, dim_xy - dim_y : dim_xy]

        return (dim_xy, dim_y, A, mQ, mz0, Pz0, B)
