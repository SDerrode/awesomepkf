import numpy as np
from prg.models.nonLinear.base_model_nonLinear import BaseModelNonLinear
from prg.models.Generate_MatrixCov import generate_block_matrix

__all__ = ["ModelCubique"]


class NumericalError(RuntimeError):
    """Raised when a numerical computation fails in a model."""

    pass


class ModelCubique(BaseModelNonLinear):

    MODEL_NAME: str = "x1_y1_cubique"

    def __init__(self):
        super().__init__(dim_x=1, dim_y=1, model_type="nonlinear")

        try:
            self.mQ = generate_block_matrix(
                self._randMatrices.rng, self.dim_x, self.dim_y, 0.30
            )
            self.mz0 = self._randMatrices.rng.standard_normal((self.dim_xy, 1))
            self.Pz0 = generate_block_matrix(
                self._randMatrices.rng, self.dim_x, self.dim_y, 0.30
            )
        except (ValueError, np.exceptions.AxisError) as e:
            raise NumericalError(
                f"[{self.MODEL_NAME}] Initialization failed: {e}"
            ) from e

    # ------------------------------------------------------------------
    def _fx(self, x, t, dt):
        if __debug__:
            if x.ndim == 2:
                assert all(a.shape == (1, 1) for a in (x, t))
            else:
                assert all(a.ndim == 3 and a.shape[1:] == (1, 1) for a in (x, t))
                assert x.shape[0] == t.shape[0]

        try:
            with np.errstate(all="raise"):
                return 0.9 * x - 0.6 * x**3 + t
        except FloatingPointError as e:
            raise NumericalError(
                f"[{self.MODEL_NAME}] _fx: floating point error at x={x}, t={t}: {e}"
            ) from e

    # ------------------------------------------------------------------
    def _hx(self, x, u, dt):
        if __debug__:
            if x.ndim == 2:
                assert all(a.shape == (1, 1) for a in (x, u))
            else:
                assert all(a.ndim == 3 and a.shape[1:] == (1, 1) for a in (x, u))
                assert x.shape[0] == u.shape[0]

        try:
            with np.errstate(all="raise"):
                return x + u
        except FloatingPointError as e:
            raise NumericalError(
                f"[{self.MODEL_NAME}] _hx: floating point error at x={x}, u={u}: {e}"
            ) from e

    # ------------------------------------------------------------------
    def _g(self, x, y, t, u, dt):
        if __debug__:
            if x.ndim == 2:
                assert all(a.shape == (1, 1) for a in (x, y, t, u))
            else:
                assert all(a.ndim == 3 and a.shape[1:] == (1, 1) for a in (x, y, t, u))
                assert x.shape[0] == y.shape[0] == t.shape[0] == u.shape[0]

        try:
            fx_val = self._fx(x, t, dt)
            hx_val = self._hx(fx_val, u, dt)
            if x.ndim == 2:
                return np.vstack((fx_val, hx_val))
            else:
                return np.concatenate((fx_val, hx_val), axis=1)
        except NumericalError:
            raise
        except ValueError as e:
            raise NumericalError(
                f"[{self.MODEL_NAME}] _g: shape mismatch during stack: {e}"
            ) from e

    # ------------------------------------------------------------------
    def _jacobiens_g(self, x, y, t, u, dt):
        if __debug__:
            if x.ndim == 2:
                assert x.shape == (1, 1)
            else:
                assert x.ndim == 3 and x.shape[1:] == (1, 1)

        try:
            with np.errstate(all="raise"):
                if x.ndim == 2:
                    dfdx = 0.9 - 1.8 * x[0, 0] ** 2

                    An = np.array([[dfdx, 0.0], [dfdx, 0.0]])
                    Bn = np.array([[1.0, 0.0], [1.0, 1.0]])
                else:
                    N = x.shape[0]
                    dfdx = 0.9 - 1.8 * x[:, 0, 0] ** 2  # (N,)

                    An = np.zeros((N, 2, 2))
                    An[:, 0, 0] = dfdx
                    An[:, 1, 0] = dfdx

                    Bn = np.tile(np.array([[1.0, 0.0], [1.0, 1.0]]), (N, 1, 1))

            return An, Bn

        except FloatingPointError as e:
            raise NumericalError(
                f"[{self.MODEL_NAME}] _jacobiens_g: floating point error at x={x}: {e}"
            ) from e
        except (IndexError, ValueError) as e:
            raise NumericalError(
                f"[{self.MODEL_NAME}] _jacobiens_g: array construction error: {e}"
            ) from e
