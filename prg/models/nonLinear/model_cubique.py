import numpy as np
from prg.models.nonLinear.base_model_nonLinear import BaseModelNonLinear
from prg.models.Generate_MatrixCov import generate_block_matrix

__all__ = ["ModelCubique"]


class NumericalError(RuntimeError):
    """Raised when a numerical computation fails in a model."""

    pass


class ModelCubique(BaseModelNonLinear):

    MODEL_NAME: str = "x1_y1_cubique"

    def __init__(self) -> None:
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
    def _fx(self, x: np.ndarray, t: np.ndarray, dt: float) -> np.ndarray:
        if __debug__:
            assert x.shape == (1, 1)
            assert t.shape == (1, 1)

        try:
            with np.errstate(all="raise"):
                return 0.9 * x - 0.2 * x**3 + t
        except FloatingPointError as e:
            raise NumericalError(
                f"[{self.MODEL_NAME}] _fx: floating point error at x={x}, t={t}: {e}"
            ) from e

    # ------------------------------------------------------------------
    def _hx(self, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        if __debug__:
            assert x.shape == (1, 1)
            assert u.shape == (1, 1)

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
            assert x.shape == (1, 1)
            assert y.shape == (1, 1)
            assert t.shape == (1, 1)
            assert u.shape == (1, 1)

        try:
            fx_val = self._fx(x, t, dt)
            hx_val = self._hx(fx_val, u, dt)
            return np.vstack((fx_val, hx_val))
        except NumericalError:
            raise  # already enriched, let it propagate
        except ValueError as e:
            raise NumericalError(
                f"[{self.MODEL_NAME}] _g: shape mismatch during vstack: {e}"
            ) from e

    # ------------------------------------------------------------------
    def _jacobiens_g(self, x, y, t, u, dt):

        if __debug__:
            assert x.shape == (1, 1)

        try:
            with np.errstate(all="raise"):
                dfdx = 0.9 - 0.6 * x[0, 0] ** 2

                An = np.array([[dfdx, 0.0], [dfdx, 0.0]])
                Bn = np.array([[1.0, 0.0], [1.0, 1.0]])

            return An, Bn

        except FloatingPointError as e:
            raise NumericalError(
                f"[{self.MODEL_NAME}] _jacobiens_g: floating point error at x={x}: {e}"
            ) from e
        except (IndexError, ValueError) as e:
            raise NumericalError(
                f"[{self.MODEL_NAME}] _jacobiens_g: array construction error: {e}"
            ) from e
