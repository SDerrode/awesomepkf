"""DynamicsMixin — evaluation of g, f, h, and their (constant) Jacobians."""

import numpy as np

from prg.utils.exceptions import NumericalError

__all__ = ["DynamicsMixin"]


class DynamicsMixin:
    """
    Mixin providing the dynamic-system evaluation routines.

    Assumes the host class exposes ``dim_x``, ``dim_y``, ``dim_xy``,
    ``A`` and ``B`` (set by the concrete subclasses ``LinearAmQ`` /
    ``LinearSigma`` after construction).
    """

    def g(self, z, noise_z, dt):
        if __debug__:
            if z.ndim == 2:
                assert all(a.shape == (self.dim_xy, 1) for a in (z, noise_z))
            else:
                assert all(
                    a.ndim == 3 and a.shape[1:] == (self.dim_xy, 1)
                    for a in (z, noise_z)
                )
                assert z.shape[0] == noise_z.shape[0]

        try:
            if z.ndim == 2:
                return self.A @ z + self.B @ noise_z
            return np.einsum("ij,njk->nik", self.A, z) + np.einsum(
                "ij,njk->nik", self.B, noise_z
            )
        except (ValueError, np.exceptions.AxisError) as e:
            raise NumericalError(
                f"[{self.__class__.__name__}] g: matrix multiplication error: {e}"
            ) from e

    def _fx(self, x, noise_x, dt):
        """
        Evaluates the transition f(x, noise_x) = A_xx @ x + B_xx @ noise_x.

        x, noise_x : (dim_x, 1)       → returns (dim_x, 1)
        x, noise_x : (N, dim_x, 1)    → returns (N, dim_x, 1)
        """
        if __debug__:
            if x.ndim == 2:
                assert all(a.shape == (self.dim_x, 1) for a in (x, noise_x))
            else:
                assert all(
                    a.ndim == 3 and a.shape[1:] == (self.dim_x, 1) for a in (x, noise_x)
                )
                assert x.shape[0] == noise_x.shape[0]

        A_xx = self.A[: self.dim_x, : self.dim_x]  # (dim_x, dim_x)
        B_xx = self.B[: self.dim_x, : self.dim_x]  # (dim_x, dim_x)

        try:
            if x.ndim == 2:
                return A_xx @ x + B_xx @ noise_x
            return np.einsum("ij,njk->nik", A_xx, x) + np.einsum(
                "ij,njk->nik", B_xx, noise_x
            )
        except (ValueError, np.exceptions.AxisError) as e:
            raise NumericalError(
                f"[{self.__class__.__name__}] _fx: matrix multiplication error: {e}"
            ) from e

    def _hx(self, x, noise_y, dt):
        """
        h(x, noise_y) = A_yx @ x + B_yy @ noise_y

        x, noise_y : (dim_x, 1)       → returns (dim_y, 1)
        x, noise_y : (N, dim_x, 1)    → returns (N, dim_y, 1)
        """
        if __debug__:
            if x.ndim == 2:
                assert x.shape == (self.dim_x, 1)
                assert noise_y.shape == (self.dim_y, 1)
            else:
                assert x.ndim == 3 and x.shape[1:] == (self.dim_x, 1)
                assert noise_y.ndim == 3 and noise_y.shape[1:] == (self.dim_y, 1)
                assert x.shape[0] == noise_y.shape[0]

        A_yx = self.A[self.dim_x :, : self.dim_x]  # (dim_y, dim_x) ← correct block
        B_yy = self.B[self.dim_x :, self.dim_x :]  # (dim_y, dim_y)

        try:
            if x.ndim == 2:
                return A_yx @ x + B_yy @ noise_y
            return np.einsum("ij,njk->nik", A_yx, x) + np.einsum(
                "ij,njk->nik", B_yy, noise_y
            )
        except (ValueError, np.exceptions.AxisError) as e:
            raise NumericalError(
                f"[{self.__class__.__name__}] _hx: matrix multiplication error: {e}"
            ) from e

    def jacobiens_g(self, z, noise_z, dt):
        if __debug__:
            if z.ndim == 2:
                assert all(a.shape == (self.dim_xy, 1) for a in (z, noise_z))
            else:
                assert all(
                    a.ndim == 3 and a.shape[1:] == (self.dim_xy, 1)
                    for a in (z, noise_z)
                )
                assert z.shape[0] == noise_z.shape[0]

        try:
            if z.ndim == 2:
                return self.A, self.B
            N = z.shape[0]
            return np.tile(self.A, (N, 1, 1)), np.tile(self.B, (N, 1, 1))
        except (ValueError, np.exceptions.AxisError) as e:
            raise NumericalError(
                f"[{self.__class__.__name__}] jacobiens_g: shape error: {e}"
            ) from e
