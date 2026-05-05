import contextlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from prg.classes.seed_generator import SeedGenerator
from prg.utils.exceptions import NumericalError
from prg.utils.generate_matrix_cov import generate_block_matrix
from prg.utils.plot_settings import BIG_SIZE, DPI, FACECOLOR

__all__ = ["BaseModelNonLinear", "NumericalError"]  # re-exported for convenience


class BaseModelNonLinear:
    """
    Base class for all non-linear models.

    Provides a unified structure for the fx, hx and g functions,
    as well as consistent management of parameters and covariance matrices.
    In optimised mode (launched with `python3 -O`), checks are disabled.
    """

    def __init__(
        self, dim_x: int, dim_y: int, model_type: str = "nonlinear", augmented=False
    ):

        assert isinstance(dim_x, int) and dim_x > 0, "dim_x must be a positive integer"
        assert isinstance(dim_y, int) and dim_y > 0, "dim_y must be a positive integer"

        self.model_type = model_type
        self.augmented = augmented
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_xy = dim_x + dim_y

        # ── Unscented-Transform parameters (consumed by UKF / UPKF) ─────────
        # alpha=0.25 deviates from Wan & Merwe's classical 1e-3 to keep the
        # central weight Wm[0] non-negative for the dimensions used here, which
        # is important for the pairwise (UPKF) augmentation. See
        # ``prg/models/linear/_base.py`` for the full rationale and tuning notes.
        self.alpha = 0.25
        self.beta = 2.0
        self.kappa = 0.0
        self.lambda_ = self.alpha**2 * (self.dim_x + self.kappa) - self.dim_x

        # State matrix / vector initialisation
        self.mQ = None
        self.mz0 = None
        self.Pz0 = None

        self.pairwiseModel = None  # set by one of the 2 subclasses

        self._randMatrices = SeedGenerator(9)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        n = cls.__name__
        cls.MODEL_NAME = n[0].lower() + n[1:]

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

    # ------------------------------------------------------------------
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
            axis = 1 if z.ndim == 3 else 0
            x, y = np.split(z, [self.dim_x], axis=axis)
            nx, ny = np.split(noise_z, [self.dim_x], axis=axis)
            return self._g(x, y, nx, ny, dt)
        except NumericalError:
            raise
        except ValueError as e:
            raise NumericalError(
                f"[{self.__class__.__name__}] g: erreur de split/shape: {e}"
            ) from e

    def jacobiens_g(self, z: np.ndarray, noise_z: np.ndarray, dt: float) -> np.ndarray:
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
            axis = 1 if z.ndim == 3 else 0
            x, y = np.split(z, [self.dim_x], axis=axis)
            nx, ny = np.split(noise_z, [self.dim_x], axis=axis)
            return self._jacobiens_g(x, y, nx, ny, dt)
        except NumericalError:
            raise
        except ValueError as e:
            raise NumericalError(
                f"[{self.__class__.__name__}] jacobiens_g: erreur de split/shape: {e}"
            ) from e

    # ------------------------------------------------------------------
    def _g(self, x, y, nx, ny, dt):
        """To be implemented in the subclass"""
        raise NotImplementedError

    def _jacobiens_g(self, x, y, nx, ny, dt):
        """To be implemented in the subclass"""
        raise NotImplementedError

    def latex_model(self) -> str:
        """
        Returns the LaTeX representation of the model (equations + Jacobians).
        To be implemented in the subclass.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    @staticmethod
    def _wrap_lambdify(f):
        """
        Guarantees that the result of sp.lambdify is always a callable.

        When a SymPy expression is constant (no free symbols,
        e.g. Bn = I for additive noise, or H = [0,1] for hx = x[-1]),
        lambdify generates a function that directly returns an ndarray
        instead of a callable accepting arguments.
        This wrapper detects this case and returns a lambda that ignores its
        arguments and always returns the constant value.
        """
        if callable(f):
            return f
        constant = np.array(f, dtype=float)
        return lambda *args: constant

    # ------------------------------------------------------------------
    def _safe_eval(self, label, lambda_fn, build_args, output_shape, x):
        """
        Shared try/except + 2D/3D dispatch for lambdified evaluators.

        Wraps the FloatingPointError → NumericalError pattern that was
        duplicated 8 times across BaseModelFxHx and BaseModelGxGy.

        Parameters
        ----------
        label : str
            Method name shown in error messages (e.g. ``"_eval_fx"``).
        lambda_fn : callable
            Output of ``sp.lambdify(...)`` (after ``_wrap_lambdify``).
        build_args : Callable[[int | None], tuple]
            Returns the positional arguments for ``lambda_fn`` at batch
            index ``i`` (3D case) or ``None`` (2D case).
        output_shape : tuple[int, ...]
            Shape of one evaluation result (e.g. ``(dim_x, 1)``).
        x : np.ndarray
            Drives 2D vs 3D dispatch via ``x.ndim``.

        Returns
        -------
        np.ndarray
            Shape ``output_shape`` if 2D, ``(N, *output_shape)`` if 3D.

        Raises
        ------
        NumericalError
            On ``FloatingPointError``, ``ValueError`` or ``IndexError``.
        """
        try:
            with np.errstate(all="raise"):
                if x.ndim == 2:
                    return np.array(
                        lambda_fn(*build_args(None)), dtype=float
                    ).reshape(output_shape)
                N = x.shape[0]
                out = np.empty((N, *output_shape))
                for i in range(N):
                    out[i] = np.array(
                        lambda_fn(*build_args(i)), dtype=float
                    ).reshape(output_shape)
                return out
        except FloatingPointError as e:
            raise NumericalError(
                f"[{self.__class__.__name__}] {label}: numerical error: {e}"
            ) from e
        except (ValueError, IndexError) as e:
            raise NumericalError(
                f"[{self.__class__.__name__}] {label}: shape error: {e}"
            ) from e

    # ------------------------------------------------------------------
    def get_params(self) -> dict:
        return {
            "dim_x": self.dim_x,
            "dim_y": self.dim_y,
            "augmented": self.augmented,
            "pairwiseModel": self.pairwiseModel,
            "g": self.g,
            "f": getattr(self, "_fx", None),
            "h": getattr(self, "_hx", None),
            "jacobiens_g": self.jacobiens_g,  # for EPKF
            "alpha": self.alpha,  # for UPKF
            "beta": self.beta,  # for UPKF
            "kappa": self.kappa,  # for UPKF
            "lambda_": self.lambda_,  # for UPKF
            "mQ": self.mQ,
            "mz0": self.mz0,
            "Pz0": self.Pz0,
        }

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dim_x={self.dim_x}, dim_y={self.dim_y})"

    # ------------------------------------------------------------------
    # Helpers common to visualisation methods
    # ------------------------------------------------------------------

    def _make_grid(
        self, n_points: int, z_range: tuple[float, float] = (-3.0, 3.0)
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Builds a regular 2D grid over z_range × z_range.

        Returns
        --------
        Z1, Z2 : (n_points, n_points)  — meshgrid
        Z_stack : (n_points², 2)        — list of points z = [x, y]ᵀ
        """
        z = np.linspace(*z_range, n_points)
        Z1, Z2 = np.meshgrid(z, z)
        Z_stack = np.stack([Z1.ravel(), Z2.ravel()], axis=1)
        return Z1, Z2, Z_stack

    def _eval_g_on_grid(self, Z_stack: np.ndarray, n_points: int) -> np.ndarray:
        """
        Evaluates g(z, noise=0, dt=1) over the entire grid.

        Returns
        --------
        G : (n_points², 2) — values of g; NaN at singular points.
        """
        noise = np.zeros((self.dim_xy, 1))
        G = np.full((Z_stack.shape[0], self.dim_xy), np.nan)
        for k in range(Z_stack.shape[0]):
            z = Z_stack[k].reshape(self.dim_xy, 1)
            try:
                val = self.g(z, noise, 1.0)
                if np.isfinite(val).all():
                    G[k] = val.ravel()
            except (NumericalError, FloatingPointError):
                pass  # leaves NaN in place
        return G

    def _eval_jac_on_grid(self, Z_stack: np.ndarray, n_points: int) -> np.ndarray:
        """
        Evaluates An = jacobiens_g(z, noise=0, dt=1)[0] over the entire grid.

        Returns
        --------
        AN : (n_points², dim_xy, dim_xy) — An at each point; NaN at singularities.
        """
        nz = self.dim_xy
        noise = np.zeros((nz, 1))
        AN = np.full((Z_stack.shape[0], nz, nz), np.nan)
        for k in range(Z_stack.shape[0]):
            z = Z_stack[k].reshape(nz, 1)
            try:
                An, _ = self.jacobiens_g(z, noise, 1.0)
                if np.isfinite(An).all():
                    AN[k] = An
            except (NumericalError, FloatingPointError):
                pass
        return AN

    @staticmethod
    def _contourf_ax(
        ax: plt.Axes,
        Z1: np.ndarray,
        Z2: np.ndarray,
        data: np.ndarray,
        title: str,
        cmap: str = "RdBu_r",
        sym: bool = False,
    ) -> None:
        """
        Draws a contourf with colorbar on an existing axis.

        Parameters
        ----------
        sym   : if True, centres the colormap on 0 (useful for signed values).
        """
        kwargs = {"cmap": cmap, "levels": 20}
        if sym:
            vmax = np.nanmax(np.abs(data))
            if vmax > 0:
                kwargs["vmin"] = -vmax
                kwargs["vmax"] = vmax
        im = ax.contourf(Z1, Z2, data, **kwargs)
        ax.set_title(title, size=BIG_SIZE)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        plt.colorbar(im, ax=ax)

    # ------------------------------------------------------------------
    def plot_g_dynamic(
        self,
        n_points: int = 150,
        quiver_stride: int = 10,
        z_range: tuple[float, float] = (-3.0, 3.0),
    ) -> None:
        """
        Visualises g(z, noise=0, dt=1) over z_range × z_range.
        Available only for dim_x=1, dim_y=1.
        Saves the figure to data/plot/.

        Subplots
        ------------
        (1,1) 3D surface of g_x(x, y)
        (1,2) 3D surface of g_y(x, y)
        (2,1) Map of the norm ‖g(x, y)‖
        (2,2) Displacement field g(z) − z  (quiver)
        """
        if self.dim_x != 1 or self.dim_y != 1:
            return

        Z1, Z2, Z_stack = self._make_grid(n_points, z_range)
        G = self._eval_g_on_grid(Z_stack, n_points)

        G1 = G[:, 0].reshape(n_points, n_points)
        G2 = G[:, 1].reshape(n_points, n_points)
        NormG = np.sqrt(G1**2 + G2**2)
        Dz1, Dz2 = G1 - Z1, G2 - Z2

        Path("data/plot").mkdir(parents=True, exist_ok=True)
        fig = plt.figure(figsize=(12, 8))

        # Surface g_x
        ax1 = fig.add_subplot(2, 2, 1, projection="3d")
        ax1.plot_surface(Z1, Z2, G1, cmap="viridis")
        ax1.set_title(r"$g_x(x,y)$", size=BIG_SIZE)
        ax1.set_xlabel(r"$x$")
        ax1.set_ylabel(r"$y$")
        ax1.view_init(30, 45)

        # Surface g_y
        ax2 = fig.add_subplot(2, 2, 2, projection="3d")
        ax2.plot_surface(Z1, Z2, G2, cmap="viridis")
        ax2.set_title(r"$g_y(x,y)$", size=BIG_SIZE)
        ax2.set_xlabel(r"$x$")
        ax2.set_ylabel(r"$y$")
        ax2.view_init(30, 45)

        # Norm ‖g‖
        ax3 = fig.add_subplot(2, 2, 3)
        self._contourf_ax(ax3, Z1, Z2, NormG, r"$\|g(x,y)\|$", cmap="plasma")

        # Displacement field
        ax4 = fig.add_subplot(2, 2, 4)
        s = quiver_stride
        ax4.quiver(Z1[::s, ::s], Z2[::s, ::s], Dz1[::s, ::s], Dz2[::s, ::s])
        ax4.set_title(r"$g(x,y) - (x,y)$", size=BIG_SIZE)
        ax4.set_xlabel(r"$x$")
        ax4.set_ylabel(r"$y$")
        ax4.set_aspect("equal")

        model_name = getattr(self, "MODEL_NAME", self.__class__.__name__)
        fig.suptitle(model_name, fontsize=BIG_SIZE + 2, fontweight="bold")
        plt.tight_layout()
        path = f"data/plot/function_g_dynamic_{model_name}.png"
        plt.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=FACECOLOR)
        plt.close(fig)

    # ------------------------------------------------------------------
    def plot_jacobian_dynamic(
        self,
        n_points: int = 100,
        z_range: tuple[float, float] = (-3.0, 3.0),
    ) -> None:
        """
        Visualises An = dg/dz evaluated at (z, noise=0, dt=1) over z_range × z_range.
        Available only for dim_x=1, dim_y=1 (An is 2×2).
        Saves the figure to data/plot/.

        Subplots  (2 rows × 3 columns)
        -------------
        Row 1 — entries of the 1st row of An:
          (1,1) ∂g_x/∂x   (1,2) ∂g_x/∂y   (1,3) Re(λ₁)
        Row 2 — entries of the 2nd row of An:
          (2,1) ∂g_y/∂x   (2,2) ∂g_y/∂y   (2,3) Re(λ₂)

        The Re(λ) fields give local stability: |Re(λ)| < 1 indicates
        contraction in that eigendirection (discrete time).
        The stability boundary |Re(λ)| = 1 is drawn as white dashed lines.
        """
        if self.dim_x != 1 or self.dim_y != 1:
            return

        Z1, Z2, Z_stack = self._make_grid(n_points, z_range)
        AN = self._eval_jac_on_grid(Z_stack, n_points)  # (n², 2, 2)

        # Reshape of the 4 inputs
        A00 = AN[:, 0, 0].reshape(n_points, n_points)
        A01 = AN[:, 0, 1].reshape(n_points, n_points)
        A10 = AN[:, 1, 0].reshape(n_points, n_points)
        A11 = AN[:, 1, 1].reshape(n_points, n_points)

        # Eigenvalues: eigenvalues of each 2×2 matrix
        # AN shape (n², 2, 2) → eigvals shape (n², 2)
        eigvals = np.linalg.eigvals(AN)  # (n², 2), complexes
        lam1 = np.real(eigvals[:, 0]).reshape(n_points, n_points)
        lam2 = np.real(eigvals[:, 1]).reshape(n_points, n_points)

        Path("data/plot").mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))

        panels = [
            (axes[0, 0], A00, r"$\partial g_x / \partial x$"),
            (axes[0, 1], A01, r"$\partial g_x / \partial y$"),
            (axes[1, 0], A10, r"$\partial g_y / \partial x$"),
            (axes[1, 1], A11, r"$\partial g_y / \partial y$"),
        ]
        for ax, data, title in panels:
            self._contourf_ax(ax, Z1, Z2, data, title, cmap="RdBu_r", sym=True)

        # Re(λ) fields with isocurve |Re(λ)| = 1
        for ax, lam, title in [
            (axes[0, 2], lam1, r"$\mathrm{Re}(\lambda_1)$"),
            (axes[1, 2], lam2, r"$\mathrm{Re}(\lambda_2)$"),
        ]:
            self._contourf_ax(ax, Z1, Z2, lam, title, cmap="RdBu_r", sym=True)
            # Stability boundary |Re(λ)| = 1
            for level, ls in [(-1.0, "--"), (1.0, "--")]:
                with contextlib.suppress(Exception):
                    ax.contour(
                        Z1,
                        Z2,
                        lam,
                        levels=[level],
                        colors="white",
                        linestyles=ls,
                        linewidths=1.2,
                    )

        model_name = getattr(self, "MODEL_NAME", self.__class__.__name__)
        fig.suptitle(
            rf"{model_name} — Jacobienne $A_n = \partial g / \partial z$",
            fontsize=BIG_SIZE + 2,
            fontweight="bold",
        )
        plt.tight_layout()
        path = f"data/plot/jacobian_dynamic_{model_name}.png"
        plt.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=FACECOLOR)
        plt.close(fig)
