from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from scipy.linalg import LinAlgError, cho_factor, cho_solve

from prg.classes.MatrixDiagnostics import CovarianceMatrix, StabilityMatrix
from prg.classes.SeedGenerator import SeedGenerator
from prg.utils.exceptions import NumericalError
from prg.utils.generate_matrix_cov import generate_block_matrix
from prg.utils.plot_settings import BIG_SIZE, DPI, FACECOLOR

__all__ = ["BaseModelLinear", "LinearAmQ", "LinearSigma"]


class BaseModelLinear:
    """
    Base class for linear models.

    Two possible parametrisations:
      1. 'linear_AmQ' : dynamics z_{n+1} = A z_n + B noise, with covariance Q
      2. 'linear_Sigma' : parametrisation via variances sxx, syy and coefficients a, b, c, d, e
    """

    def __init__(self, dim_x, dim_y, model_type, augmented=False, pairwiseModel=True):
        if not isinstance(dim_x, int) or dim_x <= 0:
            raise ValueError("dim_x must be a positive integer")
        if not isinstance(dim_y, int) or dim_y <= 0:
            raise ValueError("dim_y must be a positive integer")
        if model_type not in ("linear_AmQ", "linear_Sigma"):
            raise ValueError("model_type must be 'linear_AmQ' or 'linear_Sigma'")

        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_xy = dim_x + dim_y
        self.model_type = model_type
        self.augmented = augmented
        self.pairwiseModel = pairwiseModel

        # UPKF specific parameters
        self.alpha = 0.25
        self.beta = 2.0
        self.kappa = 0.0
        # Option 1 — valeurs classiques Wan & Merwe
        # self.alpha = 1e-3
        # self.beta = 2.0
        # self.kappa = 0.0  # → lambda_ ≈ -2 + ε, Wm[0] ≈ -1, Wc[0] ≈ 1
        # Option 2 — cubature (symétrique, tous poids positifs)
        # self.alpha = 1.0
        # self.beta  = 0.0
        # self.kappa = 0.0   # → lambda_ = 0, Wm = Wc = 1/(2n)

        self.lambda_ = self.alpha**2 * (self.dim_x + self.kappa) - self.dim_x

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        n = cls.__name__
        cls.MODEL_NAME = n[0].lower() + n[1:]

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

    # ------------------------------------------------------------------
    def __repr__(self):
        return f"{self.__class__.__name__}(dim_x={self.dim_x}, dim_y={self.dim_y}, type={self.model_type})"

    # ------------------------------------------------------------------
    # Symbolic representation
    # ------------------------------------------------------------------

    def _build_symbolic_model(self) -> None:
        """
        Builds the SymPy representation of A and B.

        Called at the end of LinearAmQ.__init__ and LinearSigma._initSigma(),
        once self.A and self.B are numerically available.

        Attempts to simplify floating-point entries as exact fractions
        (tolerance 1e-4) for a more readable LaTeX rendering. Falls back to
        raw numerical values on failure.
        """

        def _to_sp(M: np.ndarray) -> sp.Matrix:
            def _simplify(x: float) -> sp.Expr:
                try:
                    return sp.nsimplify(x, rational=True, tolerance=1e-4)
                except Exception:
                    return sp.Float(x)

            return sp.Matrix(
                M.shape[0], M.shape[1], [_simplify(float(v)) for v in M.ravel()]
            )

        try:
            self._sA = _to_sp(self.A)
            self._sB = _to_sp(self.B)
        except Exception as e:
            raise RuntimeError(
                f"[{self.__class__.__name__}] _build_symbolic_model() failed — "
                f"check that self.A and self.B are valid ndarrays.\n"
                f"Cause: {type(e).__name__}: {e}"
            ) from e

    def latex_model(self) -> str:
        """
        Returns the LaTeX representation of the linear state-space model:

            z_k = A z_{k-1} + B v_k,   v = (v^x, v^y) ~ N(0, Q)

        Typographic conventions:
          - scalar (dim=1)     : plain italic     x, y
          - vector (dim>1)     : bold lowercase   \\mathbf{x}, ...
          - noise              : v^x (state), v^y (observation)
          - matrices A, B, Q  : bold uppercase
        """
        import re

        if not hasattr(self, "_sA"):
            raise RuntimeError(
                f"[{self.__class__.__name__}] _build_symbolic_model() not called — "
                "check initialisation order."
            )

        nx, ny = self.dim_x, self.dim_y
        bold_x = nx > 1
        bold_y = ny > 1

        # ------------------------------------------------------------------
        # LaTeX names
        # ------------------------------------------------------------------
        x_n = r"\mathbf{x}" if bold_x else "x"
        y_n = r"\mathbf{y}" if bold_y else "y"
        z_n = r"\mathbf{z}"
        vx_n = r"\mathbf{v}^x" if bold_x else "v^x"
        vy_n = r"\mathbf{v}^y" if bold_y else "v^y"
        v_n = r"\mathbf{v}"
        A_n = r"\mathbf{A}"
        B_n = r"\mathbf{B}"
        Q_n = r"\mathcal{Q}"

        # ------------------------------------------------------------------
        # mQ formatted to 2 decimal places
        # ------------------------------------------------------------------
        def _np_to_sp(M: np.ndarray) -> sp.Matrix:
            return sp.Matrix(
                M.shape[0],
                M.shape[1],
                [sp.Float(round(float(v), 2)) for v in M.ravel()],
            )

        mQ_sp = _np_to_sp(self.mQ)

        def _fix_latex(s: str) -> str:
            """1.0 \\cdot 10^{-k}  →  10^{-k}  (with or without spaces around \\cdot)"""
            return re.sub(r"1\.0\s*\\cdot\s*", "", s)

        # ------------------------------------------------------------------
        # Rendering of A and B: scalar without pmatrix, matrix with
        # ------------------------------------------------------------------
        def _lat_mat(mat: sp.Matrix) -> str:
            if mat.shape == (1, 1):
                return sp.latex(mat[0, 0])
            return sp.latex(mat)

        try:
            lines = [
                r"\begin{align}",
                # ── Dynamique globale
                rf"  {z_n}_k &= {A_n}\,{z_n}_{{k-1}} + {B_n}\,{v_n}_k \\[6pt]",
                # ── Block decomposition
                rf"  \begin{{pmatrix}} {x_n}_k \\ {y_n}_k \end{{pmatrix}}"
                rf" &= {A_n} \begin{{pmatrix}} {x_n}_{{k-1}} \\ {y_n}_{{k-1}} \end{{pmatrix}}"
                rf" + {B_n} \begin{{pmatrix}} {vx_n} \\ {vy_n} \end{{pmatrix}} \\[6pt]",
                # ── Noise distribution
                rf"  {v_n} = ({vx_n},\,{vy_n})"
                rf" &\sim \mathcal{{N}}\!\left(0,\; {Q_n}\right), \qquad"
                rf" {Q_n} = {sp.latex(mQ_sp)} \\[12pt]",
                # ── Matrice A
                rf"  {A_n} &= {_lat_mat(self._sA)} \\[6pt]",
                # ── Matrice B
                rf"  {B_n} &= {_lat_mat(self._sB)}",
                r"\end{align}",
            ]
            return _fix_latex("\n".join(lines))

        except Exception as e:
            raise RuntimeError(
                f"[{self.__class__.__name__}] latex_model: LaTeX rendering failed.\n"
                f"Cause: {type(e).__name__}: {e}"
            ) from e

    # ------------------------------------------------------------------
    # Helpers common to visualisation methods
    # ------------------------------------------------------------------

    def _make_grid(
        self,
        n_points: int,
        z_range: tuple[float, float] = (-1.0, 1.0),
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Builds a regular 2D grid. Returns Z1, Z2, Z_stack."""
        z = np.linspace(*z_range, n_points)
        Z1, Z2 = np.meshgrid(z, z)
        Z_stack = np.stack([Z1.ravel(), Z2.ravel()], axis=1)
        return Z1, Z2, Z_stack

    @staticmethod
    def _heatmap_ax(
        ax: plt.Axes,
        M: np.ndarray,
        title: str,
        annotate: bool = True,
        fmt: str = ".3f",
    ) -> None:
        """
        Draws an annotated heatmap of a numerical matrix on an existing axis.

        Parameters
        ----------
        annotate : displays the numerical value in each cell.
        fmt      : display format for values.
        """
        vmax = np.abs(M).max() or 1.0
        im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        plt.colorbar(im, ax=ax)
        ax.set_title(title, size=BIG_SIZE)
        ax.set_xticks(range(M.shape[1]))
        ax.set_yticks(range(M.shape[0]))
        ax.set_xticklabels([f"$z_{{{j}}}$" for j in range(M.shape[1])])
        ax.set_yticklabels([f"$z_{{{i}}}^+$" for i in range(M.shape[0])])
        if annotate:
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    ax.text(
                        j,
                        i,
                        format(M[i, j], fmt),
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white" if abs(M[i, j]) > 0.6 * vmax else "black",
                    )

    # ------------------------------------------------------------------
    def plot_jacobian(self) -> None:
        """
        Visualises matrices A and B of the linear model and the position
        of the eigenvalues of A in the complex plane.

        Available for any dimension.
        Saves the figure to data/plot/.

        Subplots  (2 rows × 2 columns)
        -------------
        (1,1) Annotated heatmap of A — transition matrix
        (1,2) Annotated heatmap of B — noise matrix
        (2,1) Eigenvalues of A in the complex plane
              • unit circle as dashed line
              • stable points (|λ| < 1) in blue, unstable in red
        (2,2) Summary table: spectral radius, stability,
              list of eigenvalues and their moduli
        """
        try:
            eigvals = np.linalg.eigvals(self.A)
        except np.linalg.LinAlgError as e:
            raise NumericalError(
                f"[{self.__class__.__name__}] plot_jacobian: "
                f"eigenvalue computation failed: {e}"
            ) from e
        rho = np.max(np.abs(eigvals))
        is_stable = rho < 1.0

        os.makedirs("data/plot", exist_ok=True)
        model_name = getattr(self, "MODEL_NAME", self.__class__.__name__)
        fig, axes = plt.subplots(2, 2, figsize=(13, 10))

        # ── (1,1) Heatmap A ──────────────────────────────────────────
        self._heatmap_ax(axes[0, 0], self.A, r"Transition matrix $A$")

        # ── (1,2) Heatmap B ──────────────────────────────────────────
        self._heatmap_ax(axes[0, 1], self.B, r"Noise matrix $B$")

        # ── (2,1) Eigenvalues ─────────────────────────────────────────
        ax = axes[1, 0]
        theta = np.linspace(0, 2 * np.pi, 300)
        ax.plot(np.cos(theta), np.sin(theta), "k--", lw=1.0, label="unit circle")
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(0, color="gray", lw=0.5)
        for k, lam in enumerate(eigvals):
            color = "#1f77b4" if abs(lam) < 1.0 else "#d62728"
            ax.scatter(lam.real, lam.imag, s=80, color=color, zorder=5)
            ax.annotate(
                rf"$\lambda_{{{k}}}$",
                (lam.real, lam.imag),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=9,
            )
        ax.set_title(r"Eigenvalues of $A$", size=BIG_SIZE)
        ax.set_xlabel(r"$\mathrm{Re}(\lambda)$")
        ax.set_ylabel(r"$\mathrm{Im}(\lambda)$")
        ax.set_aspect("equal")
        margin = max(0.3, rho * 0.2)
        ax.set_xlim(-rho - margin, rho + margin)
        ax.set_ylim(-rho - margin, rho + margin)
        ax.legend(fontsize=8)

        # ── (2,2) Summary table ───────────────────────────────────────
        ax = axes[1, 1]
        ax.axis("off")
        stab_color = "#1f77b4" if is_stable else "#d62728"
        stab_text = "✓ Stable" if is_stable else "✗ Unstable"
        lines = [
            rf"$\rho(A) = {rho:.6f}$",
            rf"Stability: {stab_text}",
            "",
            r"Eigenvalues $\lambda_k$ :",
        ]
        for k, lam in enumerate(eigvals):
            mod = abs(lam)
            _sign = "●" if mod < 1.0 else "●"
            _c = "#1f77b4" if mod < 1.0 else "#d62728"
            lines.append(
                rf"  $\lambda_{{{k}}}$ = {lam.real:+.4f} {'+' if lam.imag >= 0 else ''}{lam.imag:.4f}j"
                rf"   $|\lambda_{{{k}}}|$ = {mod:.4f}"
            )
        text = "\n".join(lines)
        ax.text(
            0.05,
            0.95,
            text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            family="monospace",
            color=stab_color,
        )

        fig.suptitle(
            rf"{model_name} — $A_n = A$ (constante)",
            fontsize=BIG_SIZE + 2,
            fontweight="bold",
        )
        plt.tight_layout()
        path = f"data/plot/jacobian_{model_name}.png"
        try:
            plt.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=FACECOLOR)
        except OSError as e:
            raise OSError(
                f"[{self.__class__.__name__}] plot_jacobian: "
                f"failed to write '{path}': {e}"
            ) from e
        finally:
            plt.close(fig)

    # ------------------------------------------------------------------

    def plot_g_dynamic(self, n_points=150, quiver_stride=10):
        if self.dim_x != 1 or self.dim_y != 1:
            return

        Z1, Z2, Z_stack = self._make_grid(n_points, (-1.0, 1.0))
        noise = np.zeros((self.dim_xy, 1))
        G = np.zeros_like(Z_stack)
        try:
            for k in range(Z_stack.shape[0]):
                G[k] = self.g(Z_stack[k].reshape(2, 1), noise, 1.0).ravel()
        except NumericalError:
            raise
        except (ValueError, np.exceptions.AxisError) as e:
            raise NumericalError(
                f"[{self.__class__.__name__}] plot_g_dynamic: error during evaluation of g: {e}"
            ) from e

        G1 = G[:, 0].reshape(n_points, n_points)
        G2 = G[:, 1].reshape(n_points, n_points)
        NormG = np.sqrt(G1**2 + G2**2)
        Dz1, Dz2 = G1 - Z1, G2 - Z2

        os.makedirs("data/plot", exist_ok=True)
        model_name = getattr(self, "MODEL_NAME", self.__class__.__name__)
        fig = plt.figure(figsize=(12, 8), facecolor=FACECOLOR)

        ax1 = fig.add_subplot(2, 2, 1, projection="3d")
        ax1.plot_surface(Z1, Z2, G1, cmap="viridis")
        ax1.set_title(r"$g_x(x,y)$", size=BIG_SIZE)
        ax1.set_xlabel(r"$x$")
        ax1.set_ylabel(r"$y$")
        ax1.view_init(30, 45)

        ax2 = fig.add_subplot(2, 2, 2, projection="3d")
        ax2.plot_surface(Z1, Z2, G2, cmap="viridis")
        ax2.set_title(r"$g_y(x,y)$", size=BIG_SIZE)
        ax2.set_xlabel(r"$x$")
        ax2.set_ylabel(r"$y$")
        ax2.view_init(30, 45)

        ax3 = fig.add_subplot(2, 2, 3)
        im = ax3.contourf(Z1, Z2, NormG, levels=20, cmap="plasma")
        ax3.set_title(r"$\|g(x,y)\|$", size=BIG_SIZE)
        ax3.set_xlabel(r"$x$")
        ax3.set_ylabel(r"$y$")
        plt.colorbar(im, ax=ax3)

        ax4 = fig.add_subplot(2, 2, 4)
        s = quiver_stride
        ax4.quiver(Z1[::s, ::s], Z2[::s, ::s], Dz1[::s, ::s], Dz2[::s, ::s])
        ax4.set_title(r"$g(x,y) - (x,y)$", size=BIG_SIZE)
        ax4.set_xlabel(r"$x$")
        ax4.set_ylabel(r"$y$")
        ax4.set_aspect("equal")

        fig.suptitle(model_name, fontsize=BIG_SIZE + 2, fontweight="bold")
        plt.tight_layout()
        path = f"data/plot/function_g_dynamic_{model_name}.png"
        try:
            plt.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=FACECOLOR)
        except OSError as e:
            raise OSError(
                f"[{self.__class__.__name__}] plot_g_dynamic: failed to write '{path}': {e}"
            ) from e
        finally:
            plt.close(fig)


# ----------------------------------------------------------
# Subclass for the linear_AmQ parametrisation
# ----------------------------------------------------------
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


# ----------------------------------------------------------
# Subclass for the linear_Sigma parametrisation
# ----------------------------------------------------------
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
