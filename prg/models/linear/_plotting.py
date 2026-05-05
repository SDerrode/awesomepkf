"""PlottingMixin — heatmap A/B + eigenvalues + g-dynamics visualisation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from prg.utils.exceptions import NumericalError
from prg.utils.plot_settings import BIG_SIZE, DPI, FACECOLOR

__all__ = ["PlottingMixin"]


class PlottingMixin:
    """
    Mixin providing matplotlib-based visualisation helpers.

    Assumes the host class exposes ``A``, ``B``, ``dim_x``, ``dim_y``,
    ``dim_xy``, and ``g(z, noise, dt)`` (set by the concrete subclasses
    after construction).
    """

    # ------------------------------------------------------------------
    # Helpers
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
    # Public plotting methods
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

        Path("data/plot").mkdir(parents=True, exist_ok=True)
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

        Path("data/plot").mkdir(parents=True, exist_ok=True)
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
