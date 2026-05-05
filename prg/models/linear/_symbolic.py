"""SymbolicMixin — SymPy representation of (A, B) and LaTeX rendering."""

from __future__ import annotations

import re

import numpy as np
import sympy as sp

__all__ = ["SymbolicMixin"]


class SymbolicMixin:
    """
    Mixin providing the symbolic representation and LaTeX export.

    Assumes the host class exposes ``A``, ``B``, ``mQ``, ``dim_x``,
    ``dim_y`` (set by the concrete subclasses ``LinearAmQ`` /
    ``LinearSigma`` after construction).
    """

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
