"""
LaTeX rendering helpers shared between BaseModelFxHx and BaseModelGxGy.

The two ``latex_model()`` methods previously duplicated ~70 LOC of
formatting helpers (substitution dicts, noise-name builders, ``np→sp``
conversion, ``\\cdot`` cleanup, noise-last reordering, scalar-vs-matrix
LaTeX rendering). Those helpers now live in :class:`_LatexBuilder`,
which both bases instantiate at the start of their own ``latex_model``.

Each subclass only writes the assembly that is form-specific:
- FxHx renders ``f(x, v^x)`` and ``h(y, v^y)`` (h evaluated at y = f(x)).
- GxGy renders ``g_x(x, y, v^x)`` and ``g_y(x, y, v^y)`` (both depend
  on x and y simultaneously).
"""

from __future__ import annotations

import re

import numpy as np
import sympy as sp

__all__ = ["_LatexBuilder"]


class _LatexBuilder:
    """LaTeX rendering helpers parametrised by the model's ``(dim_x, dim_y, mQ)``."""

    def __init__(self, dim_x: int, dim_y: int, mQ: np.ndarray) -> None:
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.bold_x = dim_x > 1
        self.bold_y = dim_y > 1

        self.vx_names: list[str] = [
            "v^x" if dim_x == 1 else rf"v^x_{i}" for i in range(dim_x)
        ]
        self.vy_names: list[str] = [
            "v^y" if dim_y == 1 else rf"v^y_{i}" for i in range(dim_y)
        ]

        # Display names of the LHS variables
        self.x_n = r"\mathbf{x}" if self.bold_x else "x"
        self.y_n = r"\mathbf{y}" if self.bold_y else "y"
        self.vx_n = r"\mathbf{v}^x" if self.bold_x else "v^x"
        self.vy_n = r"\mathbf{v}^y" if self.bold_y else "v^y"
        self.v_n = r"\mathbf{v}"
        self.Q_n = r"\mathcal{Q}"

        self.mQ_sp = self._np_to_sp(mQ)

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _np_to_sp(M: np.ndarray) -> sp.Matrix:
        """Convert a numpy matrix to a SymPy matrix with 2-decimal rounding."""
        return sp.Matrix(
            M.shape[0],
            M.shape[1],
            [sp.Float(round(float(v), 2)) for v in M.ravel()],
        )

    @staticmethod
    def fix_latex(s: str) -> str:
        """Strip ``1.0 \\cdot`` prefixes that SymPy emits for unit coefficients."""
        return re.sub(r"1\.0\s*\\cdot\s*", "", s)

    @staticmethod
    def apply(mat: sp.Matrix, subs: dict) -> sp.Matrix:
        """Apply a substitution dict to every entry of a SymPy matrix."""
        return sp.Matrix(mat.shape[0], mat.shape[1], [e.subs(subs) for e in mat])

    @staticmethod
    def lat_jac(mat: sp.Matrix) -> str:
        """Render a Jacobian: scalar (1×1) without pmatrix, otherwise with."""
        if mat.shape == (1, 1):
            return sp.latex(mat[0, 0])
        return sp.latex(mat)

    # ------------------------------------------------------------------
    # Substitution dict
    # ------------------------------------------------------------------

    @staticmethod
    def sub_dict(syms, base: str, bold: bool, names: list[str] | None = None) -> dict:
        """
        Build a ``{old_symbol: pretty_symbol}`` mapping.

        - dim 1 + no override: ``base`` (no index, no bold)
        - dim>1 + bold: ``\\mathbf{base}_i``
        - ``names`` overrides everything (used for noise renaming)
        """
        subs: dict = {}
        for i, s in enumerate(syms):
            if names is not None:
                latex_name = names[i]
            elif len(syms) == 1:
                latex_name = rf"\mathbf{{{base}}}" if bold else base
            else:
                latex_name = rf"\mathbf{{{base}}}_{i}" if bold else rf"{base}_{i}"
            subs[s] = sp.Symbol(latex_name, real=True)
        return subs

    # ------------------------------------------------------------------
    # Noise-last reordering
    # ------------------------------------------------------------------

    def reorder_noise_last(self, expr: sp.Expr) -> str:
        """
        Render an expression with noise terms forced to the end.

        Works around SymPy's canonical Add ordering by extracting the
        deterministic part (noise → 0) and appending the noise part
        afterwards. For multiplicative noise (e.g. ``x*exp(A + v^x)``)
        falls back to plain ``sp.latex(expr)``.
        """
        noise_syms = {sp.Symbol(n, real=True) for n in self.vx_names + self.vy_names}
        zero_subs = dict.fromkeys(noise_syms, 0)
        det = expr.subs(zero_subs)
        noise = sp.expand(expr - det)
        if noise == 0:
            return sp.latex(det)
        if det == 0:
            return sp.latex(noise)
        # multiplicative noise — give up reordering
        if not noise.free_symbols.issubset(noise_syms):
            return sp.latex(expr)
        noise_lat = sp.latex(noise)
        sep = " " if noise_lat.startswith("-") else " + "
        return sp.latex(det) + sep + noise_lat

    def lat(self, mat: sp.Matrix) -> str:
        """Render a value matrix; scalar without ``pmatrix``, vector with."""
        if mat.shape == (1, 1):
            return self.reorder_noise_last(mat[0, 0])
        parts = [self.reorder_noise_last(e) for e in mat]
        rows = r" \\ ".join(parts)
        return rf"\begin{{pmatrix}} {rows} \end{{pmatrix}}"

    # ------------------------------------------------------------------
    # Reusable LaTeX block
    # ------------------------------------------------------------------

    def noise_distribution_line(self) -> str:
        """Return the standard noise-distribution row of an ``align`` block."""
        return (
            rf"  {self.v_n} = ({self.vx_n},\,{self.vy_n})"
            rf" &\sim \mathcal{{N}}\!\left(0,\; {self.Q_n}\right), \qquad"
            rf" {self.Q_n} = {sp.latex(self.mQ_sp)} \\[12pt]"
        )
