from abc import ABC, abstractmethod

import numpy as np
import sympy as sp

from prg.models.nonLinear.base_model_nonLinear import BaseModelNonLinear
from prg.utils.exceptions import NumericalError

__all__ = ["BaseModelGxGy"]


class BaseModelGxGy(BaseModelNonLinear, ABC):
    """
    Parent class for models defined symbolically via symbolic_model().

    Handles dim_x >= 1 and dim_y >= 1 with full feedback:
    gx and gy can simultaneously depend on x, y, t and u.

    The subclass must implement:

        def symbolic_model(self, sx, sy, st, su):
            ...
            return sgx, sgy

    where:
        sx  : sp.Matrix(dim_x, 1) — symbols x0 .. x_{dim_x-1}
        sy  : sp.Matrix(dim_y, 1) — symbols y0 .. y_{dim_y-1}
        st  : sp.Matrix(dim_x, 1) — symbols t0 .. t_{dim_x-1}  (state noise)
        su  : sp.Matrix(dim_y, 1) — symbols u0 .. u_{dim_y-1}  (obs noise)
        sgx : sp.Matrix(dim_x, 1) — expression of the transition  gx(x, y, t, u)
        sgy : sp.Matrix(dim_y, 1) — expression of the observation gy(x, y, t, u)

    Example dim_x=1, dim_y=1:

        def symbolic_model(self, sx, sy, st, su):
            x, y, t, u = sx[0], sy[0], st[0], su[0]
            sgx = sp.Matrix([a * x + b * sp.tanh(y) + t])
            sgy = sp.Matrix([c * y + d * sp.sin(x) + u])
            return sgx, sgy

    The Jacobians An = dg/dz and Bn = dg/d_noise are computed automatically
    (no chain rule: gx and gy are evaluated at the same point (x, y)).

        An = d[gx; gy] / d[x; y]   (dim_xy, dim_xy)
        Bn = d[gx; gy] / d[t; u]   (dim_xy, dim_xy)
    """

    # ------------------------------------------------------------------
    @abstractmethod
    def symbolic_model(self, sx, sy, st, su):
        """
        To be implemented in the subclass.

        Parameters
        ----------
        sx : sp.Matrix(dim_x, 1)  — state symbols         x0 .. x_{dim_x-1}
        sy : sp.Matrix(dim_y, 1)  — observation symbols   y0 .. y_{dim_y-1}
        st : sp.Matrix(dim_x, 1)  — state noise symbols   t0 .. t_{dim_x-1}
        su : sp.Matrix(dim_y, 1)  — obs. noise symbols    u0 .. u_{dim_y-1}

        Returns
        --------
        sgx : sp.Matrix(dim_x, 1) — transition  gx(x, y, t, u)
        sgy : sp.Matrix(dim_y, 1) — observation gy(x, y, t, u)
        """

    # ------------------------------------------------------------------
    def __init__(self, dim_x=1, dim_y=1, model_type="nonlinear", augmented=False):
        super().__init__(dim_x, dim_y, model_type, augmented)
        self._build_symbolic_model()
        self.pairwiseModel = True

    # ------------------------------------------------------------------
    def _build_symbolic_model(self):
        nx, ny, _nz = self.dim_x, self.dim_y, self.dim_xy

        # Real symbol vectors
        self._sx = sp.Matrix([sp.Symbol(f"x{i}", real=True) for i in range(nx)])
        self._sy = sp.Matrix([sp.Symbol(f"y{i}", real=True) for i in range(ny)])
        self._st = sp.Matrix([sp.Symbol(f"t{i}", real=True) for i in range(nx)])
        self._su = sp.Matrix([sp.Symbol(f"u{i}", real=True) for i in range(ny)])

        # Model provided by the subclass
        try:
            self._sgx, self._sgy = self.symbolic_model(
                self._sx, self._sy, self._st, self._su
            )
        except Exception as e:
            raise RuntimeError(
                f"[{self.__class__.__name__}] symbolic_model() failed — "
                f"check parameter initialization order and SymPy expressions.\n"
                f"Cause: {type(e).__name__}: {e}"
            ) from e

        # Shape validation
        if not isinstance(self._sgx, sp.Matrix) or self._sgx.shape != (nx, 1):
            raise ValueError(
                f"symbolic_model() must return sgx with shape ({nx}, 1), "
                f"got {getattr(self._sgx, 'shape', type(self._sgx))}"
            )
        if not isinstance(self._sgy, sp.Matrix) or self._sgy.shape != (ny, 1):
            raise ValueError(
                f"symbolic_model() must return sgy with shape ({ny}, 1), "
                f"got {getattr(self._sgy, 'shape', type(self._sgy))}"
            )

        # g = [gx ; gy] — full vector for the Jacobians
        sg = self._sgx.col_join(self._sgy)  # (nz, 1)
        sz = self._sx.col_join(self._sy)  # (nz, 1) — augmented state z
        sn = self._st.col_join(self._su)  # (nz, 1) — augmented noise

        # Direct symbolic Jacobians (no chain rule)
        self._sAn = sg.jacobian(sz)  # (nz, nz) : dg/dz
        self._sBn = sg.jacobian(sn)  # (nz, nz) : dg/d_noise

        # Symbol tuples for lambdify
        all_syms = tuple(self._sx) + tuple(self._sy) + tuple(self._st) + tuple(self._su)

        # NumPy compilation.
        # Note: if an expression is constant (e.g. Bn = I for additive noise),
        # lambdify returns an ndarray instead of a callable.
        # We normalise via _wrap_lambdify to guarantee a callable in all cases.
        self._gx_num = self._wrap_lambdify(sp.lambdify(all_syms, self._sgx, "numpy"))
        self._gy_num = self._wrap_lambdify(sp.lambdify(all_syms, self._sgy, "numpy"))
        self._An_num = self._wrap_lambdify(sp.lambdify(all_syms, self._sAn, "numpy"))
        self._Bn_num = self._wrap_lambdify(sp.lambdify(all_syms, self._sBn, "numpy"))

    # ------------------------------------------------------------------
    # Internal numerical evaluations
    # ------------------------------------------------------------------

    def _args(self, x, y, t, u, i=None):
        """Builds the argument tuple for lambdify at index i (batch) or 2D."""
        if i is None:
            return tuple(x[:, 0]) + tuple(y[:, 0]) + tuple(t[:, 0]) + tuple(u[:, 0])
        return (
            tuple(x[i, :, 0])
            + tuple(y[i, :, 0])
            + tuple(t[i, :, 0])
            + tuple(u[i, :, 0])
        )

    def _eval_gx(self, x, y, t, u):
        """
        Evaluates gx(x, y, t, u) numerically.
          2D  : x(dim_x,1), y(dim_y,1), ...  → (dim_x, 1)
          3D  : x(N,dim_x,1), ...             → (N, dim_x, 1)
        """
        try:
            with np.errstate(all="raise"):
                if x.ndim == 2:
                    return np.array(
                        self._gx_num(*self._args(x, y, t, u)), dtype=float
                    ).reshape(self.dim_x, 1)
                N = x.shape[0]
                out = np.empty((N, self.dim_x, 1))
                for i in range(N):
                    out[i] = np.array(
                        self._gx_num(*self._args(x, y, t, u, i)), dtype=float
                    ).reshape(self.dim_x, 1)
                return out
        except FloatingPointError as e:
            raise NumericalError(
                f"[{self.__class__.__name__}] _eval_gx: numerical error at x={x}, y={y}: {e}"
            ) from e
        except (ValueError, IndexError) as e:
            raise NumericalError(
                f"[{self.__class__.__name__}] _eval_gx: shape error at x={x}, y={y}: {e}"
            ) from e

    def _eval_gy(self, x, y, t, u):
        """
        Evaluates gy(x, y, t, u) numerically.
          2D  : → (dim_y, 1)
          3D  : → (N, dim_y, 1)
        """
        try:
            with np.errstate(all="raise"):
                if x.ndim == 2:
                    return np.array(
                        self._gy_num(*self._args(x, y, t, u)), dtype=float
                    ).reshape(self.dim_y, 1)
                N = x.shape[0]
                out = np.empty((N, self.dim_y, 1))
                for i in range(N):
                    out[i] = np.array(
                        self._gy_num(*self._args(x, y, t, u, i)), dtype=float
                    ).reshape(self.dim_y, 1)
                return out
        except FloatingPointError as e:
            raise NumericalError(
                f"[{self.__class__.__name__}] _eval_gy: numerical error at x={x}, y={y}: {e}"
            ) from e
        except (ValueError, IndexError) as e:
            raise NumericalError(
                f"[{self.__class__.__name__}] _eval_gy: shape error at x={x}, y={y}: {e}"
            ) from e

    def _eval_An(self, x, y, t, u):
        """
        Evaluates An = dg/dz numerically.
          2D  : → (dim_xy, dim_xy)
          3D  : → (N, dim_xy, dim_xy)
        """
        try:
            with np.errstate(all="raise"):
                nz = self.dim_xy
                if x.ndim == 2:
                    return np.array(
                        self._An_num(*self._args(x, y, t, u)), dtype=float
                    ).reshape(nz, nz)
                N = x.shape[0]
                out = np.empty((N, nz, nz))
                for i in range(N):
                    out[i] = np.array(
                        self._An_num(*self._args(x, y, t, u, i)), dtype=float
                    ).reshape(nz, nz)
                return out
        except FloatingPointError as e:
            raise NumericalError(
                f"[{self.__class__.__name__}] _eval_An: numerical error at x={x}, y={y}: {e}"
            ) from e
        except (ValueError, IndexError) as e:
            raise NumericalError(
                f"[{self.__class__.__name__}] _eval_An: shape error at x={x}, y={y}: {e}"
            ) from e

    def _eval_Bn(self, x, y, t, u):
        """
        Evaluates Bn = dg/d_noise numerically.
          2D  : → (dim_xy, dim_xy)
          3D  : → (N, dim_xy, dim_xy)
        """
        try:
            with np.errstate(all="raise"):
                nz = self.dim_xy
                if x.ndim == 2:
                    return np.array(
                        self._Bn_num(*self._args(x, y, t, u)), dtype=float
                    ).reshape(nz, nz)
                N = x.shape[0]
                out = np.empty((N, nz, nz))
                for i in range(N):
                    out[i] = np.array(
                        self._Bn_num(*self._args(x, y, t, u, i)), dtype=float
                    ).reshape(nz, nz)
                return out
        except FloatingPointError as e:
            raise NumericalError(
                f"[{self.__class__.__name__}] _eval_Bn: numerical error at x={x}, y={y}: {e}"
            ) from e
        except (ValueError, IndexError) as e:
            raise NumericalError(
                f"[{self.__class__.__name__}] _eval_Bn: shape error at x={x}, y={y}: {e}"
            ) from e

    # ------------------------------------------------------------------
    # Interfaces called by _g and jacobiens_g (BaseModelNonLinear)
    # ------------------------------------------------------------------

    def _gx(self, x, y, t, u, dt):
        return self._eval_gx(x, y, t, u)

    def _gy(self, x, y, t, u, dt):
        return self._eval_gy(x, y, t, u)

    def _jacobiens_g(self, x, y, t, u, dt):
        """
        Direct Jacobians of g(z, noise) = [gx(x,y,t,u) ; gy(x,y,t,u)].

            An = dg/dz     (dim_xy, dim_xy)   evaluated at (x, y, t, u)
            Bn = dg/dnoise (dim_xy, dim_xy)   evaluated at (x, y, t, u)

        No chain rule: gx and gy are evaluated at the same point (x, y).
        """
        # The _eval_* methods catch FloatingPointError → NumericalError upstream.
        # We simply let NumericalError propagate without intercepting it.
        An = self._eval_An(x, y, t, u)
        Bn = self._eval_Bn(x, y, t, u)
        return An, Bn

    # ------------------------------------------------------------------
    def _g(self, x, y, t, u, dt):

        if __debug__:
            assert isinstance(dt, (float, int))
            if x.ndim == 2:
                assert all(a.shape == (self.dim_x, 1) for a in (x, t))
                assert all(a.shape == (self.dim_y, 1) for a in (y, u))
            else:
                assert all(
                    a.ndim == 3 and a.shape[1:] == (self.dim_x, 1) for a in (x, t)
                )
                assert all(
                    a.ndim == 3 and a.shape[1:] == (self.dim_y, 1) for a in (y, u)
                )
                assert x.shape[0] == y.shape[0] == t.shape[0] == u.shape[0]

        try:
            gx_val = self._gx(x, y, t, u, dt)
            gy_val = self._gy(x, y, t, u, dt)
            if x.ndim == 2:
                return np.vstack((gx_val, gy_val))
            return np.concatenate((gx_val, gy_val), axis=1)
        except NumericalError:
            raise
        except ValueError as e:
            raise NumericalError(
                f"[{self.__class__.__name__}] _g: shape mismatch during stack: {e}"
            ) from e

    # ------------------------------------------------------------------
    def latex_model(self) -> str:
        """
        Returns a LaTeX representation of the nonlinear state-space model.

        Typographic conventions:
          - scalar (dim=1)          : plain italic     x, y
          - vector (dim>1)          : bold lowercase   \\mathbf{x}, ...
          - noise                   : v^x (state), v^y (observation)
          - Jacobian matrices       : bold uppercase   \\mathbf{A}_n, \\mathbf{B}_n
          - Gaussian noise          : (v^x, v^y) ~ N(0, Q)
        """
        nx, ny = self.dim_x, self.dim_y

        # ------------------------------------------------------------------
        # LaTeX names for noise: v^x_i / v^y_i (or v^x / v^y if dim=1)
        # ------------------------------------------------------------------
        def _vx_name(i: int) -> str:
            return "v^x" if nx == 1 else rf"v^x_{i}"

        def _vy_name(i: int) -> str:
            return "v^y" if ny == 1 else rf"v^y_{i}"

        # ------------------------------------------------------------------
        # Construction of the substitution dictionary sym → LaTeX symbol
        # dim=1 : x0 → x,  t0 → v^x  (scalar, no index)
        # dim>1 : x0 → x_0, t0 → v^x_0, ... with bold if vector
        # ------------------------------------------------------------------
        def _sub_dict(syms, base: str, bold: bool, names=None) -> dict:
            subs = {}
            for i, s in enumerate(syms):
                if names is not None:
                    latex_name = names[i]
                elif len(syms) == 1:
                    latex_name = rf"\mathbf{{{base}}}" if bold else base
                else:
                    latex_name = rf"\mathbf{{{base}}}_{i}" if bold else rf"{base}_{i}"
                subs[s] = sp.Symbol(latex_name, real=True)
            return subs

        bold_x = nx > 1
        bold_y = ny > 1

        vx_names = [_vx_name(i) for i in range(nx)]
        vy_names = [_vy_name(i) for i in range(ny)]

        subs = {}
        subs.update(_sub_dict(list(self._sx), "x", bold=bold_x))
        subs.update(_sub_dict(list(self._sy), "y", bold=bold_y))
        subs.update(_sub_dict(list(self._st), "t", bold=False, names=vx_names))
        subs.update(_sub_dict(list(self._su), "u", bold=False, names=vy_names))

        def _apply(mat: sp.Matrix) -> sp.Matrix:
            return sp.Matrix(mat.shape[0], mat.shape[1], [e.subs(subs) for e in mat])

        sgx_s = _apply(self._sgx)
        sgy_s = _apply(self._sgy)
        sAn_s = _apply(self._sAn)
        sBn_s = _apply(self._sBn)

        # ------------------------------------------------------------------
        # Left-hand side names
        # ------------------------------------------------------------------
        x_n = r"\mathbf{x}" if bold_x else "x"
        y_n = r"\mathbf{y}" if bold_y else "y"
        vx_n = r"\mathbf{v}^x" if bold_x else "v^x"
        vy_n = r"\mathbf{v}^y" if bold_y else "v^y"
        z_n = r"\mathbf{z}"
        v_n = r"\mathbf{v}"
        An_n = r"\mathbf{A}_n"
        Bn_n = r"\mathbf{B}_n"
        Q_n = r"\mathcal{Q}"

        # ------------------------------------------------------------------
        # mQ formatting (2 decimal places)
        # ------------------------------------------------------------------
        def _np_to_sp(M: np.ndarray) -> sp.Matrix:
            return sp.Matrix(
                M.shape[0],
                M.shape[1],
                [sp.Float(round(float(v), 2)) for v in M.ravel()],
            )

        mQ_sp = _np_to_sp(self.mQ)

        # ------------------------------------------------------------------
        # Rendering of gx / gy: scalar without brackets, vector with ()
        # Reorder to put noise last via sp.Add.
        # ------------------------------------------------------------------
        import re

        def _fix_latex(s: str) -> str:
            """1.0 \\cdot 10^{-k}  →  10^{-k}  (with or without spaces around \\cdot)"""
            return re.sub(r"1\.0\s*\\cdot\s*", "", s)

        def _reorder_noise_last(expr: sp.Expr) -> str:
            """
            Renders a LaTeX expression with noise terms forced to the end.
            Works around sp.Add canonical sorting by substituting noise=0
            to extract the deterministic part, then concatenates manually.
            Only reorders when noise is purely additive (noise term depends
            exclusively on noise symbols). For multiplicative noise (e.g.
            x*exp(A + vx)), falls back to sp.latex(expr) directly.
            """
            noise_syms = {sp.Symbol(n, real=True) for n in vx_names + vy_names}
            zero_subs = dict.fromkeys(noise_syms, 0)
            det = expr.subs(zero_subs)  # deterministic part
            noise = sp.expand(expr - det)  # noise contribution

            if noise == 0:
                return sp.latex(det)
            if det == 0:
                return sp.latex(noise)

            # For multiplicative noise (e.g. x*exp(A+vx)), noise contains
            # state variables — not purely additive. Render the full expression.
            if not noise.free_symbols.issubset(noise_syms):
                return sp.latex(expr)

            noise_lat = sp.latex(noise)
            # If the noise term starts with '-', no '+' is needed
            sep = " " if noise_lat.startswith("-") else " + "
            return sp.latex(det) + sep + noise_lat

        def _lat(mat: sp.Matrix) -> str:
            if mat.shape == (1, 1):
                return _reorder_noise_last(mat[0, 0])
            parts = [_reorder_noise_last(e) for e in mat]
            # Manual column rendering with parentheses
            rows = r" \\ ".join(parts)
            return rf"\begin{{pmatrix}} {rows} \end{{pmatrix}}"

        # ------------------------------------------------------------------
        # Assembly
        # ------------------------------------------------------------------
        lines = [
            r"\begin{aligned}",
            # ── Dynamics
            rf"  g_x\!\left({x_n},\,{y_n},\,{vx_n}\right)"
            rf" &= {_lat(sgx_s)} \\[6pt]",
            # ── Observation
            rf"  g_y\!\left({x_n},\,{y_n},\,{vy_n}\right)"
            rf" &= {_lat(sgy_s)} \\[6pt]",
            # ── Noise distribution (just after the equations)
            rf"  {v_n} = ({vx_n},\,{vy_n})"
            rf" &\sim \mathcal{{N}}\!\left(0,\; {Q_n}\right), \qquad"
            rf" {Q_n} = {sp.latex(mQ_sp)} \\[12pt]",
            # ── State Jacobian
            rf"  {An_n} &= \frac{{\partial\, g}}{{\partial\, {z_n}}}"
            rf" = {sp.latex(sAn_s)} \\[6pt]",
            # ── Noise Jacobian
            rf"  {Bn_n} &= \frac{{\partial\, g}}{{\partial\, {v_n}}}"
            rf" = {sp.latex(sBn_s)}",
            r"\end{aligned}",
        ]

        return _fix_latex("\n".join(lines))
