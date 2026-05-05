from abc import ABC, abstractmethod

import numpy as np
import sympy as sp

from prg.models.nonlinear._latex_helpers import _LatexBuilder
from prg.models.nonlinear.base_model_nonlinear import BaseModelNonLinear
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
        return self._safe_eval(
            "_eval_gx", self._gx_num,
            lambda i: self._args(x, y, t, u, i),
            (self.dim_x, 1), x,
        )

    def _eval_gy(self, x, y, t, u):
        """
        Evaluates gy(x, y, t, u) numerically.
          2D  : → (dim_y, 1)
          3D  : → (N, dim_y, 1)
        """
        return self._safe_eval(
            "_eval_gy", self._gy_num,
            lambda i: self._args(x, y, t, u, i),
            (self.dim_y, 1), x,
        )

    def _eval_An(self, x, y, t, u):
        """
        Evaluates An = dg/dz numerically.
          2D  : → (dim_xy, dim_xy)
          3D  : → (N, dim_xy, dim_xy)
        """
        nz = self.dim_xy
        return self._safe_eval(
            "_eval_An", self._An_num,
            lambda i: self._args(x, y, t, u, i),
            (nz, nz), x,
        )

    def _eval_Bn(self, x, y, t, u):
        """
        Evaluates Bn = dg/d_noise numerically.
          2D  : → (dim_xy, dim_xy)
          3D  : → (N, dim_xy, dim_xy)
        """
        nz = self.dim_xy
        return self._safe_eval(
            "_eval_Bn", self._Bn_num,
            lambda i: self._args(x, y, t, u, i),
            (nz, nz), x,
        )

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
        b = _LatexBuilder(self.dim_x, self.dim_y, self.mQ)

        # gx and gy both depend on (x, y, t, u) — single substitution dict
        subs = {
            **b.sub_dict(list(self._sx), "x", bold=b.bold_x),
            **b.sub_dict(list(self._sy), "y", bold=b.bold_y),
            **b.sub_dict(list(self._st), "t", bold=False, names=b.vx_names),
            **b.sub_dict(list(self._su), "u", bold=False, names=b.vy_names),
        }

        sgx_s = b.apply(self._sgx, subs)
        sgy_s = b.apply(self._sgy, subs)
        sAn_s = b.apply(self._sAn, subs)
        sBn_s = b.apply(self._sBn, subs)

        z_n = r"\mathbf{z}"
        An_n = r"\mathbf{A}_n"
        Bn_n = r"\mathbf{B}_n"

        lines = [
            r"\begin{aligned}",
            rf"  g_x\!\left({b.x_n},\,{b.y_n},\,{b.vx_n}\right) &= {b.lat(sgx_s)} \\[6pt]",
            rf"  g_y\!\left({b.x_n},\,{b.y_n},\,{b.vy_n}\right) &= {b.lat(sgy_s)} \\[6pt]",
            b.noise_distribution_line(),
            rf"  {An_n} &= \frac{{\partial\, g}}{{\partial\, {z_n}}}"
            rf" = {sp.latex(sAn_s)} \\[6pt]",
            rf"  {Bn_n} &= \frac{{\partial\, g}}{{\partial\, {b.v_n}}}"
            rf" = {sp.latex(sBn_s)}",
            r"\end{aligned}",
        ]
        return b.fix_latex("\n".join(lines))
