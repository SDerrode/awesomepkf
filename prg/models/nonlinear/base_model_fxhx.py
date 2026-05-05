from abc import ABC, abstractmethod

import numpy as np
import sympy as sp

from prg.models.nonLinear._latex_helpers import _LatexBuilder
from prg.models.nonLinear.base_model_nonLinear import BaseModelNonLinear
from prg.utils.exceptions import NumericalError

__all__ = ["BaseModelFxHx"]


class BaseModelFxHx(BaseModelNonLinear, ABC):
    """
    Parent class for models defined symbolically via symbolic_model().

    Handles dim_x >= 1 and dim_y >= 1.

    The subclass must implement:

        def symbolic_model(self, sx, st, su):
            ...
            return sfx, shx

    where:
        sx  : sp.Matrix(dim_x, 1) — symbols x0 .. x_{dim_x-1}
        st  : sp.Matrix(dim_x, 1) — symbols t0 .. t_{dim_x-1}
        su  : sp.Matrix(dim_y, 1) — symbols u0 .. u_{dim_y-1}
        sfx : sp.Matrix(dim_x, 1) — expression of the transition f(x, t)
        shx : sp.Matrix(dim_y, 1) — expression of the observation h(x, u)

    Example dim_x=1, dim_y=1:

        def symbolic_model(self, sx, st, su):
            x, t, u = sx[0], st[0], su[0]
            sfx = 0.9 * x - 0.6 * x**3 + t
            shx = x + u
            return sp.Matrix([sfx]), sp.Matrix([shx])

    Example dim_x=2, dim_y=1:

        def symbolic_model(self, sx, st, su):
            x1, x2 = sx[0], sx[1]
            t1, t2 = st[0], st[1]
            u      = su[0]
            sfx = sp.Matrix([(1 - k)*x1 + d*x2 + t1,
                              x2 - d*(a*sp.sin(x1) + b*x2) + t2])
            shx = sp.Matrix([x1**2 / (1 + x1**2) + g*sp.sin(x2) + u])
            return sfx, shx
    """

    # ------------------------------------------------------------------
    @abstractmethod
    def symbolic_model(self, sx, st, su):
        """
        To be implemented in the subclass.

        Parameters
        ----------
        sx : sp.Matrix(dim_x, 1)  — state symbols         x0 .. x_{dim_x-1}
        st : sp.Matrix(dim_x, 1)  — state noise symbols   t0 .. t_{dim_x-1}
        su : sp.Matrix(dim_y, 1)  — obs. noise symbols    u0 .. u_{dim_y-1}

        Returns
        --------
        sfx : sp.Matrix(dim_x, 1) — transition  f(x, t)
        shx : sp.Matrix(dim_y, 1) — observation h(x, u)
        """

    # ------------------------------------------------------------------
    def __init__(self, dim_x=1, dim_y=1, model_type="nonlinear", augmented=False):
        super().__init__(dim_x, dim_y, model_type, augmented)
        self._build_symbolic_model()
        self.pairwiseModel = False

    # ------------------------------------------------------------------
    def _args_fx(self, x, t, i=None):
        """Builds the argument tuple (x, t) for lambdify."""
        if i is None:
            return tuple(x[:, 0]) + tuple(t[:, 0])
        return tuple(x[i, :, 0]) + tuple(t[i, :, 0])

    def _args_hx(self, x, u, i=None):
        """Builds the argument tuple (x, u) for lambdify."""
        if i is None:
            return tuple(x[:, 0]) + tuple(u[:, 0])
        return tuple(x[i, :, 0]) + tuple(u[i, :, 0])

    # ------------------------------------------------------------------
    def _build_symbolic_model(self):
        nx, ny = self.dim_x, self.dim_y

        # Real symbol vectors
        self._sx = sp.Matrix([sp.Symbol(f"x{i}", real=True) for i in range(nx)])
        self._st = sp.Matrix([sp.Symbol(f"t{i}", real=True) for i in range(nx)])
        self._su = sp.Matrix([sp.Symbol(f"u{i}", real=True) for i in range(ny)])

        # Model provided by the subclass
        try:
            self._sfx, self._shx = self.symbolic_model(self._sx, self._st, self._su)
        except Exception as e:
            raise RuntimeError(
                f"[{self.__class__.__name__}] symbolic_model() failed — "
                f"check parameter initialization order and SymPy expressions.\n"
                f"Cause: {type(e).__name__}: {e}"
            ) from e

        # Shape validation
        if not isinstance(self._sfx, sp.Matrix) or self._sfx.shape != (nx, 1):
            raise ValueError(
                f"symbolic_model() must return sfx with shape ({nx}, 1), "
                f"got {getattr(self._sfx, 'shape', type(self._sfx))}"
            )
        if not isinstance(self._shx, sp.Matrix) or self._shx.shape != (ny, 1):
            raise ValueError(
                f"symbolic_model() must return shx with shape ({ny}, 1), "
                f"got {getattr(self._shx, 'shape', type(self._shx))}"
            )

        # Symbolic Jacobians
        self._sA = self._sfx.jacobian(self._sx)  # (nx, nx) : df/dx
        self._sH = self._shx.jacobian(self._sx)  # (ny, nx) : dh/dx

        # Symbol tuples for lambdify
        sx_t = tuple(self._sx)
        st_t = tuple(self._st)
        su_t = tuple(self._su)

        # NumPy compilation.
        # Note: if an expression is constant (e.g. H = [0,1] for hx = x[-1]),
        # lambdify returns an ndarray instead of a callable.
        # We normalise via _wrap_lambdify to guarantee a callable in all cases.
        self._fx_num = self._wrap_lambdify(sp.lambdify(sx_t + st_t, self._sfx, "numpy"))
        self._hx_num = self._wrap_lambdify(sp.lambdify(sx_t + su_t, self._shx, "numpy"))
        self._A_num = self._wrap_lambdify(sp.lambdify(sx_t, self._sA, "numpy"))
        self._H_num = self._wrap_lambdify(sp.lambdify(sx_t, self._sH, "numpy"))

    # ------------------------------------------------------------------
    # Internal numerical evaluations
    # ------------------------------------------------------------------

    def _normalise_aux(self, aux, dim, x_ndim, N=None):
        """Broadcast ``aux`` (scalar / (dim,1) / (N,dim,1)) to the expected shape."""
        if x_ndim == 2:
            if np.ndim(aux) == 0:
                return np.full((dim, 1), aux)
            return np.asarray(aux)
        if np.ndim(aux) == 0:
            return np.full((N, dim, 1), aux)
        return np.broadcast_to(np.asarray(aux), (N, dim, 1))

    def _eval_fx(self, x, t):
        """
        Evaluates f(x, t) numerically.
        x : (dim_x, 1)     t : (dim_x, 1) or scalar  → returns (dim_x, 1)
        x : (N, dim_x, 1)  t : (N, dim_x, 1) or scalar → returns (N, dim_x, 1)
        """
        N = None if x.ndim == 2 else x.shape[0]
        t_norm = self._normalise_aux(t, self.dim_x, x.ndim, N)
        return self._safe_eval(
            "_eval_fx",
            self._fx_num,
            lambda i: self._args_fx(x, t_norm, i),
            (self.dim_x, 1),
            x,
        )

    def _eval_hx(self, x, u):
        """
        Evaluates h(x, u) numerically.
        x : (dim_x, 1),    u : (dim_y, 1) or scalar   → returns (dim_y, 1)
        x : (N, dim_x, 1), u : (N, dim_y, 1) or scalar → returns (N, dim_y, 1)
        """
        N = None if x.ndim == 2 else x.shape[0]
        u_norm = self._normalise_aux(u, self.dim_y, x.ndim, N)
        return self._safe_eval(
            "_eval_hx",
            self._hx_num,
            lambda i: self._args_hx(x, u_norm, i),
            (self.dim_y, 1),
            x,
        )

    def _eval_A(self, x):
        """
        Evaluates df/dx numerically.
          x : (dim_x, 1)     → returns (dim_x, dim_x)
          x : (N, dim_x, 1)  → returns (N, dim_x, dim_x)
        """
        return self._safe_eval(
            "_eval_A",
            self._A_num,
            lambda i: tuple(x[:, 0]) if i is None else tuple(x[i, :, 0]),
            (self.dim_x, self.dim_x),
            x,
        )

    def _eval_H(self, x):
        """
        Evaluates dh/dx numerically.
          x : (dim_x, 1)     → returns (dim_y, dim_x)
          x : (N, dim_x, 1)  → returns (N, dim_y, dim_x)
        """
        return self._safe_eval(
            "_eval_H",
            self._H_num,
            lambda i: tuple(x[:, 0]) if i is None else tuple(x[i, :, 0]),
            (self.dim_y, self.dim_x),
            x,
        )

    # ------------------------------------------------------------------
    # _fx / _hx interfaces called by _g
    # ------------------------------------------------------------------

    def _fx(self, x, t, dt):
        return self._eval_fx(x, t)

    def _hx(self, x, u, dt):
        return self._eval_hx(x, u)

    # ------------------------------------------------------------------
    def _jacobiens_g(self, x, y, t, u, dt):
        """
        Computes the Jacobians of g(z) = [f(x,t) ; h(f(x,t), u)] with respect
        to z = [x ; y] and the noise noise_z = [t ; u].

        Block structure (nz = dim_x + dim_y):

            An = dg/dz    = [ df/dx              0     ]
                            [ dh/dx @ df/dx       0    ]

            Bn = dg/dnoise = [ I_nx        0      ]
                             [ dh/dx       I_ny   ]

        dh/dx is evaluated at f(x, t) (chain rule).
        """
        # The _eval_* methods catch FloatingPointError → NumericalError upstream.
        # We simply let NumericalError propagate without intercepting it.
        nx, ny, nz = self.dim_x, self.dim_y, self.dim_xy

        if x.ndim == 2:
            fx_val = self._eval_fx(x, t)  # (nx, 1)
            dfdx = self._eval_A(x)  # (nx, nx)
            dhdx = self._eval_H(fx_val)  # (ny, nx) evaluated at f(x)

            An = np.zeros((nz, nz))
            An[:nx, :nx] = dfdx
            An[nx:, :nx] = dhdx @ dfdx  # chain rule

            Bn = np.zeros((nz, nz))
            Bn[:nx, :nx] = np.eye(nx)
            Bn[nx:, :nx] = dhdx
            Bn[nx:, nx:] = np.eye(ny)

        else:
            N = x.shape[0]
            fx_val = self._eval_fx(x, t)  # (N, nx, 1)
            dfdx = self._eval_A(x)  # (N, nx, nx)
            dhdx = self._eval_H(fx_val)  # (N, ny, nx)

            An = np.zeros((N, nz, nz))
            An[:, :nx, :nx] = dfdx
            An[:, nx:, :nx] = np.einsum("nij,njk->nik", dhdx, dfdx)

            Bn = np.zeros((N, nz, nz))
            Bn[:, :nx, :nx] = np.tile(np.eye(nx), (N, 1, 1))
            Bn[:, nx:, :nx] = dhdx
            Bn[:, nx:, nx:] = np.tile(np.eye(ny), (N, 1, 1))

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
            fx_val = self._fx(x, t, dt)
            hx_val = self._hx(fx_val, u, dt)
            if x.ndim == 2:
                return np.vstack((fx_val, hx_val))
            return np.concatenate((fx_val, hx_val), axis=1)
        except NumericalError:
            raise
        except ValueError as e:
            raise NumericalError(
                f"[{self.__class__.__name__}] _g: shape mismatch during stack: {e}"
            ) from e

    # ------------------------------------------------------------------
    def latex_model(self) -> str:
        """
        Returns a LaTeX representation of the nonlinear state-space model f/h.

        Typographic conventions:
          - scalar (dim=1)          : plain italic     x, y
          - vector (dim>1)          : bold lowercase   \\mathbf{x}, ...
          - noise                   : v^x (state), v^y (observation)
          - Jacobian matrices       : bold uppercase   \\mathbf{A}, \\mathbf{H}
          - Gaussian noise          : v = (v^x, v^y) ~ N(0, Q)
        """
        b = _LatexBuilder(self.dim_x, self.dim_y, self.mQ)

        # f, A, H are all expressed in terms of x → use the same substitutions
        subs_x = {
            **b.sub_dict(list(self._sx), "x", bold=b.bold_x),
            **b.sub_dict(list(self._st), "t", bold=False, names=b.vx_names),
            **b.sub_dict(list(self._su), "u", bold=False, names=b.vy_names),
        }

        # h is evaluated at y = f(x), so x-symbols in shx are renamed to y
        subs_y = {
            **b.sub_dict(list(self._sx), "y", bold=b.bold_y),
            **b.sub_dict(list(self._st), "t", bold=False, names=b.vx_names),
            **b.sub_dict(list(self._su), "u", bold=False, names=b.vy_names),
        }

        sfx_s = b.apply(self._sfx, subs_x)
        shx_s = b.apply(self._shx, subs_y)
        sA_s = b.apply(self._sA, subs_x)
        sH_s = b.apply(self._sH, subs_x)

        lines = [
            r"\begin{align}",
            rf"  f\!\left({b.x_n},\,{b.vx_n}\right) &= {b.lat(sfx_s)} \\[6pt]",
            rf"  h\!\left({b.y_n},\,{b.vy_n}\right) &= {b.lat(shx_s)} \\[6pt]",
            b.noise_distribution_line(),
            rf"  \dfrac{{\partial\, f}}{{\partial\, {b.x_n}}}"
            rf" &= {b.lat_jac(sA_s)} \\[10pt]",
            rf"  \dfrac{{\partial\, h}}{{\partial\, {b.y_n}}} &= {b.lat_jac(sH_s)}",
            r"\end{align}",
        ]
        return b.fix_latex("\n".join(lines))
