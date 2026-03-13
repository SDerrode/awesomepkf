#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod

import numpy as np
import sympy as sp

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

    def _eval_fx(self, x, t):
        """
        Evaluates f(x, t) numerically.
        x : (dim_x, 1)     t : (dim_x, 1) or scalar  → returns (dim_x, 1)
        x : (N, dim_x, 1)  t : (N, dim_x, 1) or scalar → returns (N, dim_x, 1)
        """
        try:
            with np.errstate(all="raise"):
                if x.ndim == 2:
                    # Normalise t → (dim_x, 1)
                    t_norm = (
                        np.full((self.dim_x, 1), t)
                        if np.ndim(t) == 0
                        else np.asarray(t)
                    )
                    return np.array(
                        self._fx_num(*self._args_fx(x, t_norm)), dtype=float
                    ).reshape(self.dim_x, 1)

                N = x.shape[0]
                # Normalise t → (N, dim_x, 1)
                if np.ndim(t) == 0:
                    t_norm = np.full((N, self.dim_x, 1), t)
                else:
                    t_norm = np.broadcast_to(np.asarray(t), (N, self.dim_x, 1))

                out = np.empty((N, self.dim_x, 1))
                for i in range(N):
                    out[i] = np.array(
                        self._fx_num(*self._args_fx(x, t_norm, i)), dtype=float
                    ).reshape(self.dim_x, 1)
                return out

        except FloatingPointError as e:
            raise NumericalError(
                f"[{self.__class__.__name__}] _eval_fx: numerical error at x={x}, t={t}: {e}"
            ) from e
        except (ValueError, IndexError) as e:
            raise NumericalError(
                f"[{self.__class__.__name__}] _eval_fx: shape error at x={x}, t={t}: {e}"
            ) from e

    def _eval_hx(self, x, u):
        """
        Evaluates h(x, u) numerically.
        x : (dim_x, 1),    u : (dim_y, 1) or scalar   → returns (dim_y, 1)
        x : (N, dim_x, 1), u : (N, dim_y, 1) or scalar → returns (N, dim_y, 1)
        """
        try:
            with np.errstate(all="raise"):
                if x.ndim == 2:
                    u_norm = (
                        np.full((self.dim_y, 1), u)
                        if np.ndim(u) == 0
                        else np.asarray(u)
                    )
                    return np.array(
                        self._hx_num(*self._args_hx(x, u_norm)), dtype=float
                    ).reshape(self.dim_y, 1)

                N = x.shape[0]
                if np.ndim(u) == 0:
                    u_norm = np.full((N, self.dim_y, 1), u)
                else:
                    u_norm = np.broadcast_to(np.asarray(u), (N, self.dim_y, 1))

                out = np.empty((N, self.dim_y, 1))
                for i in range(N):

                    out[i] = np.array(
                        self._hx_num(*self._args_hx(x, u_norm, i)), dtype=float
                    ).reshape(self.dim_y, 1)
                return out

        except FloatingPointError as e:
            raise NumericalError(
                f"[{self.__class__.__name__}] _eval_hx: numerical error at x={x}, u={u}: {e}"
            ) from e
        except (ValueError, IndexError) as e:
            raise NumericalError(
                f"[{self.__class__.__name__}] _eval_hx: shape error at x={x}, u={u}: {e}"
            ) from e

    def _eval_A(self, x):
        """
        Evaluates df/dx numerically.
          x : (dim_x, 1)     → returns (dim_x, dim_x)
          x : (N, dim_x, 1)  → returns (N, dim_x, dim_x)
        """
        try:
            with np.errstate(all="raise"):
                if x.ndim == 2:
                    return np.array(self._A_num(*tuple(x[:, 0])), dtype=float).reshape(
                        self.dim_x, self.dim_x
                    )
                N = x.shape[0]
                out = np.empty((N, self.dim_x, self.dim_x))
                for i in range(N):
                    out[i] = np.array(
                        self._A_num(*tuple(x[i, :, 0])), dtype=float
                    ).reshape(self.dim_x, self.dim_x)
                return out
        except FloatingPointError as e:
            raise NumericalError(
                f"[{self.__class__.__name__}] _eval_A: numerical error at x={x}: {e}"
            ) from e
        except (ValueError, IndexError) as e:
            raise NumericalError(
                f"[{self.__class__.__name__}] _eval_A: shape error at x={x}: {e}"
            ) from e

    def _eval_H(self, x):
        """
        Evaluates dh/dx numerically.
          x : (dim_x, 1)     → returns (dim_y, dim_x)
          x : (N, dim_x, 1)  → returns (N, dim_y, dim_x)
        """
        try:
            with np.errstate(all="raise"):
                if x.ndim == 2:
                    return np.array(self._H_num(*tuple(x[:, 0])), dtype=float).reshape(
                        self.dim_y, self.dim_x
                    )
                N = x.shape[0]
                out = np.empty((N, self.dim_y, self.dim_x))
                for i in range(N):
                    out[i] = np.array(
                        self._H_num(*tuple(x[i, :, 0])), dtype=float
                    ).reshape(self.dim_y, self.dim_x)
                return out
        except FloatingPointError as e:
            raise NumericalError(
                f"[{self.__class__.__name__}] _eval_H: numerical error at x={x}: {e}"
            ) from e
        except (ValueError, IndexError) as e:
            raise NumericalError(
                f"[{self.__class__.__name__}] _eval_H: shape error at x={x}: {e}"
            ) from e

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
            else:
                return np.concatenate((fx_val, hx_val), axis=1)
        except NumericalError:
            raise
        except ValueError as e:
            raise NumericalError(
                f"[{self.__class__.__name__}] _g: shape mismatch during stack: {e}"
            ) from e

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
        nx, ny = self.dim_x, self.dim_y
        bold_x = nx > 1
        bold_y = ny > 1

        # ------------------------------------------------------------------
        # LaTeX names for noise
        # ------------------------------------------------------------------
        def _vx_name(i: int) -> str:
            return "v^x" if nx == 1 else rf"v^x_{i}"

        def _vy_name(i: int) -> str:
            return "v^y" if ny == 1 else rf"v^y_{i}"

        vx_names = [_vx_name(i) for i in range(nx)]
        vy_names = [_vy_name(i) for i in range(ny)]

        # ------------------------------------------------------------------
        # Substitution dictionary sym → clean LaTeX symbol
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

        subs = {}
        subs.update(_sub_dict(list(self._sx), "x", bold=bold_x))
        subs.update(_sub_dict(list(self._st), "t", bold=False, names=vx_names))
        subs.update(_sub_dict(list(self._su), "u", bold=False, names=vy_names))

        def _apply(mat: sp.Matrix) -> sp.Matrix:
            return sp.Matrix(mat.shape[0], mat.shape[1], [e.subs(subs) for e in mat])

        sfx_s = _apply(self._sfx)

        # For h: substitute x by y in the expression (h is evaluated at f(x)=y_{k})
        subs_xy = {}
        subs_xy.update(_sub_dict(list(self._sx), "y", bold=bold_y))
        subs_xy.update(_sub_dict(list(self._st), "t", bold=False, names=vx_names))
        subs_xy.update(_sub_dict(list(self._su), "u", bold=False, names=vy_names))

        def _apply_xy(mat: sp.Matrix) -> sp.Matrix:
            return sp.Matrix(mat.shape[0], mat.shape[1], [e.subs(subs_xy) for e in mat])

        shx_s = _apply_xy(self._shx)
        sA_s = _apply(self._sA)
        sH_s = _apply(self._sH)

        # ------------------------------------------------------------------
        # Left-hand side names
        # ------------------------------------------------------------------
        x_n = r"\mathbf{x}" if bold_x else "x"
        y_n = r"\mathbf{y}" if bold_y else "y"
        vx_n = r"\mathbf{v}^x" if bold_x else "v^x"
        vy_n = r"\mathbf{v}^y" if bold_y else "v^y"
        v_n = r"\mathbf{v}"
        A_n = r"\mathbf{A}"
        H_n = r"\mathbf{H}"
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
        # LaTeX rendering with noise forced last
        # Works around sp.Add canonical sorting by isolating the deterministic part.
        # ------------------------------------------------------------------
        import re

        def _fix_latex(s: str) -> str:
            """1.0 \\cdot 10^{-k}  →  10^{-k}  (with or without spaces around \\cdot)"""
            return re.sub(r"1\.0\s*\\cdot\s*", "", s)

        def _reorder_noise_last(expr: sp.Expr) -> str:
            zero_subs = {sp.Symbol(n, real=True): 0 for n in vx_names + vy_names}
            det = expr.subs(zero_subs)
            noise = sp.expand(expr - det)
            if noise == 0:
                return sp.latex(det)
            if det == 0:
                return sp.latex(noise)
            noise_lat = sp.latex(noise)
            sep = " " if noise_lat.startswith("-") else " + "
            return sp.latex(det) + sep + noise_lat

        def _lat(mat: sp.Matrix) -> str:
            if mat.shape == (1, 1):
                return _reorder_noise_last(mat[0, 0])
            parts = [_reorder_noise_last(e) for e in mat]
            rows = r" \\ ".join(parts)
            return rf"\begin{{pmatrix}} {rows} \end{{pmatrix}}"

        def _lat_jac(mat: sp.Matrix) -> str:
            """Jacobian rendering: scalar without pmatrix, matrix with."""
            if mat.shape == (1, 1):
                return sp.latex(mat[0, 0])
            return sp.latex(mat)

        # ------------------------------------------------------------------
        # Assembly
        # ------------------------------------------------------------------
        lines = [
            r"\begin{align}",
            # ── Transition
            rf"  f\!\left({x_n},\,{vx_n}\right)" rf" &= {_lat(sfx_s)} \\[6pt]",
            # ── Observation (h evaluated at y = f(x))
            rf"  h\!\left({y_n},\,{vy_n}\right)" rf" &= {_lat(shx_s)} \\[6pt]",
            # ── Noise distribution
            rf"  {v_n} = ({vx_n},\,{vy_n})"
            rf" &\sim \mathcal{{N}}\!\left(0,\; {Q_n}\right), \qquad"
            rf" {Q_n} = {sp.latex(mQ_sp)} \\[12pt]",
            # ── Transition Jacobian (dfrac + no pmatrix if scalar)
            rf"  \dfrac{{\partial\, f}}{{\partial\, {x_n}}}"
            rf" &= {_lat_jac(sA_s)} \\[10pt]",
            # ── Observation Jacobian
            rf"  \dfrac{{\partial\, h}}{{\partial\, {y_n}}}" rf" &= {_lat_jac(sH_s)}",
            r"\end{align}",
        ]

        return _fix_latex("\n".join(lines))
