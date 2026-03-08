#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod

import numpy as np
import sympy as sp

from prg.models.nonLinear.base_model_nonLinear import BaseModelNonLinear
from prg.exceptions import NumericalError

__all__ = ["BaseModelFxHx"]


class BaseModelFxHx(BaseModelNonLinear, ABC):
    """
    Classe mère pour les modèles définis symboliquement via symbolic_model().

    Gère dim_x >= 1 et dim_y >= 1.

    La sous-classe doit implémenter :

        def symbolic_model(self, sx, st, su):
            ...
            return sfx, shx

    où :
        sx  : sp.Matrix(dim_x, 1) — symboles x0 .. x_{dim_x-1}
        st  : sp.Matrix(dim_x, 1) — symboles t0 .. t_{dim_x-1}
        su  : sp.Matrix(dim_y, 1) — symboles u0 .. u_{dim_y-1}
        sfx : sp.Matrix(dim_x, 1) — expression de la transition f(x, t)
        shx : sp.Matrix(dim_y, 1) — expression de l'observation h(x, u)

    Exemple dim_x=1, dim_y=1 :

        def symbolic_model(self, sx, st, su):
            x, t, u = sx[0], st[0], su[0]
            sfx = 0.9 * x - 0.6 * x**3 + t
            shx = x + u
            return sp.Matrix([sfx]), sp.Matrix([shx])

    Exemple dim_x=2, dim_y=1 :

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
        À implémenter dans la sous-classe.

        Paramètres
        ----------
        sx : sp.Matrix(dim_x, 1)  — symboles d'état        x0 .. x_{dim_x-1}
        st : sp.Matrix(dim_x, 1)  — symboles bruit d'état  t0 .. t_{dim_x-1}
        su : sp.Matrix(dim_y, 1)  — symboles bruit obs.    u0 .. u_{dim_y-1}

        Retourne
        --------
        sfx : sp.Matrix(dim_x, 1) — transition  f(x, t)
        shx : sp.Matrix(dim_y, 1) — observation h(x, u)
        """

    # ------------------------------------------------------------------
    def __init__(self, dim_x=1, dim_y=1, model_type="nonlinear", augmented=False):
        super().__init__(dim_x, dim_y, model_type, augmented)
        self._build_symbolic_model()

    # ------------------------------------------------------------------
    def _args_fx(self, x, t, i=None):
        """Construit le tuple d'arguments (x, t) pour lambdify."""
        if i is None:
            return tuple(x[:, 0]) + tuple(t[:, 0])
        return tuple(x[i, :, 0]) + tuple(t[i, :, 0])

    def _args_hx(self, x, u, i=None):
        """Construit le tuple d'arguments (x, u) pour lambdify."""
        if i is None:
            return tuple(x[:, 0]) + tuple(u[:, 0])
        return tuple(x[i, :, 0]) + tuple(u[i, :, 0])

    # ------------------------------------------------------------------
    def _build_symbolic_model(self):
        nx, ny = self.dim_x, self.dim_y

        # Vecteurs de symboles réels
        self._sx = sp.Matrix([sp.Symbol(f"x{i}", real=True) for i in range(nx)])
        self._st = sp.Matrix([sp.Symbol(f"t{i}", real=True) for i in range(nx)])
        self._su = sp.Matrix([sp.Symbol(f"u{i}", real=True) for i in range(ny)])

        # Modèle fourni par la sous-classe
        try:
            self._sfx, self._shx = self.symbolic_model(self._sx, self._st, self._su)
        except Exception as e:
            raise RuntimeError(
                f"[{self.__class__.__name__}] symbolic_model() failed — "
                f"check parameter initialization order and SymPy expressions.\n"
                f"Cause: {type(e).__name__}: {e}"
            ) from e

        # Validation des shapes
        if not isinstance(self._sfx, sp.Matrix) or self._sfx.shape != (nx, 1):
            raise ValueError(
                f"symbolic_model() doit retourner sfx de shape ({nx}, 1), "
                f"got {getattr(self._sfx, 'shape', type(self._sfx))}"
            )
        if not isinstance(self._shx, sp.Matrix) or self._shx.shape != (ny, 1):
            raise ValueError(
                f"symbolic_model() doit retourner shx de shape ({ny}, 1), "
                f"got {getattr(self._shx, 'shape', type(self._shx))}"
            )

        # Jacobiennes symboliques
        self._sA = self._sfx.jacobian(self._sx)  # (nx, nx) : df/dx
        self._sH = self._shx.jacobian(self._sx)  # (ny, nx) : dh/dx

        # Tuples de symboles pour lambdify
        sx_t = tuple(self._sx)
        st_t = tuple(self._st)
        su_t = tuple(self._su)

        # Compilation NumPy.
        # Note : si une expression est constante (ex. H = [0,1] pour hx = x[-1]),
        # lambdify retourne un ndarray au lieu d'un callable.
        # On normalise via _wrap_lambdify pour garantir un callable dans tous les cas.
        self._fx_num = self._wrap_lambdify(sp.lambdify(sx_t + st_t, self._sfx, "numpy"))
        self._hx_num = self._wrap_lambdify(sp.lambdify(sx_t + su_t, self._shx, "numpy"))
        self._A_num = self._wrap_lambdify(sp.lambdify(sx_t, self._sA, "numpy"))
        self._H_num = self._wrap_lambdify(sp.lambdify(sx_t, self._sH, "numpy"))

    # ------------------------------------------------------------------
    # Évaluations numériques internes
    # ------------------------------------------------------------------

    def _eval_fx(self, x, t):
        """
        Évalue f(x, t) numériquement.
          x, t : (dim_x, 1)       → retourne (dim_x, 1)
          x, t : (N, dim_x, 1)   → retourne (N, dim_x, 1)
        """
        try:
            with np.errstate(all="raise"):
                if x.ndim == 2:
                    return np.array(
                        self._fx_num(*self._args_fx(x, t)), dtype=float
                    ).reshape(self.dim_x, 1)
                N = x.shape[0]
                out = np.empty((N, self.dim_x, 1))
                for i in range(N):
                    out[i] = np.array(
                        self._fx_num(*self._args_fx(x, t, i)), dtype=float
                    ).reshape(self.dim_x, 1)
                return out
        except FloatingPointError as e:
            raise NumericalError(
                f"[{self.__class__.__name__}] _eval_fx: erreur numérique à x={x}, t={t}: {e}"
            ) from e
        except (ValueError, IndexError) as e:
            raise NumericalError(
                f"[{self.__class__.__name__}] _eval_fx: erreur de shape à x={x}, t={t}: {e}"
            ) from e

    def _eval_hx(self, x, u):
        """
        Évalue h(x, u) numériquement.
          x : (dim_x, 1),  u : (dim_y, 1)     → retourne (dim_y, 1)
          x : (N, dim_x, 1), u : (N, dim_y, 1) → retourne (N, dim_y, 1)
        """
        try:
            with np.errstate(all="raise"):
                if x.ndim == 2:
                    return np.array(
                        self._hx_num(*self._args_hx(x, u)), dtype=float
                    ).reshape(self.dim_y, 1)
                N = x.shape[0]
                out = np.empty((N, self.dim_y, 1))
                for i in range(N):
                    out[i] = np.array(
                        self._hx_num(*self._args_hx(x, u, i)), dtype=float
                    ).reshape(self.dim_y, 1)
                return out
        except FloatingPointError as e:
            raise NumericalError(
                f"[{self.__class__.__name__}] _eval_hx: erreur numérique à x={x}, u={u}: {e}"
            ) from e
        except (ValueError, IndexError) as e:
            raise NumericalError(
                f"[{self.__class__.__name__}] _eval_hx: erreur de shape à x={x}, u={u}: {e}"
            ) from e

    def _eval_A(self, x):
        """
        Évalue df/dx numériquement.
          x : (dim_x, 1)     → retourne (dim_x, dim_x)
          x : (N, dim_x, 1)  → retourne (N, dim_x, dim_x)
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
                f"[{self.__class__.__name__}] _eval_A: erreur numérique à x={x}: {e}"
            ) from e
        except (ValueError, IndexError) as e:
            raise NumericalError(
                f"[{self.__class__.__name__}] _eval_A: erreur de shape à x={x}: {e}"
            ) from e

    def _eval_H(self, x):
        """
        Évalue dh/dx numériquement.
          x : (dim_x, 1)     → retourne (dim_y, dim_x)
          x : (N, dim_x, 1)  → retourne (N, dim_y, dim_x)
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
                f"[{self.__class__.__name__}] _eval_H: erreur numérique à x={x}: {e}"
            ) from e
        except (ValueError, IndexError) as e:
            raise NumericalError(
                f"[{self.__class__.__name__}] _eval_H: erreur de shape à x={x}: {e}"
            ) from e

    # ------------------------------------------------------------------
    # Interfaces _fx / _hx appelées par _g
    # ------------------------------------------------------------------

    def _fx(self, x, t, dt):
        return self._eval_fx(x, t)

    def _hx(self, x, u, dt):
        return self._eval_hx(x, u)

    # ------------------------------------------------------------------
    def _jacobiens_g(self, x, y, t, u, dt):
        """
        Calcule les jacobiennes de g(z) = [f(x,t) ; h(f(x,t), u)] par rapport
        à z = [x ; y] et au bruit noise_z = [t ; u].

        Structure par blocs (nz = dim_x + dim_y) :

            An = dg/dz    = [ df/dx              0     ]
                            [ dh/dx @ df/dx       0    ]

            Bn = dg/dnoise = [ I_nx        0      ]
                             [ dh/dx       I_ny   ]

        dh/dx est évalué en f(x, t) (règle de la chaîne).
        """
        # Les _eval_* catchent FloatingPointError → NumericalError en amont.
        # On se contente de laisser remonter NumericalError sans l'intercepter.
        nx, ny, nz = self.dim_x, self.dim_y, self.dim_xy

        if x.ndim == 2:
            fx_val = self._eval_fx(x, t)  # (nx, 1)
            dfdx = self._eval_A(x)  # (nx, nx)
            dhdx = self._eval_H(fx_val)  # (ny, nx) évalué en f(x)

            An = np.zeros((nz, nz))
            An[:nx, :nx] = dfdx
            An[nx:, :nx] = dhdx @ dfdx  # règle de la chaîne

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
            else:
                return np.concatenate((fx_val, hx_val), axis=1)
        except NumericalError:
            raise
        except ValueError as e:
            raise NumericalError(
                f"[{self.__class__.__name__}] _g: shape mismatch during stack: {e}"
            ) from e

    # ------------------------------------------------------------------
    def latex_model(self):
        return rf"""
            \begin{{align}}
            x_{{k+1}} &= {sp.latex(self._sfx)} \\
            y_k       &= {sp.latex(self._shx)}
            \end{{align}}

            Jacobians:

            \begin{{align}}
            \frac{{\partial f}}{{\partial x}} &= {sp.latex(self._sA)} \\
            \frac{{\partial h}}{{\partial x}} &= {sp.latex(self._sH)}
            \end{{align}}
            """
