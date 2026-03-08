#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod

import numpy as np
import sympy as sp

from prg.models.nonLinear.base_model_nonLinear import BaseModelNonLinear
from prg.exceptions import NumericalError

__all__ = ["BaseModelGxGy"]


class BaseModelGxGy(BaseModelNonLinear, ABC):
    """
    Classe mère pour les modèles définis symboliquement via symbolic_model().

    Gère dim_x >= 1 et dim_y >= 1 avec rétroaction complète :
    gx et gy peuvent dépendre simultanément de x, y, t et u.

    La sous-classe doit implémenter :

        def symbolic_model(self, sx, sy, st, su):
            ...
            return sgx, sgy

    où :
        sx  : sp.Matrix(dim_x, 1) — symboles x0 .. x_{dim_x-1}
        sy  : sp.Matrix(dim_y, 1) — symboles y0 .. y_{dim_y-1}
        st  : sp.Matrix(dim_x, 1) — symboles t0 .. t_{dim_x-1}  (bruit état)
        su  : sp.Matrix(dim_y, 1) — symboles u0 .. u_{dim_y-1}  (bruit obs)
        sgx : sp.Matrix(dim_x, 1) — expression de la transition  gx(x, y, t, u)
        sgy : sp.Matrix(dim_y, 1) — expression de l'observation  gy(x, y, t, u)

    Exemple dim_x=1, dim_y=1 :

        def symbolic_model(self, sx, sy, st, su):
            x, y, t, u = sx[0], sy[0], st[0], su[0]
            sgx = sp.Matrix([a * x + b * sp.tanh(y) + t])
            sgy = sp.Matrix([c * y + d * sp.sin(x) + u])
            return sgx, sgy

    Les jacobiennes An = dg/dz et Bn = dg/d_noise sont calculées automatiquement
    (pas de chain rule : gx et gy sont évaluées au même point (x, y)).

        An = d[gx; gy] / d[x; y]   (dim_xy, dim_xy)
        Bn = d[gx; gy] / d[t; u]   (dim_xy, dim_xy)
    """

    # ------------------------------------------------------------------
    @abstractmethod
    def symbolic_model(self, sx, sy, st, su):
        """
        À implémenter dans la sous-classe.

        Paramètres
        ----------
        sx : sp.Matrix(dim_x, 1)  — symboles d'état        x0 .. x_{dim_x-1}
        sy : sp.Matrix(dim_y, 1)  — symboles d'observation y0 .. y_{dim_y-1}
        st : sp.Matrix(dim_x, 1)  — symboles bruit d'état  t0 .. t_{dim_x-1}
        su : sp.Matrix(dim_y, 1)  — symboles bruit obs.    u0 .. u_{dim_y-1}

        Retourne
        --------
        sgx : sp.Matrix(dim_x, 1) — transition  gx(x, y, t, u)
        sgy : sp.Matrix(dim_y, 1) — observation gy(x, y, t, u)
        """

    # ------------------------------------------------------------------
    def __init__(self, dim_x=1, dim_y=1, model_type="nonlinear", augmented=False):
        super().__init__(dim_x, dim_y, model_type, augmented)
        self._build_symbolic_model()

    # ------------------------------------------------------------------
    def _build_symbolic_model(self):
        nx, ny, nz = self.dim_x, self.dim_y, self.dim_xy

        # Vecteurs de symboles réels
        self._sx = sp.Matrix([sp.Symbol(f"x{i}", real=True) for i in range(nx)])
        self._sy = sp.Matrix([sp.Symbol(f"y{i}", real=True) for i in range(ny)])
        self._st = sp.Matrix([sp.Symbol(f"t{i}", real=True) for i in range(nx)])
        self._su = sp.Matrix([sp.Symbol(f"u{i}", real=True) for i in range(ny)])

        # Modèle fourni par la sous-classe
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

        # Validation des shapes
        if not isinstance(self._sgx, sp.Matrix) or self._sgx.shape != (nx, 1):
            raise ValueError(
                f"symbolic_model() doit retourner sgx de shape ({nx}, 1), "
                f"got {getattr(self._sgx, 'shape', type(self._sgx))}"
            )
        if not isinstance(self._sgy, sp.Matrix) or self._sgy.shape != (ny, 1):
            raise ValueError(
                f"symbolic_model() doit retourner sgy de shape ({ny}, 1), "
                f"got {getattr(self._sgy, 'shape', type(self._sgy))}"
            )

        # g = [gx ; gy] — vecteur complet pour les jacobiennes
        sg = self._sgx.col_join(self._sgy)  # (nz, 1)
        sz = self._sx.col_join(self._sy)  # (nz, 1) — état augmenté z
        sn = self._st.col_join(self._su)  # (nz, 1) — bruit augmenté

        # Jacobiennes symboliques directes (pas de chain rule)
        self._sAn = sg.jacobian(sz)  # (nz, nz) : dg/dz
        self._sBn = sg.jacobian(sn)  # (nz, nz) : dg/d_noise

        # Tuples de symboles pour lambdify
        all_syms = tuple(self._sx) + tuple(self._sy) + tuple(self._st) + tuple(self._su)

        # Compilation NumPy.
        # Note : si une expression est constante (ex. Bn = I quand le bruit
        # est additif), lambdify retourne un ndarray au lieu d'un callable.
        # On normalise via _wrap_lambdify pour garantir un callable dans tous les cas.
        self._gx_num = self._wrap_lambdify(sp.lambdify(all_syms, self._sgx, "numpy"))
        self._gy_num = self._wrap_lambdify(sp.lambdify(all_syms, self._sgy, "numpy"))
        self._An_num = self._wrap_lambdify(sp.lambdify(all_syms, self._sAn, "numpy"))
        self._Bn_num = self._wrap_lambdify(sp.lambdify(all_syms, self._sBn, "numpy"))

    # ------------------------------------------------------------------
    # Évaluations numériques internes
    # ------------------------------------------------------------------

    def _args(self, x, y, t, u, i=None):
        """Construit le tuple d'arguments pour lambdify à l'indice i (batch) ou 2D."""
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
        Évalue gx(x, y, t, u) numériquement.
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
                f"[{self.__class__.__name__}] _eval_gx: erreur numérique à x={x}, y={y}: {e}"
            ) from e
        except (ValueError, IndexError) as e:
            raise NumericalError(
                f"[{self.__class__.__name__}] _eval_gx: erreur de shape à x={x}, y={y}: {e}"
            ) from e

    def _eval_gy(self, x, y, t, u):
        """
        Évalue gy(x, y, t, u) numériquement.
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
                f"[{self.__class__.__name__}] _eval_gy: erreur numérique à x={x}, y={y}: {e}"
            ) from e
        except (ValueError, IndexError) as e:
            raise NumericalError(
                f"[{self.__class__.__name__}] _eval_gy: erreur de shape à x={x}, y={y}: {e}"
            ) from e

    def _eval_An(self, x, y, t, u):
        """
        Évalue An = dg/dz numériquement.
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
                f"[{self.__class__.__name__}] _eval_An: erreur numérique à x={x}, y={y}: {e}"
            ) from e
        except (ValueError, IndexError) as e:
            raise NumericalError(
                f"[{self.__class__.__name__}] _eval_An: erreur de shape à x={x}, y={y}: {e}"
            ) from e

    def _eval_Bn(self, x, y, t, u):
        """
        Évalue Bn = dg/d_noise numériquement.
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
                f"[{self.__class__.__name__}] _eval_Bn: erreur numérique à x={x}, y={y}: {e}"
            ) from e
        except (ValueError, IndexError) as e:
            raise NumericalError(
                f"[{self.__class__.__name__}] _eval_Bn: erreur de shape à x={x}, y={y}: {e}"
            ) from e

    # ------------------------------------------------------------------
    # Interfaces appelées par _g et jacobiens_g (BaseModelNonLinear)
    # ------------------------------------------------------------------

    def _gx(self, x, y, t, u, dt):
        return self._eval_gx(x, y, t, u)

    def _gy(self, x, y, t, u, dt):
        return self._eval_gy(x, y, t, u)

    def _jacobiens_g(self, x, y, t, u, dt):
        """
        Jacobiennes directes de g(z, noise) = [gx(x,y,t,u) ; gy(x,y,t,u)].

            An = dg/dz     (dim_xy, dim_xy)   évalué en (x, y, t, u)
            Bn = dg/dnoise (dim_xy, dim_xy)   évalué en (x, y, t, u)

        Pas de chain rule : gx et gy sont évaluées au même point (x, y).
        """
        # Les _eval_* catchent FloatingPointError → NumericalError en amont.
        # On se contente de laisser remonter NumericalError sans l'intercepter.
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
            else:
                return np.concatenate((gx_val, gy_val), axis=1)
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
            x_{{k+1}} &= {sp.latex(self._sgx)} \\
            y_k       &= {sp.latex(self._sgy)}
            \end{{align}}

            Jacobians:

            \begin{{align}}
            \frac{{\partial g}}{{\partial z}} &= {sp.latex(self._sAn)} \\
            \frac{{\partial g}}{{\partial n}} &= {sp.latex(self._sBn)}
            \end{{align}}
            """
