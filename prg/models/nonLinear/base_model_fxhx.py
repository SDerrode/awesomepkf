#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sympy as sp

from prg.models.nonLinear.base_model_nonLinear import BaseModelNonLinear
from prg.exceptions import NumericalError

__all__ = ["BaseModelFxHx"]


class BaseModelFxHx(BaseModelNonLinear):
    """
    Classe mère pour les modèles définis par _fx (transition d'état)
    et _hx (fonction d'observation).

    Fournit _g par défaut.
    Les sous-classes doivent implémenter _fx, _hx et _jacobiens_g.
    """

    def __init__(self, dim_x=1, dim_y=1, model_type="nonlinear", augmented=False):
        # print("BaseModelFxHx - __init__")
        super().__init__(dim_x, dim_y, model_type, augmented)
        # print("BaseModelFxHx - __init__")

        self._build_symbolic_model()
        # print("BaseModelFxHx - __init__")

    def _build_symbolic_model(self):
        # print("  BaseModelFxHx - _build_symbolic_model")

        # variables symboliques
        self._sx = sp.Symbol("x", real=True)  # ← ajouter real=True
        self._st = sp.Symbol("t", real=True)
        self._su = sp.Symbol("u", real=True)

        # modèle fourni par la classe fille
        self._sfx, self._shx = self.symbolic_model(self._sx, self._st, self._su)
        for name, mat in [("sfx", self._sfx), ("shx", self._shx)]:
            if not isinstance(mat, sp.Matrix) or mat.shape != (1, 1):
                raise ValueError(
                    f"symbolic_model() doit retourner des sp.Matrix(1,1), "
                    f"got {type(mat)} shape={getattr(mat, 'shape', '?')} for {name}"
                )
        # print(self._sfx)
        # print(self._shx)
        # input("ATTENTE")

        # jacobiennes
        self._sA = sp.diff(self._sfx, self._sx)
        self._sH = sp.diff(self._shx, self._sx)
        # print(self._sA)
        # print(self._sH)
        # input("ATTENTE 2")

        # compilation numpy — extraire [0,0] pour éviter l'encapsulation matricielle
        self._fx_num = sp.lambdify((self._sx, self._st), self._sfx[0, 0], "numpy")
        self._hx_num = sp.lambdify((self._sx, self._su), self._shx[0, 0], "numpy")

        self._A_num = sp.lambdify((self._sx,), self._sA[0, 0], "numpy")
        self._H_num = sp.lambdify((self._sx,), self._sH[0, 0], "numpy")
        # print(self._A_num)
        # print(self._H_num)
        # input("ATTENTE 4")

    # ------------------------------------------------------------------

    def _fx(self, x, t, dt):
        return self._fx_num(x, t)

    def _hx(self, x, u, dt):
        return self._hx_num(x, u)

    def _jacobiens_g(self, x, y, t, u, dt):
        try:
            with np.errstate(all="raise"):

                if x.ndim == 2:
                    x0 = x[0, 0]
                    t0 = t[0, 0]
                    fx_val = float(self._fx_num(x0, t0))  # A = fx(x)
                    dfdx = float(self._A_num(x0))  # df/dx en x
                    dhdx = float(self._H_num(fx_val))  # dh/dx en fx(x) ← correction clé
                    An = np.array([[dfdx, 0.0], [dhdx * dfdx, 0.0]])
                    Bn = np.array(
                        [[1.0, 0.0], [dhdx, 1.0]]
                    )  # dhdx ici, pas sA hardcodé
                else:
                    N = x.shape[0]
                    x0 = x[:, 0, 0]
                    t0 = t[:, 0, 0]
                    fx_val = self._fx_num(x0, t0)
                    dfdx = self._A_num(x0)
                    dhdx = self._H_num(fx_val)  # ← évalué en fx(x)
                    An = np.zeros((N, 2, 2))
                    An[:, 0, 0] = dfdx
                    An[:, 1, 0] = dhdx * dfdx
                    Bn = np.tile(np.eye(2), (N, 1, 1))
                    Bn[:, 1, 0] = dhdx
                return An, Bn

        except FloatingPointError as e:
            raise NumericalError(
                f"[{self.__class__.__name__}] _jacobiens_g: erreur numérique: {e}"
            ) from e

    def latex_model(self):

        return rf"""
            \begin{{align}}
            x_{{k+1}} &= {sp.latex(self._sfx)} \\
            y_k &= {sp.latex(self._shx)}
            \end{{align}}

            Jacobian:

            \begin{{align}}
            \frac{{\partial f}}{{\partial x}} &= {sp.latex(self._sA)} \\
            \frac{{\partial h}}{{\partial x}} &= {sp.latex(self._sH)}
            \end{{align}}
            """

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
            # print("       BaseModelFxHx - _g")
            fx_val = self._fx(x, t, dt)
            # print(fx_val)
            hx_val = self._hx(fx_val, u, dt)
            # print(hx_val)
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
