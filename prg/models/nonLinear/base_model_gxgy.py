#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from prg.models.nonLinear.base_model_nonLinear import BaseModelNonLinear
from prg.exceptions import NumericalError

__all__ = ["BaseModelGxGy"]


class BaseModelGxGy(BaseModelNonLinear):
    """
    Classe mère pour les modèles définis par _gx (transition d'état avec
    rétroaction sur l'observation) et _gy (fonction d'observation).

    Fournit _g par défaut.
    Les sous-classes doivent implémenter _gx, _gy et _jacobiens_g.
    """

    # ------------------------------------------------------------------
    def _gx(self, x, y, t, u, dt):
        raise NotImplementedError

    def _gy(self, x, y, t, u, dt):
        raise NotImplementedError

    # ------------------------------------------------------------------
    def _g(self, x, y, t, u, dt):
        if __debug__:
            assert isinstance(dt, (float, int))
            if x.ndim == 2:
                assert all(a.shape == (self.dim_x, 1) for a in (x, t))
                assert all(a.shape == (self.dim_y, 1) for a in (y, u))
            else:
                assert all(a.ndim == 3 and a.shape[1:] == (self.dim_x, 1) for a in (x, t))
                assert all(a.ndim == 3 and a.shape[1:] == (self.dim_y, 1) for a in (y, u))
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
