#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from typing import Callable, Any, Union, Optional

# ----------------------------------------------------------------------
# Classe ActiveView
# ----------------------------------------------------------------------
class ActiveView:
    """
    Vue sur une sous-matrice de `parent_matrix`.
    Appelle `callback()` à chaque modification.
    """

    def __init__(
        self,
        parent_matrix: np.ndarray,
        rows: Union[slice, list[int], np.ndarray, int],
        cols: Union[slice, list[int], np.ndarray, int],
        callback: Callable[[], None]
    ) -> None:
        self._parent: np.ndarray = parent_matrix
        self._rows: Union[slice, list[int], np.ndarray, int] = rows
        self._cols: Union[slice, list[int], np.ndarray, int] = cols
        self._callback: Callable[[], None] = callback

    def _submatrix(self) -> np.ndarray:
        """Retourne la sous-matrice correspondante."""
        if isinstance(self._rows, (list, np.ndarray)) and isinstance(self._cols, (list, np.ndarray)):
            return self._parent[np.ix_(self._rows, self._cols)]
        else:
            return self._parent[self._rows, self._cols]

    def __getitem__(self, key: Any) -> np.ndarray:
        return self._submatrix()[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        sub = self._submatrix()
        sub[key] = value
        if isinstance(self._rows, (list, np.ndarray)) and isinstance(self._cols, (list, np.ndarray)):
            self._parent[np.ix_(self._rows, self._cols)] = sub
        else:
            self._parent[self._rows, self._cols] = sub
        if __debug__:
            self._callback()

    def __sub__(self, other: Any) -> "ActiveView":
        if not isinstance(other, ActiveView):
            return NotImplemented
        A = self.value
        B = other.value
        if A.shape != B.shape:
            raise ValueError(f"Shape mismatch: {A.shape} vs {B.shape}")
        diff = A - B
        return ActiveView(diff, range(diff.shape[0]), range(diff.shape[1]), lambda: None)

    def __repr__(self) -> str:
        return f"ActiveView(\n{self.value}\n)"

    @property
    def value(self) -> np.ndarray:
        return self._submatrix()

    def __array__(self, dtype: Optional[np.dtype] = None) -> np.ndarray:
        return np.asarray(self.value, dtype=dtype)

    def copy(self) -> np.ndarray:
        return self.value.copy()

    def __neg__(self) -> "ActiveView":
        return ActiveView(-self.value, range(self.value.shape[0]), range(self.value.shape[1]), lambda: None)

    def __add__(self, other: Any) -> "ActiveView":
        if isinstance(other, ActiveView):
            other = other.value
        return ActiveView(self.value + other, range(self.value.shape[0]), range(self.value.shape[1]), lambda: None)

    def __radd__(self, other: Any) -> "ActiveView":
        return self.__add__(other)
    