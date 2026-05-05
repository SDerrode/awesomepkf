"""HistoryTracker core — record / persist / inspect the recorded history.

The plotting and metrics methods live in two mixins:
- :class:`prg.classes.history_tracker._plot_mixin._PlotMixin`
- :class:`prg.classes.history_tracker._metrics_mixin._MetricsMixin`
"""

from __future__ import annotations

import pickle
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from prg.classes.history_tracker._metrics_mixin import _MetricsMixin
from prg.classes.history_tracker._plot_mixin import _PlotMixin
from prg.utils.exceptions import ParamError

__all__ = ["HistoryTracker"]


class HistoryTracker(_PlotMixin, _MetricsMixin):
    """
    Records and visualises the evolution of quantities over iterations.

    This class is useful for tracking variables in simulations, filters
    (Kalman, particle, etc.) or any iterative algorithm. It allows:

    - Recording quantities at each iteration via `record()`.
    - Converting the history to a pandas DataFrame for analysis.
    - Computing and displaying errors via `compute_errors()`.
    - Plotting variables with covariances and ±2σ envelopes via `plot()`.
    - Saving/reloading the history via pickle.

    Attributes
    ----------
    _history : list[dict[str, Any]]
        List of records made via `record()`.
    verbose : int
        Verbosity level:
        0 = warnings only
        1 = main information
        2 = detailed debug
    """

    def __init__(self, verbose: int = 0):
        """
        Initialises an empty HistoryTracker.

        Parameters
        ----------
        verbose : int, optional
            Verbosity level (0, 1, 2). Default 0.

        Raises
        ------
        ParamError
            If ``verbose`` does not belong to ``{0, 1, 2}``.
        """
        if verbose not in (0, 1, 2):
            raise ParamError("verbose must be 0, 1 or 2.")
        self._history: list[dict[str, Any]] = []
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, *args, **kwargs) -> None:
        """
        Records the current state.

        - If a dataclass (PKFStep) is passed, it is converted to a dict.
        - Otherwise, **kwargs are accepted as before.

        Raises
        ------
        TypeError
            If the keys of ``kwargs`` are not all strings.
        """
        if len(args) == 1 and is_dataclass(args[0]):
            self._history.append(asdict(args[0]))
        else:
            if not all(isinstance(k, str) for k in kwargs):
                raise TypeError("All keys must be strings.")
            self._history.append(kwargs.copy())

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._history)

    def last(self) -> dict[str, Any] | None:
        return self._history[-1] if self._history else None

    def clear(self) -> None:
        self._history.clear()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_pickle(self, path: str) -> None:
        p = Path(path)
        if p.parent != Path():
            p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as f:
            pickle.dump(self._history, f)

    @classmethod
    def load_pickle(cls, path: str) -> HistoryTracker:
        """
        Reloads a HistoryTracker from a pickle file.

        .. warning::
           Uses ``pickle.load``, which executes arbitrary code on
           deserialisation. Only call this on files you produced
           yourself via ``save_pickle()``. Do not load history files
           from untrusted sources.

        Parameters
        ----------
        path : str
            Path to the pickle file.

        Returns
        -------
        HistoryTracker
            A HistoryTracker object containing the reloaded history.

        Raises
        ------
        FileNotFoundError
            If the file does not exist. — stdlib, intentional.
        TypeError
            If the file content is not a list. — stdlib, intentional.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {path}")
        # pickle is intentional — caller responsibility (see docstring warning).
        with p.open("rb") as f:
            data = pickle.load(f)
        if not isinstance(data, list):
            raise TypeError("The file does not contain a list of records.")
        tracker = cls()
        tracker._history = data

        return tracker

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._history)

    def __repr__(self) -> str:
        return f"<HistoryTracker n_records={len(self)} - address: {hex(id(self))}>"
