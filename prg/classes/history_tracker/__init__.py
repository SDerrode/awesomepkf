"""
HistoryTracker package.

The class lives in :mod:`._core`; plotting and metrics methods are
in :mod:`._plot_mixin` and :mod:`._metrics_mixin` respectively. The
single import of interest is :class:`HistoryTracker`, re-exported here.
"""

from prg.classes.history_tracker._core import HistoryTracker

__all__ = ["HistoryTracker"]
