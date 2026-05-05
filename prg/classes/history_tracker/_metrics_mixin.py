"""Metrics mixin: ``compute_errors`` and ``_compute_sigma_envelope``."""

from __future__ import annotations

import numpy as np
import pandas as pd
from rich.console import Console

from prg.utils.display import rich_show_fields
from prg.utils.exceptions import NumericalError
from prg.utils.metrics import compute_errors as _compute_errors_fn
from prg.utils.numerics import EPS_ABS, EPS_REL

__all__ = ["_MetricsMixin"]


class _MetricsMixin:
    """
    Provides ``compute_errors`` and ``_compute_sigma_envelope``.

    Assumes the host class exposes ``self.as_dataframe()`` and
    ``self.verbose`` (set by the core class).
    """

    def compute_errors(self, model, ListeA, ListeB, ListeC, ListeD=None, ListeE=None):
        """
        Computes and displays error reports between different data series.

        Parameters
        ----------
        ListeA, ListeB, ListeC : list[str]
            Names of the columns to compare.
        ListeD, ListeE : list[str] or None
            Additional columns for certain filters (e.g. particle).
        """
        df = self.as_dataframe()
        _console = Console(force_terminal=True, color_system="truecolor")

        if ListeD is None or ListeE is None:
            for a, b, c in zip(ListeA, ListeB, ListeC, strict=False):
                reportError = _compute_errors_fn(
                    model,
                    df[a].to_numpy(),
                    df[b].to_numpy(),
                    df[c].to_numpy(),
                    None,
                    None,
                )
                if self.verbose > 0:
                    rich_show_fields(
                        reportError,
                        [
                            "mse_total",
                            "mae_total",
                            "nees_mean",
                            "nis_mean",
                            "list_mses_X_and_Y",
                            "list_maes_X_and_Y",
                        ],
                        title=f"{a} vs {b}",
                    )
        else:
            for a, b, c, d, e in zip(ListeA, ListeB, ListeC, ListeD, ListeE, strict=False):
                reportError = _compute_errors_fn(
                    model,
                    df[a].to_numpy(),
                    df[b].to_numpy(),
                    df[c].to_numpy(),
                    df[d].to_numpy(),
                    df[e].to_numpy(),
                )
                if self.verbose > 0:
                    rich_show_fields(
                        reportError,
                        [
                            "mse_total",
                            "mae_total",
                            "nees_mean",
                            "nis_mean",
                            "list_mses_X_and_Y",
                            "list_maes_X_and_Y",
                        ],
                        title=f"{a} vs {b}",
                    )

    def _compute_sigma_envelope(
        self, var_series: pd.Series, col_name: str
    ) -> np.ndarray:
        """
        Computes a stable sigma from a variance series and detects anomalies.

        Parameters
        ----------
        var_series : pd.Series
            Pandas series containing diagonal covariances.
        col_name : str
            Variable name (for error/log messages).

        Returns
        -------
        np.ndarray
            Numpy array containing corrected σ.

        Raises
        ------
        NumericalError
            If strongly negative variances are detected.
        """
        v = var_series.values
        scale = np.nanmax(np.abs(v))
        tol = max(EPS_ABS, EPS_REL * scale)

        slightly_negative = (v < 0) & (v >= -tol)
        strongly_negative = v < -tol

        if strongly_negative.any():
            idx = np.where(strongly_negative)[0][:5]
            raise NumericalError(
                f"Strongly negative variance detected in {col_name!r} "
                f"(max |v|={np.max(np.abs(v[strongly_negative])):.3e}, "
                f"tol={tol:.3e}, first indices={idx.tolist()})."
            )

        v_clipped = np.where(slightly_negative, 0.0, v)
        return np.sqrt(v_clipped)
