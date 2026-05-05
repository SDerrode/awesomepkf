"""Plot mixin: ``plot`` orchestrator + 6 single-responsibility helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

from prg.utils.exceptions import ParamError
from prg.utils.plot_settings import BIG_SIZE, DPI, FACECOLOR

__all__ = ["_PlotMixin"]


class _PlotMixin:
    """
    Provides ``plot`` and its 6 helpers.

    Assumes the host class exposes ``self.as_dataframe()`` and the
    ``self._compute_sigma_envelope()`` method (from the metrics mixin).
    """

    def plot(
        self,
        title,
        list_param,
        list_label,
        list_covar,
        window,
        basename="plot",
        show=True,
        base_dir=None,
        **kwargs,
    ):
        """
        Plots the evolution of states with their ±2σ covariance envelopes.

        Parameters
        ----------
        title : str
            Global figure title.
        list_param : list[str]
            Names of columns to plot.
        list_label : list[str]
            Labels for the legend.
        list_covar : list[str or None]
            Columns containing the associated covariance (or None).
        window : dict
            Time window to plot, with keys 'xmin' and 'xmax'.
        basename : str, optional
            File name if saving (default "plot").
        show : bool, optional
            Displays the figure if True (default True).
        base_dir : str, optional
            Save directory if show=False.

        Returns
        -------
        fig, axes : tuple
            Matplotlib figure and axes.

        Raises
        ------
        ParamError
            If the lists ``list_param``, ``list_label``, ``list_covar``
            do not have the same length, or if ``window`` is malformed,
            or if a column is absent from the DataFrame.
        ParamError
            If the first element of the column is not a numpy vector.
        NumericalError
            If strongly negative variances are detected during the
            computation of ±2σ envelopes (via ``_compute_sigma_envelope``).
        """
        df, xmin, xmax = self._plot_validate_and_select(
            list_param, list_label, list_covar, window
        )
        nb_components = self._plot_components_count(df, list_param)

        df_subset, df_subset_var, list_labels_p, list_labels_e, list_has_var = (
            self._plot_prepare_dataframes(df, list_param, list_covar, nb_components)
        )

        fig, axes = self._plot_create_figure(nb_components, title)

        self._plot_components(
            axes, df_subset, df_subset_var, list_labels_p, list_labels_e,
            list_has_var, list_label, nb_components,
        )

        self._plot_format_axes(axes, xmin, xmax)
        fig.canvas.draw_idle()

        self._plot_save_or_show(fig, show, base_dir, basename)
        return fig, axes

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _plot_validate_and_select(self, list_param, list_label, list_covar, window):
        """Validate args and return (df_window, xmin, xmax)."""
        if not (len(list_param) == len(list_label) == len(list_covar)):
            raise ParamError(
                "list_param, list_label and list_covar must have the same length."
            )

        for key in ("xmin", "xmax"):
            if key not in window:
                raise ParamError(f"window must contain the key '{key}'.")

        xmin, xmax = window["xmin"], window["xmax"]
        df = self.as_dataframe().iloc[xmin:xmax]

        if df.empty:
            df = self.as_dataframe()
            if df.empty:
                raise ParamError("No data recorded.")
            xmin, xmax = 0, len(df)

        for p in list_param:
            if p not in df.columns:
                raise ParamError(
                    f"'{p}' is not a known column: {list(df.columns)}."
                )
        return df, xmin, xmax

    @staticmethod
    def _plot_components_count(df, list_param):
        """Return the per-record component count (e.g. dim_x for state vectors)."""
        first = df[list_param[0]].iloc[0]
        if not hasattr(first, "shape"):
            raise ParamError(
                f"The first element of '{list_param[0]}' is not a numpy vector."
            )
        return first.shape[0]

    @staticmethod
    def _plot_prepare_dataframes(df, list_param, list_covar, nb_components):
        """Flatten vector-valued cells and pull diagonals from covariance cells."""
        df_subset = pd.DataFrame()
        df_subset_var = pd.DataFrame()
        list_labels_p = []
        list_labels_e = []
        list_has_var = []

        for p, e in zip(list_param, list_covar, strict=False):
            has_var = e is not None
            list_has_var += [has_var] * nb_components

            list_labels_p_local = [f"{p}_{c}" for c in range(nb_components)]
            list_labels_e_local = [f"{e}_{c}" for c in range(nb_components)]
            list_labels_p += list_labels_p_local
            list_labels_e += list_labels_e_local

            df_subset[list_labels_p_local] = df[p].apply(
                lambda x: pd.Series(x.flatten())
            )

            if has_var:
                df_subset_var[list_labels_e_local] = df[e].apply(
                    lambda x: pd.Series(x.diagonal())
                )

        return df_subset, df_subset_var, list_labels_p, list_labels_e, list_has_var

    @staticmethod
    def _plot_create_figure(nb_components, title):
        """Create the (fig, axes) layout. ``axes`` is always a list."""
        fig, axes = plt.subplots(
            nb_components,
            1,
            figsize=(7, 2 * nb_components),
            sharex=True,
            facecolor=FACECOLOR,
        )
        if nb_components == 1:
            axes = [axes]
        fig.suptitle(title, y=0.85, fontsize=BIG_SIZE)
        return fig, axes

    def _plot_components(
        self,
        axes,
        df_subset,
        df_subset_var,
        list_labels_p,
        list_labels_e,
        list_has_var,
        list_label,
        nb_components,
    ):
        """Plot each variable component + its ±2σ envelope where available."""
        for i, (col_p, col_e, has_var) in enumerate(
            zip(list_labels_p, list_labels_e, list_has_var, strict=False)
        ):
            j = i % nb_components
            k = i // nb_components
            df_subset[col_p].plot(ax=axes[j], label=list_label[k], alpha=0.5)

            if not has_var:
                continue
            if col_e not in df_subset_var:
                raise ParamError(f"Variance '{col_e}' absent from df_subset_var.")

            # Raises NumericalError if strongly negative variances are detected
            sigma = self._compute_sigma_envelope(df_subset_var[col_e], col_e)

            y_upper = df_subset[col_p] + 2.0 * sigma
            y_lower = df_subset[col_p] - 2.0 * sigma
            color = axes[j].lines[-1].get_color()
            axes[j].fill_between(
                df_subset.index,
                y_lower,
                y_upper,
                color=color,
                alpha=0.2,
                label=f"{list_label[k]} ± 2*" + r"$\sigma$",
            )

    @staticmethod
    def _plot_format_axes(axes, xmin, xmax):
        """Apply grid, legend, x-axis limits and locator on the bottom axis."""
        for ax in axes:
            ax.grid(True, linestyle="--", alpha=0.6)
        handles, labels = axes[-1].get_legend_handles_labels()
        unique = dict(zip(labels, handles, strict=False))
        axes[-1].legend(
            unique.values(),
            unique.keys(),
            loc="upper center",
            bbox_to_anchor=(0.5, 1.15),
            ncol=len(unique),
        )
        axes[-1].set_xlim(xmin, xmax - 1)
        axes[-1].set_xlabel("n")
        axes[-1].minorticks_off()
        axes[-1].xaxis.set_major_locator(mticker.MaxNLocator(nbins=5, integer=True))

    @staticmethod
    def _plot_save_or_show(fig, show, base_dir, basename):
        """Either display the figure interactively or save it as PNG and close."""
        if show:
            plt.show()
            return
        out_dir = Path(base_dir or ".")
        out_dir.mkdir(parents=True, exist_ok=True)
        save_path = out_dir / f"{basename}.png"
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight", facecolor=FACECOLOR)
        plt.close(fig)
