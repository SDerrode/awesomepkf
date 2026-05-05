import pickle
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from rich.console import Console

from prg.utils.exceptions import NumericalError, ParamError
from prg.utils.numerics import EPS_ABS, EPS_REL
from prg.utils.plot_settings import BIG_SIZE, DPI, FACECOLOR
from prg.utils.utils import compute_errors, rich_show_fields

__all__ = ["HistoryTracker"]


class HistoryTracker:
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
    def save_pickle(self, path: str) -> None:
        p = Path(path)
        if p.parent != Path():
            p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as f:
            pickle.dump(self._history, f)

    @classmethod
    def load_pickle(cls, path: str) -> "HistoryTracker":
        """
        Reloads a HistoryTracker from a pickle file.

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
        with p.open("rb") as f:
            data = pickle.load(f)
        if not isinstance(data, list):
            raise TypeError("The file does not contain a list of records.")
        tracker = cls()
        tracker._history = data

        return tracker

    # ------------------------------------------------------------------
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
                reportError = compute_errors(
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
                reportError = compute_errors(
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
                f"(example indices {idx.tolist()}, values {v[idx].tolist()}).",
                matrix_name=col_name,
            )

        v_corrected = v.copy()
        v_corrected[slightly_negative] = 0.0
        return np.sqrt(v_corrected)

    # ------------------------------------------------------------------
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

        first = df[list_param[0]].iloc[0]
        if not hasattr(first, "shape"):
            raise ParamError(
                f"The first element of '{list_param[0]}' is not a numpy vector."
            )

        nb_components = first.shape[0]

        df_subset = pd.DataFrame()
        df_subset_var = pd.DataFrame()
        list_labels_p = []
        list_labels_e = []
        list_has_var = []

        for p, e in zip(list_param, list_covar, strict=False):
            has_var = e is not None
            list_has_var += [has_var] * nb_components

            list_labels_p_local = []
            list_labels_e_local = []
            for component in range(nb_components):
                list_labels_p_local.append(f"{p}_{component}")
                list_labels_e_local.append(f"{e}_{component}")
            list_labels_p += list_labels_p_local
            list_labels_e += list_labels_e_local

            df_subset[list_labels_p_local] = df[p].apply(
                lambda x: pd.Series(x.flatten())
            )

            if has_var:
                df_subset_var[list_labels_e_local] = df[e].apply(
                    lambda x: pd.Series(x.diagonal())
                )

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

        for i, (col_p, col_e, has_var) in enumerate(
            zip(list_labels_p, list_labels_e, list_has_var, strict=False)
        ):
            j = i % nb_components
            k = i // nb_components
            df_subset[col_p].plot(ax=axes[j], label=list_label[k], alpha=0.5)

            if has_var and col_e not in df_subset_var:
                raise ParamError(f"Variance '{col_e}' absent from df_subset_var.")

            if has_var:
                # Raises NumericalError if strongly negative variances are detected
                sigma = self._compute_sigma_envelope(df_subset_var[col_e], col_e)

                y_upper = df_subset[col_p] + 2.0 * sigma
                y_lower = df_subset[col_p] - 2.0 * sigma
                last_line = axes[j].lines[-1]
                color = last_line.get_color()
                axes[j].fill_between(
                    df_subset.index,
                    y_lower,
                    y_upper,
                    color=color,
                    alpha=0.2,
                    label=f"{list_label[k]} ± 2*" + r"$\sigma$",
                )

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
        fig.canvas.draw_idle()

        if show:
            plt.show()
        else:
            out_dir = Path(base_dir or ".")
            out_dir.mkdir(parents=True, exist_ok=True)
            save_path = out_dir / f"{basename}.png"
            fig.savefig(save_path, dpi=DPI, bbox_inches="tight", facecolor=FACECOLOR)
            plt.close(fig)

        return fig, axes

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._history)

    def __repr__(self) -> str:
        return f"<HistoryTracker n_records={len(self)} - address: {hex(id(self))}>"


# ======================================================================


@dataclass
class SimpleStep:
    iter: int
    x: float
    new_x: float
    diff: float


class A:
    """Toy class to illustrate the usage of HistoryTracker."""

    def __init__(self, x0: float = 1.0, verbose: int = 1):
        assert isinstance(x0, (int, float)), "x0 must be a number"
        assert verbose in [0, 1, 2], "verbose must be 0, 1 or 2"

        self.x = float(x0)
        self.verbose = verbose
        self.history = HistoryTracker(verbose=verbose)

    def iterate_gen(self, n: int | None = None):
        k = 0
        while n is None or k < n:
            new_x = np.cos(self.x)
            diff = abs(new_x - self.x)
            step = SimpleStep(iter=k, x=self.x, new_x=new_x, diff=diff)
            self.history.record(step)
            yield step
            self.x = new_x
            k += 1

    def iterate_list(self, n: int):
        """Returns the complete list of iterations."""
        assert isinstance(n, int) and n > 0, "n must be a positive integer"
        return list(self.iterate_gen(n))


# ======================================================================
if __name__ == "__main__":
    verbose = 1
    graph_dir = Path("data") / "plot"
    tracker_dir = Path("data") / "historyTracker"
    graph_dir.mkdir(parents=True, exist_ok=True)
    tracker_dir.mkdir(parents=True, exist_ok=True)

    a = A(x0=1.0, verbose=verbose)
    for step in a.iterate_gen(5):
        print(step)

    a.history.plot(
        title="Evolution of x",
        list_param=["x"],
        list_label=["x"],
        list_covar=[None],
        window={"xmin": 0, "xmax": len(a.history)},
        show=False,
        base_dir=str(graph_dir),
    )
    a.history.save_pickle(str(tracker_dir / "history_run_a.pkl"))
