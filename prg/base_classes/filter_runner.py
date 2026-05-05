"""
Generic filter runner.

Replaces the 18 per-filter runner files (``*_runner_base.py``,
``*_runner_simulation.py``, ``*_runner_from_file.py``) with a single
parametrised :class:`FilterRunner` driven by :data:`FILTER_SPECS`.

Two execution modes:
- ``"simulation"``: simulate ``N`` time steps using the model.
- ``"from_file"``: load observations from a CSV.
"""

from pathlib import Path
from typing import Literal

from prg.base_classes.filter_specs import FILTER_SPECS, FilterSpec
from prg.base_classes.runner_base import BaseRunner
from prg.classes.ParamLinear import ParamLinear
from prg.classes.ParamNonLinear import ParamNonLinear
from prg.classes.SigmaPointsSet import SigmaPointsSet
from prg.models.linear import ModelFactoryLinear
from prg.models.nonLinear import ModelFactoryNonLinear
from prg.utils.exceptions import FilterError, ParamError, PKFError
from prg.utils.plot_settings import WINDOW
from prg.utils.utils import file_data_generator

__all__ = ["FilterRunner", "Mode"]


Mode = Literal["simulation", "from_file"]


class FilterRunner(BaseRunner):
    """
    Generic runner for any of the supported filters.

    Parameters
    ----------
    filter_name : str
        Key into :data:`FILTER_SPECS` (``"pkf"``, ``"epkf"``, ``"upkf"``,
        ``"ukf"``, ``"pf"``, ``"ppf"``).
    model_name : str
        Model identifier resolved against the linear / nonlinear factories.
    mode : {"simulation", "from_file"}
        Whether to simulate observations from the model or load them from
        a CSV file.
    N : int, optional
        Number of time steps (required when ``mode="simulation"``).
    sKey : int, optional
        Random seed.
    data_filename : str, optional
        CSV filename (used when ``mode="from_file"``). If ``None``, falls
        back to ``data{Linear,NonLinear}_{model_name}.csv``.
    sigmaSet : str, optional
        Sigma-point set key (UKF/UPKF only).
    n_particles : int, optional
        Particle count (PF/PPF only).
    verbose, plot, save_history, base_dir
        See :class:`BaseRunner`.

    Raises
    ------
    ParamError
        On unknown filter / mode / extra-kwarg validation failure.
    PKFError
        On filter instantiation failure.
    """

    def __init__(
        self,
        filter_name: str,
        model_name: str,
        mode: Mode,
        N: int | None = None,
        sKey: int | None = None,
        data_filename: str | None = None,
        sigmaSet: str | None = None,
        n_particles: int | None = None,
        verbose: int = 1,
        plot: bool = False,
        save_history: bool = False,
        base_dir: str = ".",
    ) -> None:
        if filter_name not in FILTER_SPECS:
            raise ParamError(
                f"Unknown filter {filter_name!r}. "
                f"Available: {sorted(FILTER_SPECS)}."
            )
        self.spec: FilterSpec = FILTER_SPECS[filter_name]

        if mode not in ("simulation", "from_file"):
            raise ParamError(
                f"Unknown mode {mode!r}, expected 'simulation' or 'from_file'."
            )
        self.mode: Mode = mode

        if mode == "simulation" and not (isinstance(N, int) and N > 0):
            raise ParamError(
                f"N must be a strictly positive integer in simulation mode, got {N!r}."
            )

        if "sigmaSet" in self.spec.requires and sigmaSet is not None \
                and sigmaSet not in SigmaPointsSet.registry:
            raise ParamError(
                f"Unknown sigma-point set: {sigmaSet!r}. "
                f"Available: {list(SigmaPointsSet.registry.keys())}."
            )

        if "n_particles" in self.spec.requires and n_particles is not None \
                and not (isinstance(n_particles, int) and n_particles > 0):
            raise ParamError(
                f"n_particles must be None or a strictly positive integer, "
                f"got {n_particles!r}."
            )

        self.N = N if N is not None else -1
        self.sKey = sKey
        self.sigmaSet = sigmaSet
        self.n_particles = n_particles

        super().__init__(model_name, verbose, plot, save_history, base_dir)

        if mode == "from_file":
            kind = "Linear" if self.spec.is_linear else "NonLinear"
            self.data_filename = str(
                Path(self.datafile_dir)
                / (data_filename if data_filename else f"data{kind}_{model_name}.csv")
            )
        else:
            self.data_filename = None

        try:
            self.runner_instance = self._build_filter()
        except PKFError:
            raise
        except Exception as e:
            raise PKFError(
                f"Failed to instantiate {self.spec.acronym} for model {model_name!r}."
            ) from e

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_filter(self):
        kwargs = {"param": self.param, "sKey": self.sKey, "verbose": self.verbose}
        if "sigmaSet" in self.spec.requires:
            kwargs["sigmaSet"] = self.sigmaSet
        if "n_particles" in self.spec.requires:
            kwargs["n_particles"] = self.n_particles
        return self.spec.filter_class(**kwargs)

    def _get_model_factory(self):
        return ModelFactoryLinear, ModelFactoryNonLinear

    def _get_param_class(self):
        return ParamLinear, ParamNonLinear

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _plot_results(self) -> None:
        title = f"Observation data from {self.model_name}"

        self.runner_instance.history.plot(
            title,
            list_param=["ykp1"],
            list_label=["Observations y"],
            list_covar=[None],
            window=WINDOW,
            basename=f"{self.spec.name}_observations_{self.model_name}",
            show=False,
            base_dir=self.graph_dir,
        )

        if self.runner_instance.ground_truth:
            title = (
                f"'{self.model_name}' model data filtered with {self.spec.acronym}"
            )
            self.runner_instance.history.plot(
                title,
                list_param=["xkp1", "Xkp1_update"],
                list_label=["x true", "x estimated"],
                list_covar=[None, "PXXkp1_update"],
                window=WINDOW,
                basename=f"{self.spec.name}_{self.model_name}",
                show=False,
                base_dir=self.graph_dir,
            )

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(self, i: int = 0) -> list:
        """Execute the filter; return a copy of the recorded history."""

        if self.mode == "from_file":
            if not Path(self.data_filename).exists():
                raise FileNotFoundError(
                    f"Data file not found: {self.data_filename!r}."
                )
            try:
                self.runner_instance.process_N_data(
                    N=None,
                    data_generator=file_data_generator(
                        self.data_filename,
                        self.param.dim_x,
                        self.param.dim_y,
                        self.verbose,
                    ),
                )
            except PKFError:
                raise
            except Exception as e:
                raise FilterError(
                    f"Filtering failed (file mode) for model {self.model_name!r}."
                ) from e
            history_filename = f"history_run_{self.spec.name}_file_{i}.pkl"
        else:
            try:
                self.runner_instance.process_N_data(N=self.N)
            except PKFError:
                raise
            except Exception as e:
                raise FilterError(
                    f"Filtering failed (simulation mode) for model {self.model_name!r}."
                ) from e
            history_filename = f"history_run_{self.spec.name}_simulation_{i}.pkl"

        if self.save_history:
            self._save_history(history_filename)

        self._compute_errors()

        if self.plot:
            self._plot_results()

        return self.runner_instance.history._history.copy()
