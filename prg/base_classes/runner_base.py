import os
from abc import ABC, abstractmethod

from prg.utils.exceptions import FilterError, ParamError, PKFError

__all__ = ["BaseRunner"]


# ---------------------------------------------------------
# Base Runner
# ---------------------------------------------------------


class BaseRunner(ABC):
    """
    Abstract base runner for all PKF/EPKF/PPF/UPKF workflows.
    Factorizes directories, model building, and history management.
    """

    def __init__(
        self,
        model_name: str,
        verbose: int = 1,
        plot: bool = False,
        save_history: bool = False,
        base_dir: str = ".",
        **kwargs,
    ) -> None:
        """
        Initialises the base runner.

        Parameters
        ----------
        model_name : str
            Name of the model to instantiate.
        verbose : int, optional
            Verbosity level (0, 1 or 2). Default is 1.
        plot : bool, optional
            If True, displays plots after the run.
        save_history : bool, optional
            If True, saves the history as pickle.
        base_dir : str, optional
            Base directory for outputs.
        **kwargs
            Additional arguments (n_particles, sigmaSet, etc.).

        Raises
        ------
        ParamError
            If ``verbose`` is not in ``{0, 1, 2}``.
        PKFError
            If model or parameter construction fails.
        """

        if verbose not in (0, 1, 2):
            raise ParamError("verbose must be 0, 1 or 2.")

        self.model_name = model_name
        self.verbose = verbose
        self.plot = plot
        self.save_history = save_history
        self.base_dir = base_dir
        self._extra_args = kwargs

        self.tracker_dir, self.datafile_dir, self.graph_dir = self._setup_directories()

        self.model, self.param = self._build_model()
        self.runner_instance = None  # Will be set by child (pkf, epkf, ppf, upkf, ukf)

    # ----------------------------------------------------------

    def _setup_directories(self) -> tuple[str, str, str]:
        base_dir = os.path.join(self.base_dir, "data")

        tracker_dir = os.path.join(base_dir, "historyTracker")
        datafile_dir = os.path.join(base_dir, "datafile")
        graph_dir = os.path.join(base_dir, "plot")

        os.makedirs(tracker_dir, exist_ok=True)
        os.makedirs(datafile_dir, exist_ok=True)
        os.makedirs(graph_dir, exist_ok=True)

        return tracker_dir, datafile_dir, graph_dir

    # ----------------------------------------------------------

    @abstractmethod
    def _get_model_factory(self):
        """Return the factory class for model creation."""
        pass

    @abstractmethod
    def _get_param_class(self) -> type:
        """Return the Param class (ParamLinear / ParamNonLinear)."""
        pass

    def _build_model(self):
        """
        Instantiates the model and builds the parameter object.

        Returns
        -------
        tuple[model, param]
            The model and the built parameter object.

        Raises
        ------
        ParamError
            If ``model_name`` is not found in either factory.
        PKFError
            If model or parameter construction fails.
        """
        factoryL, factoryNL = self._get_model_factory()
        param_class_linear, param_class_nonlinear = self._get_param_class()

        if self.model_name in factoryL.list_models():
            factory = factoryL
            param_class = param_class_linear
        elif self.model_name in factoryNL.list_models():
            factory = factoryNL
            param_class = param_class_nonlinear
        else:
            raise ParamError(
                f"Model {self.model_name!r} not found in any registered factory. "
                f"Linear: {factoryL.list_models()}, "
                f"NonLinear: {factoryNL.list_models()}."
            )

        try:
            model = factory.create(self.model_name)
        except PKFError:
            raise
        except Exception as e:
            raise PKFError(f"Failed to create model {self.model_name!r}.") from e

        params = model.get_params().copy()
        dim_x = params.pop("dim_x")
        dim_y = params.pop("dim_y")

        try:
            param = param_class(self.verbose, dim_x, dim_y, **params)
        except PKFError:
            raise
        except Exception as e:
            raise PKFError(
                f"Failed to build parameters for model {self.model_name!r}."
            ) from e

        if self.verbose > 1:
            param.summary()

        return model, param

    # ==========================================================
    # Post-processing
    # ==========================================================

    def _compute_errors(self) -> None:
        """
        Computes and displays filtering errors.

        Raises
        ------
        FilterError
            If error computation fails unexpectedly.
        """

        try:
            ikp1_last = self.runner_instance.history.last()["ikp1"]
            Skp1_last = self.runner_instance.history.last()["Skp1"]
            self.runner_instance.history.compute_errors(
                self.runner_instance,
                ["xkp1"],
                ["Xkp1_update"],
                ["PXXkp1_update"],
                ["ikp1"] if ikp1_last is not None else None,
                ["Skp1"] if Skp1_last is not None else None,
            )
        except PKFError:
            raise
        except RuntimeError as e:
            # Typical case: degenerate covariance matrix Pk (rank 0, det ≈ 0)
            # after PPF particle collapse or filter divergence.
            raise FilterError(
                f"Error computation failed — covariance matrix may be degenerate "
                f"(particle collapse or filter divergence). Detail: {e}"
            ) from e
        except Exception as e:
            raise FilterError("Unexpected error during error computation.") from e

    # ----------------------------------------------------------

    def _save_history(self, filename: str) -> None:
        """
        Saves the filter history to a pickle file.

        Raises
        ------
        OSError
            If saving fails (permissions, disk space, etc.).
        """
        filepath = os.path.join(self.tracker_dir, filename)
        self.runner_instance.history.save_pickle(filepath)

    # ==========================================================
    # Abstract execution
    # ==========================================================

    @abstractmethod
    def run(self) -> list:
        """Execute the main workflow."""
        pass
