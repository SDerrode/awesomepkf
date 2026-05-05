from abc import ABC, abstractmethod
from pathlib import Path

from prg.utils.exceptions import FilterError, ParamError, PKFError
from prg.utils.utils import data_to_dataframe, save_dataframe_to_csv

__all__ = ["BaseDataSimulator"]


# =============================================================
# Base class
# =============================================================


class BaseDataSimulator(ABC):
    """
    Abstract base class for data simulation.
    """

    def __init__(
        self,
        model_name: str,
        N: int,
        sKey: int | None,
        data_file_name: str | None,
        verbose: int,
        withoutX: bool,
    ) -> None:
        """
        Initialises the data simulator.

        Parameters
        ----------
        model_name : str
            Name of the model to instantiate.
        N : int
            Number of time steps to simulate.
        sKey : int or None
            Random seed (None = strong random seed).
        data_file_name : str or None
            Output file name. If None, uses the default name.
        verbose : int
            Verbosity level (0, 1 or 2).
        withoutX : bool
            If True, does not record ground truth in the CSV.

        Raises
        ------
        ParamError
            If ``verbose`` is not in ``{0, 1, 2}``, if ``N`` is not
            a strictly positive integer, or if ``sKey`` is negative.
        PKFError
            If parameter or model construction fails.
        """
        if verbose not in (0, 1, 2):
            raise ParamError("verbose must be 0, 1 or 2.")

        self.verbose = verbose

        self.model_name = model_name
        self.N = N
        self.sKey = sKey
        self.withoutX = withoutX

        if data_file_name is None:
            data_file_name = self.default_filename()
        self.data_file_name = data_file_name

        self.base_dir = "data"
        self.datafile_dir = str(Path(self.base_dir) / "datafile")

        self._validate_inputs()
        self.param = self._build_parameters()

    # ---------------------------------------------------------
    # Abstract methods
    # ---------------------------------------------------------

    @abstractmethod
    def default_filename(self) -> str:
        pass

    @abstractmethod
    def create_model(self):
        pass

    @abstractmethod
    def create_param(self, dim_x, dim_y, params):
        pass

    @abstractmethod
    def create_pkf(self):
        pass

    # ---------------------------------------------------------
    # Shared logic
    # ---------------------------------------------------------

    def _validate_inputs(self) -> None:
        """
        Validates the input parameters common to all simulators.

        Raises
        ------
        ParamError
            If ``N`` is not a strictly positive integer,
            or if ``sKey`` is strictly negative.
        """
        if not (isinstance(self.N, int) and self.N > 0):
            raise ParamError(f"N must be a strictly positive integer, got {self.N!r}.")
        if self.sKey is not None and self.sKey < 0:
            raise ParamError(
                f"sKey must be None or a non-negative integer, got {self.sKey!r}."
            )

    def _build_parameters(self):
        """
        Instantiates the model and builds the parameter object.

        Returns
        -------
        ParamLinear | ParamNonLinear
            The built parameter object.

        Raises
        ------
        PKFError
            If model or parameter creation fails (wrapping
            any exception raised by ``create_model`` or
            ``create_param``).
        """

        try:
            model = self.create_model()
        except PKFError:
            raise  # already typed — let it propagate as-is
        except Exception as e:
            raise PKFError(f"Failed to create model {self.model_name!r}.") from e

        params = model.get_params().copy()
        dim_x = params.pop("dim_x")
        dim_y = params.pop("dim_y")

        try:
            param = self.create_param(dim_x, dim_y, params)
        except PKFError:
            raise  # CovarianceError, ParamError, etc. — let it propagate
        except Exception as e:
            raise PKFError(
                f"Failed to build parameters for model {self.model_name!r}."
            ) from e

        if self.verbose > 1:
            param.summary()

        return param

    def run(self) -> None:
        """
        Executes the simulation and saves the data to a CSV file.

        Raises
        ------
        PKFError
            If filter creation or simulation fails.
        OSError
            If saving the CSV file fails (permissions, disk space, etc.).
        """

        try:
            pkf = self.create_pkf()
        except PKFError:
            raise
        except Exception as e:
            raise PKFError(
                f"Failed to create PKF for model {self.model_name!r}."
            ) from e

        try:
            list_data = pkf.simulate_N_data(N=self.N)
        except PKFError:
            raise
        except Exception as e:
            raise FilterError(
                f"Simulation failed after instantiation of PKF "
                f"for model {self.model_name!r}."
            ) from e

        df = data_to_dataframe(
            list_data,
            self.param.dim_x,
            self.param.dim_y,
            withoutX=self.withoutX,
        )

        Path(self.datafile_dir).mkdir(parents=True, exist_ok=True)

        filepath = str(Path(self.datafile_dir) / self.data_file_name)
        save_dataframe_to_csv(df, filepath)  # OSError remonte naturellement
