from prg.base_classes.runner_base import BaseRunner
from prg.classes.NonLinear_EPKF import NonLinear_EPKF
from prg.classes.ParamLinear import ParamLinear
from prg.classes.ParamNonLinear import ParamNonLinear
from prg.models.linear import ModelFactoryLinear
from prg.models.nonLinear import ModelFactoryNonLinear
from prg.utils.exceptions import PKFError
from prg.utils.plot_settings import WINDOW

__all__ = ["BaseNonLinearEPKFRunner"]


class BaseNonLinearEPKFRunner(BaseRunner):

    def __init__(
        self,
        model_name,
        verbose=1,
        plot=False,
        save_history=False,
        base_dir=".",
    ):
        """
        Raises
        ------
        ParamError
            Si ``verbose`` est invalide ou ``model_name`` inconnu.
        PKFError
            Si l'instanciation de ``NonLinear_EPKF`` échoue.
        """
        super().__init__(model_name, verbose, plot, save_history, base_dir)

        try:
            self.runner_instance = NonLinear_EPKF(
                param=self.param, sKey=self.sKey, verbose=self.verbose
            )
        except PKFError:
            raise
        except Exception as e:
            raise PKFError(
                f"Failed to instantiate NonLinear_EPKF for model {model_name!r}."
            ) from e

    def _get_model_factory(self):
        return ModelFactoryLinear, ModelFactoryNonLinear

    def _get_param_class(self):
        return ParamLinear, ParamNonLinear

    def _plot_results(self) -> None:

        title = f"Observation data from {self.model_name}"

        self.runner_instance.history.plot(
            title,
            list_param=["ykp1"],
            list_label=["Observations y"],
            list_covar=[None],
            window=WINDOW,
            basename=f"epkf_observations_{self.model_name}",
            show=False,
            base_dir=self.graph_dir,
        )

        if self.runner_instance.ground_truth:
            title = f"'{self.model_name}' model data filtered with EPKF"

            self.runner_instance.history.plot(
                title,
                list_param=["xkp1", "Xkp1_update"],
                list_label=["x true", "x estimated"],
                list_covar=[None, "PXXkp1_update"],
                window=WINDOW,
                basename=f"epkf_{self.model_name}",
                show=False,
                base_dir=self.graph_dir,
            )
