from prg.classes.ParamNonLinear import ParamNonLinear
from prg.classes.ParamLinear import ParamLinear
from prg.classes.NonLinear_UPKF import NonLinear_UPKF
from prg.models.nonLinear import ModelFactoryNonLinear
from prg.models.linear import ModelFactoryLinear
from prg.base_classes.runner_base import BaseRunner
from prg.utils.plot_settings import WINDOW

__all__ = ["BaseNonLinearUPKFRunner"]


class BaseNonLinearUPKFRunner(BaseRunner):

    def __init__(
        self,
        model_name,
        sigmaSet=None,
        verbose=1,
        plot=False,
        save_history=False,
        base_dir=".",
    ):

        self.sigmaSet = sigmaSet

        super().__init__(
            model_name, verbose, plot, save_history, base_dir, sigmaSet=sigmaSet
        )

        self.runner_instance = NonLinear_UPKF(
            param=self.param,
            sigmaSet=self.sigmaSet,
            sKey=self.sKey,
            verbose=self.verbose,
        )

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
            basename=f"upkf_observations_{self.model_name}",
            show=False,
            base_dir=self.graph_dir,
        )

        if self.runner_instance.ground_truth:
            title = f"'{self.model_name}' model data filtered with UPKF"

            self.runner_instance.history.plot(
                title,
                list_param=["xkp1", "Xkp1_update"],
                list_label=["x true", "x estimated"],
                list_covar=[None, "PXXkp1_update"],
                window=WINDOW,
                basename=f"upkf_{self.model_name}",
                show=False,
                base_dir=self.graph_dir,
            )
