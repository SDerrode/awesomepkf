from classes.ParamNonLinear import ParamNonLinear
from classes.NonLinear_PF import NonLinear_PF
from models.nonLinear import ModelFactoryNonLinear
from base_classes.runner_base import BaseRunner
from others.plot_settings import WINDOW

class BaseNonLinearPFRunner(BaseRunner):

    def __init__(self, model_name, nbParticles=None, verbose=1, plot=False, save_history=False, base_dir="."):
        self.nbParticles=nbParticles
        super().__init__(model_name, verbose, plot, save_history, base_dir)
        
        self.runner_instance = NonLinear_PF(param=self.param, nbParticles=self.nbParticles, sKey=self.sKey, verbose=self.verbose)

    def _get_model_factory(self):
        return ModelFactoryNonLinear

    def _get_param_class(self):
        return ParamNonLinear

    # ----------------------------------------------------------

    def _plot_results(self) -> None:

        title = f"Observation data from {self.model_name}"

        self.runner_instance.history.plot(
            title,
            list_param=["ykp1"],
            list_label=["Observations y"],
            list_covar=[None],
            window=WINDOW,
            basename=f"pf_observations_{self.model_name}",
            show=False,
            base_dir=self.graph_dir
        )
        
        if self.runner_instance.ground_truth:
            title = f"'{self.model_name}' model data filtered with PF"

            self.runner_instance.history.plot(
                title,
                list_param=["xkp1", "Xkp1_update"],
                list_label=["x true", "x estimated"],
                list_covar=[None, "PXXkp1_update"],
                window=WINDOW,
                basename=f"pf_{self.model_name}",
                show=False,
                base_dir=self.graph_dir
            )