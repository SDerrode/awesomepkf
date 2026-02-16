from classes.ParamLinear import ParamLinear
from classes.Linear_PKF import Linear_PKF
from models.linear import ModelFactoryLinear
from base_classes.runner_base import BaseRunner
from others.plot_settings import WINDOW

class LinearPKFRunner(BaseRunner):

    def __init__(self, model_name, verbose=1, plot=False, save_history=False, base_dir="."):
        super().__init__(model_name, verbose, plot, save_history, base_dir)
        self.runner_instance = Linear_PKF(self.param, verbose=self.verbose)

    def _get_model_factory(self):
        return ModelFactoryLinear

    def _get_param_class(self):
        return ParamLinear

    def run(self):
        # Simulation logic here
        data = self.runner_instance.simulate_N_data(N=1000)
        # Post-processing
        self._compute_errors()
        if self.save_history:
            self._save_history("linear_history.pkl")

# ----------------------------------------------------------

    def _plot_results(self) -> None:

        title = f"Observation data from {self.model_name}"

        self.runner_instance.history.plot(
            title,
            list_param=["ykp1"],
            list_label=["Observations y"],
            list_covar=[None],
            window=WINDOW,
            basename=f"pkf_observations_{self.model_name}",
            show=False,
            base_dir=self.graph_dir
        )
        
        if self.runner_instance.ground_truth:
            title = f"'{self.model_name}' model data filtered with PKF"

            self.runner_instance.history.plot(
                title,
                list_param=["xkp1", "Xkp1_update"],
                list_label=["x true", "x estimated"],
                list_covar=[None, "PXXkp1_update"],
                window=WINDOW,
                basename=f"pkf_{self.model_name}",
                show=False,
                base_dir=self.graph_dir
            )