from classes.ParamNonLinear import ParamNonLinear
from classes.NonLinear_EPKF import NonLinear_EPKF
from models.nonLinear import ModelFactoryNonLinear
from base_classes.runner_base import BaseRunner
from others.plot_settings import WINDOW

class NonLinearEPKFRunner(BaseRunner):

    def __init__(self, model_name, ell=None, verbose=1, plot=False, save_history=False, base_dir="."):
        self.ell = ell
        super().__init__(model_name, verbose, plot, save_history, base_dir)

        self.runner_instance = NonLinear_EPKF(self.param, ell=ell, verbose=self.verbose)

    def _get_model_factory(self):
        return ModelFactoryNonLinear

    def _get_param_class(self):
        return ParamNonLinear

    def run(self):
        # Simulation logic
        data = self.runner_instance.simulate_N_data(N=1000)
        self._compute_errors()
        if self.save_history:
            self._save_history("epkf_history.pkl")

    # ----------------------------------------------------------

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
            base_dir=self.graph_dir
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
                base_dir=self.graph_dir
            )