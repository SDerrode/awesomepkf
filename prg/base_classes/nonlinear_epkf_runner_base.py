from classes.ParamNonLinear import ParamNonLinear
from classes.NonLinear_EPKF import NonLinear_EPKF
from models.nonLinear import ModelFactoryNonLinear
from base_classes.runner_base import BaseRunner

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
