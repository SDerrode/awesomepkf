from classes.ParamNonLinear import ParamNonLinear
from classes.NonLinear_UPKF import NonLinear_UPKF
from models.nonLinear import ModelFactoryNonLinear
from base_classes.runner_base import BaseRunner

class NonLinearUPKFRunner(BaseRunner):

    def __init__(self, model_name, sigmaSet=None, verbose=1, plot=False, save_history=False, base_dir="."):
        self.sigmaSet=sigmaSet
        super().__init__(model_name, verbose, plot, save_history, base_dir, sigmaSet=sigmaSet)
        self.runner_instance = NonLinear_UPKF(self.param, sigmaSet=sigmaSet, verbose=self.verbose)

    def _get_model_factory(self):
        return ModelFactoryNonLinear

    def _get_param_class(self):
        return ParamNonLinear

    def run(self):
        # Simulation logic
        data = self.runner_instance.simulate_N_data(N=1000)
        self._compute_errors()
        if self.save_history:
            self._save_history("upkf_history.pkl")
