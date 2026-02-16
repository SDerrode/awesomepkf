from classes.ParamLinear import ParamLinear
from classes.Linear_PKF import Linear_PKF
from models.linear import ModelFactoryLinear
from base_classes.runner_base import BaseRunner

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
