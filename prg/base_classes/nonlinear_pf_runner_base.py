from classes.ParamNonLinear import ParamNonLinear
from classes.NonLinear_PF import NonLinear_PF
from models.nonLinear import ModelFactoryNonLinear
from base_classes.runner_base import BaseRunner

class NonLinearPFRunner(BaseRunner):

    def __init__(self, model_name, nbParticles=None, verbose=1, plot=False, save_history=False, base_dir="."):
        self.nbParticles=nbParticles
        super().__init__(model_name, verbose, plot, save_history, base_dir)
        self.runner_instance = NonLinear_PF(self.param, nbParticles=nbParticles, verbose=self.verbose)

    def _get_model_factory(self):
        return ModelFactoryNonLinear

    def _get_param_class(self):
        return ParamNonLinear

    def run(self):
        # Simulation logic
        data = self.runner_instance.simulate_N_data(N=1000)
        self._compute_errors()
        if self.save_history:
            self._save_history("pf_history.pkl")
            