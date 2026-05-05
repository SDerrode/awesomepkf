from prg.base_classes.simulator_base import BaseDataSimulator
from prg.classes.ParamNonLinear import ParamNonLinear
from prg.classes.PKF import PKF
from prg.models.nonLinear import ModelFactoryNonLinear

__all__ = ["NonLinearDataSimulator"]


class NonLinearDataSimulator(BaseDataSimulator):

    def default_filename(self) -> str:
        return f"dataNonLinear_{self.model_name}.csv"

    def create_model(self):
        return ModelFactoryNonLinear.create(self.model_name)

    def create_param(self, dim_x, dim_y, params):
        return ParamNonLinear(self.verbose, dim_x, dim_y, **params)

    def create_pkf(self):
        return PKF(self.param, sKey=self.sKey, verbose=self.verbose)
