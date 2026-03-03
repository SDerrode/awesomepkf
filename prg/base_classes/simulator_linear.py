#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from prg.base_classes.simulator_base import BaseDataSimulator
from prg.models.linear import ModelFactoryLinear
from prg.classes.Linear_PKF import Linear_PKF
from prg.classes.ParamLinear import ParamLinear

__all__ = ["LinearDataSimulator"]


class LinearDataSimulator(BaseDataSimulator):

    def default_filename(self) -> str:
        return f"dataLinear_{self.model_name}.csv"

    def create_model(self):
        return ModelFactoryLinear.create(self.model_name)

    def create_param(self, dim_x, dim_y, params):
        return ParamLinear(self.verbose, dim_x, dim_y, **params)

    def create_pkf(self):
        return Linear_PKF(self.param, sKey=self.sKey, verbose=self.verbose)
