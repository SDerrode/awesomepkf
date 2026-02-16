#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from base_classes.simulator_base import BaseDataSimulator

from models.nonLinear import ModelFactoryNonLinear
from classes.NonLinear_PKF import NonLinear_PKF
from classes.ParamNonLinear import ParamNonLinear


class NonLinearDataSimulator(BaseDataSimulator):

    def default_filename(self) -> str:
        return f"dataNonLinear_{self.model_name}.csv"

    def create_model(self):
        return ModelFactoryNonLinear.create(self.model_name)

    def create_param(self, dim_x, dim_y, params):
        return ParamNonLinear(self.verbose, dim_x, dim_y, **params)

    def create_pkf(self):
        return NonLinear_PKF(self.param, sKey=self.sKey, verbose=self.verbose)
