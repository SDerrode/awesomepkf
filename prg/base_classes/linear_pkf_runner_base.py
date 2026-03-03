#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from prg.classes.ParamNonLinear import ParamNonLinear
from prg.classes.ParamLinear import ParamLinear
from prg.classes.Linear_PKF import Linear_PKF
from prg.models.nonLinear import ModelFactoryNonLinear
from prg.models.linear import ModelFactoryLinear
from prg.base_classes.runner_base import BaseRunner
from prg.utils.plot_settings import WINDOW
from prg.exceptions import PKFError

__all__ = ["BaseLinearPKFRunner"]


class BaseLinearPKFRunner(BaseRunner):

    def __init__(
        self, model_name, verbose=1, plot=False, save_history=False, base_dir="."
    ):
        """
        Initialise le runner linéaire PKF.

        Raises
        ------
        ParamError
            Si ``verbose`` est invalide ou si ``model_name`` est inconnu
            (levée par ``BaseRunner.__init__``).
        PKFError
            Si l'instanciation de ``Linear_PKF`` échoue.
        """
        super().__init__(model_name, verbose, plot, save_history, base_dir)

        try:
            self.runner_instance = Linear_PKF(
                param=self.param, sKey=self.sKey, verbose=self.verbose
            )
        except PKFError:
            raise
        except Exception as e:
            raise PKFError(
                f"Failed to instantiate Linear_PKF for model {model_name!r}."
            ) from e

    def _get_model_factory(self):
        return ModelFactoryLinear, ModelFactoryNonLinear

    def _get_param_class(self):
        return ParamLinear, ParamNonLinear

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
            base_dir=self.graph_dir,
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
                base_dir=self.graph_dir,
            )
