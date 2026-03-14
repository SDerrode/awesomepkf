#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from prg.classes.ParamNonLinear import ParamNonLinear
from prg.classes.ParamLinear import ParamLinear
from prg.classes.NonLinear_PF import NonLinear_PF
from prg.models.nonLinear import ModelFactoryNonLinear
from prg.models.linear import ModelFactoryLinear
from prg.base_classes.runner_base import BaseRunner
from prg.utils.plot_settings import WINDOW
from prg.utils.exceptions import ParamError, PKFError

__all__ = ["BaseNonLinearPFRunner"]


class BaseNonLinearPFRunner(BaseRunner):

    def __init__(
        self,
        model_name,
        n_particles=None,
        verbose=1,
        plot=False,
        save_history=False,
        base_dir=".",
    ):
        """
        Raises
        ------
        ParamError
            Si ``verbose`` est invalide, ``model_name`` inconnu,
            ou ``n_particles`` n'est pas un entier strictement positif.
        PKFError
            Si l'instanciation de ``NonLinear_PF`` échoue.
        """
        if n_particles is not None and not (
            isinstance(n_particles, int) and n_particles > 0
        ):
            raise ParamError(
                f"n_particles must be None or a strictly positive integer, "
                f"got {n_particles!r}."
            )

        self.n_particles = n_particles

        super().__init__(model_name, verbose, plot, save_history, base_dir)

        try:
            self.runner_instance = NonLinear_PF(
                param=self.param,
                n_particles=self.n_particles,
                sKey=self.sKey,
                verbose=self.verbose,
            )
        except PKFError:
            raise
        except Exception as e:
            raise PKFError(
                f"Failed to instantiate NonLinear_PF for model {model_name!r}."
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
            basename=f"pf_observations_{self.model_name}",
            show=False,
            base_dir=self.graph_dir,
        )

        if self.runner_instance.ground_truth:
            title = f"'{self.model_name}' model data filtered with PF"

            self.runner_instance.history.plot(
                title,
                list_param=["xkp1", "Xkp1_update"],
                list_label=["x true", "x estimated"],
                list_covar=[None, "PXXkp1_update"],
                window=WINDOW,
                basename=f"pf_{self.model_name}",
                show=False,
                base_dir=self.graph_dir,
            )
