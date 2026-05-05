from prg.base_classes.runner_base import BaseRunner
from prg.classes.NonLinear_UKF import NonLinear_UKF
from prg.classes.ParamLinear import ParamLinear
from prg.classes.ParamNonLinear import ParamNonLinear
from prg.classes.SigmaPointsSet import SigmaPointsSet
from prg.models.linear import ModelFactoryLinear
from prg.models.nonLinear import ModelFactoryNonLinear
from prg.utils.exceptions import ParamError, PKFError
from prg.utils.plot_settings import WINDOW

__all__ = ["BaseNonLinearUKFRunner"]


class BaseNonLinearUKFRunner(BaseRunner):

    def __init__(
        self,
        model_name,
        sigmaSet=None,
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
            ou ``sigmaSet`` n'est pas une clé connue du registre.
        PKFError
            Si l'instanciation de ``NonLinear_UKF`` échoue.
        """

        if sigmaSet is not None and sigmaSet not in SigmaPointsSet.registry:
            raise ParamError(
                f"Unknown sigma-point set: {sigmaSet!r}. "
                f"Available: {list(SigmaPointsSet.registry.keys())}."
            )

        self.sigmaSet = sigmaSet

        super().__init__(
            model_name, verbose, plot, save_history, base_dir, sigmaSet=sigmaSet
        )

        try:
            self.runner_instance = NonLinear_UKF(
                param=self.param,
                sigmaSet=self.sigmaSet,
                sKey=self.sKey,
                verbose=self.verbose,
            )
        except PKFError:
            raise
        except Exception as e:
            raise PKFError(
                f"Failed to instantiate NonLinear_UKF for model {model_name!r}. It should be 'classic', not 'pairwise'!"
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
            basename=f"ukf_observations_{self.model_name}",
            show=False,
            base_dir=self.graph_dir,
        )

        if self.runner_instance.ground_truth:
            title = f"'{self.model_name}' model data filtered with UKF"

            self.runner_instance.history.plot(
                title,
                list_param=["xkp1", "Xkp1_update"],
                list_label=["x true", "x estimated"],
                list_covar=[None, "PXXkp1_update"],
                window=WINDOW,
                basename=f"ukf_{self.model_name}",
                show=False,
                base_dir=self.graph_dir,
            )
