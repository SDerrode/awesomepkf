
from prg.base_classes.nonlinear_pf_runner_base import BaseNonLinearPFRunner
from prg.utils.exceptions import FilterError, ParamError, PKFError

__all__ = ["NonLinearPFRunnerSim"]


class NonLinearPFRunnerSim(BaseNonLinearPFRunner):
    """
    Runner for nonlinear simulation + PF filtering.
    """

    def __init__(
        self,
        model_name: str,
        N: int,
        sKey: int | None,
        n_particles: int | None,
        verbose: int,
        plot: bool,
        save_history: bool,
        base_dir: str = ".",
    ) -> None:
        """
        Raises
        ------
        ParamError
            Si ``N`` n'est pas un entier strictement positif,
            ``verbose`` invalide, ``model_name`` inconnu,
            ou ``n_particles`` invalide.
        PKFError
            Si l'instanciation du filtre échoue.
        """
        if not (isinstance(N, int) and N > 0):
            raise ParamError(f"N must be a strictly positive integer, got {N!r}.")

        self.N = N
        self.sKey = sKey

        super().__init__(model_name, n_particles, verbose, plot, save_history, base_dir)

    # ==========================================================

    def run(self, i: int = 0) -> list:
        """
        Raises
        ------
        FilterError
            Si la simulation ou le filtrage échoue de manière inattendue.
        PKFError
            Si une erreur du domaine PKF remonte du filtre.
        """

        try:
            self.runner_instance.process_N_data(N=self.N)
        except PKFError:
            raise
        except Exception as e:
            raise FilterError(
                f"Filtering failed (simulation mode) for model {self.model_name!r}."
            ) from e

        if self.save_history:
            self._save_history(f"history_run_pf_simulation_{i}.pkl")

        self._compute_errors()

        if self.plot:
            self._plot_results()

        return self.runner_instance.history._history.copy()
