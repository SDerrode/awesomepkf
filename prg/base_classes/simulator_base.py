#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from abc import ABC, abstractmethod

from prg.utils.utils import save_dataframe_to_csv, data_to_dataframe
from prg.utils.exceptions import FilterError, ParamError, PKFError

__all__ = ["BaseDataSimulator"]


# =============================================================
# Base class
# =============================================================


class BaseDataSimulator(ABC):
    """
    Abstract base class for data simulation.
    """

    def __init__(
        self,
        model_name: str,
        N: int,
        sKey: int | None,
        data_file_name: str | None,
        verbose: int,
        withoutX: bool,
    ) -> None:
        """
        Initialise le simulateur de données.

        Parameters
        ----------
        model_name : str
            Nom du modèle à instancier.
        N : int
            Nombre de pas de temps à simuler.
        sKey : int or None
            Graine aléatoire (None = graine aléatoire forte).
        data_file_name : str or None
            Nom du fichier de sortie. Si None, utilise le nom par défaut.
        verbose : int
            Niveau de verbosité (0, 1 ou 2).
        withoutX : bool
            Si True, n'enregistre pas la vérité terrain dans le CSV.

        Raises
        ------
        ParamError
            Si ``verbose`` n'est pas dans ``{0, 1, 2}``, si ``N`` n'est pas
            un entier strictement positif, ou si ``sKey`` est négatif.
        PKFError
            Si la construction des paramètres ou du modèle échoue.
        """
        if verbose not in (0, 1, 2):
            raise ParamError("verbose must be 0, 1 or 2.")

        self.verbose = verbose

        self.model_name = model_name
        self.N = N
        self.sKey = sKey
        self.withoutX = withoutX

        if data_file_name is None:
            data_file_name = self.default_filename()
        self.data_file_name = data_file_name

        self.base_dir = os.path.join(".", "data")
        self.datafile_dir = os.path.join(self.base_dir, "datafile")

        self._validate_inputs()
        self.param = self._build_parameters()

    # ---------------------------------------------------------
    # Abstract methods
    # ---------------------------------------------------------

    @abstractmethod
    def default_filename(self) -> str:
        pass

    @abstractmethod
    def create_model(self):
        pass

    @abstractmethod
    def create_param(self, dim_x, dim_y, params):
        pass

    @abstractmethod
    def create_pkf(self):
        pass

    # ---------------------------------------------------------
    # Shared logic
    # ---------------------------------------------------------

    def _validate_inputs(self) -> None:
        """
        Valide les paramètres d'entrée communs à tous les simulateurs.

        Raises
        ------
        ParamError
            Si ``N`` n'est pas un entier strictement positif,
            ou si ``sKey`` est strictement négatif.
        """
        if not (isinstance(self.N, int) and self.N > 0):
            raise ParamError(f"N must be a strictly positive integer, got {self.N!r}.")
        if self.sKey is not None and self.sKey < 0:
            raise ParamError(
                f"sKey must be None or a non-negative integer, got {self.sKey!r}."
            )

    def _build_parameters(self):
        """
        Instancie le modèle et construit l'objet de paramètres.

        Returns
        -------
        ParamLinear | ParamNonLinear
            L'objet de paramètres construit.

        Raises
        ------
        PKFError
            Si la création du modèle ou des paramètres échoue (wrapping
            de toute exception levée par ``create_model`` ou
            ``create_param``).
        """

        try:
            model = self.create_model()
        except PKFError:
            raise  # déjà typée — on laisse remonter telle quelle
        except Exception as e:
            raise PKFError(f"Failed to create model {self.model_name!r}.") from e

        params = model.get_params().copy()
        dim_x = params.pop("dim_x")
        dim_y = params.pop("dim_y")

        try:
            param = self.create_param(dim_x, dim_y, params)
        except PKFError:
            raise  # CovarianceError, ParamError, etc. — on laisse remonter
        except Exception as e:
            raise PKFError(
                f"Failed to build parameters for model {self.model_name!r}."
            ) from e

        if self.verbose > 1:
            param.summary()

        return param

    def run(self) -> None:
        """
        Exécute la simulation et sauvegarde les données dans un fichier CSV.

        Raises
        ------
        PKFError
            Si la création du filtre ou la simulation échoue.
        OSError
            Si la sauvegarde du fichier CSV échoue (permissions, espace disque, etc.).
        """

        try:
            pkf = self.create_pkf()
        except PKFError:
            raise
        except Exception as e:
            raise PKFError(
                f"Failed to create PKF for model {self.model_name!r}."
            ) from e

        try:
            list_data = pkf.simulate_N_data(N=self.N)
        except PKFError:
            raise
        except Exception as e:
            raise FilterError(
                f"Simulation failed after instantiation of PKF "
                f"for model {self.model_name!r}."
            ) from e

        df = data_to_dataframe(
            list_data,
            self.param.dim_x,
            self.param.dim_y,
            withoutX=self.withoutX,
        )

        os.makedirs(self.datafile_dir, exist_ok=True)

        filepath = os.path.join(self.datafile_dir, self.data_file_name)
        save_dataframe_to_csv(df, filepath)  # OSError remonte naturellement
