#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import importlib
import inspect
import pkgutil
import traceback
from pathlib import Path

from prg.models.nonLinear.base_model_nonLinear import BaseModelNonLinear

# tous les modules sont importables
p = Path(__file__).parent
__all__ = [f.stem for f in p.glob("*.py") if not f.name.startswith("_")]


class ModelFactoryNonLinear:
    """Fabrique automatique : découvre et instancie tous les modèles du dossier."""

    _registry: dict[str, type] = {}

    _EXCLUDED_MODULES = {
        "base_model_nonLinear",
        "base_model_fxhx",
        "base_model_gxgy",
    }

    @classmethod
    def _discover_models(cls) -> None:
        """
        Scan the package directory and register all subclasses of BaseModelNonLinear.
        """

        # éviter de rescanner si déjà fait
        if cls._registry:
            return

        package_dir = Path(__file__).parent
        package_name = __package__

        for module_info in pkgutil.iter_modules([str(package_dir)]):

            module_name = module_info.name

            if module_name in cls._EXCLUDED_MODULES:
                continue

            full_module_name = f"{package_name}.{module_name}"

            try:
                module = importlib.import_module(full_module_name)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to import module '{full_module_name}': "
                    f"{type(e).__name__}: {e}\n"
                    f"{''.join(traceback.format_exc())}"
                ) from e

            # recherche des classes dans le module
            for _, obj in inspect.getmembers(module, inspect.isclass):

                # ignorer les classes importées
                if obj.__module__ != module.__name__:
                    continue

                if not issubclass(obj, BaseModelNonLinear):
                    continue

                if obj is BaseModelNonLinear:
                    continue

                # ignorer les classes abstraites (intermédiaires non instanciables)
                if inspect.isabstract(obj):
                    continue

                cls._registry[obj.MODEL_NAME] = obj

    @classmethod
    def create(cls, name):
        cls._discover_models()
        return cls._registry[name]()

    @classmethod
    def list_models(cls):
        cls._discover_models()
        return sorted(cls._registry.keys())
