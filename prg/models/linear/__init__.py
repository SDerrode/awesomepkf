#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import importlib
import pkgutil
from pathlib import Path

from prg.models.linear.base_model_linear import BaseModelLinear, LinearAmQ, LinearSigma

# tous les modules sont importables
p = Path(__file__).parent
__all__ = [f.stem for f in p.glob("*.py") if not f.name.startswith("_")]


class ModelFactoryLinear:
    """Fabrique automatique : découvre et instancie tous les modèles du dossier."""

    _registry = {}

    @classmethod
    def _discover_models(cls):
        """Scanne tous les modules dans ce paquet et enregistre les sous-classes de BaseModelLinear."""
        package_dir = Path(__file__).parent
        for _, module_name, _ in pkgutil.iter_modules([str(package_dir)]):
            if module_name in ["base_model_linear", "LinearSigma", "LinearAmQ"]:
                continue
            module = importlib.import_module(f"{__package__}.{module_name}")
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and (issubclass(attr, LinearAmQ) or issubclass(attr, LinearSigma))
                    and attr is not BaseModelLinear
                    and attr is not LinearAmQ
                    and attr is not LinearSigma
                ):

                    name = getattr(
                        attr, "MODEL_NAME", attr.__name__.lower().replace("model", "")
                    )
                    cls._registry[name] = attr

    @classmethod
    def create(cls, name: str) -> BaseModelLinear:
        if not cls._registry:
            cls._discover_models()

        key = name.strip()

        if key not in cls._registry:
            raise ValueError(
                f"Modèle inconnu: '{key}'. "
                f"Disponibles: {list(cls._registry.keys())}"
            )

        return cls._registry[key]()

    @classmethod
    def list_models(cls):
        """Retourne la liste des modèles disponibles."""
        if not cls._registry:
            cls._discover_models()
        return list(cls._registry.keys())
