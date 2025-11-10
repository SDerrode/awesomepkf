import importlib
import pkgutil
from pathlib import Path
from .base_model_nonLinear import BaseModelNonLinear

class ModelFactory:
    """Fabrique automatique : découvre et instancie tous les modèles du dossier."""

    _registry = {}

    @classmethod
    def _discover_models(cls):
        """Scanne tous les modules dans ce paquet et enregistre les sous-classes de BaseModelNonLinear."""
        package_dir = Path(__file__).parent
        for _, module_name, _ in pkgutil.iter_modules([str(package_dir)]):
            if module_name == "base_model":
                continue
            module = importlib.import_module(f"{__package__}.{module_name}")
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, type) and issubclass(attr, BaseModelNonLinear) and attr is not BaseModelNonLinear:
                    name = getattr(attr, "MODEL_NAME", attr.__name__.lower().replace("model", ""))
                    cls._registry[name] = attr

    @classmethod
    def create(cls, name: str) -> BaseModelNonLinear:
        """Crée un modèle par son nom."""
        if not cls._registry:
            cls._discover_models()
        key = name.strip()
        if key not in cls._registry:
            raise ValueError(f"Modèle inconnu: '{key}'. "
                             f"Disponibles: {list(cls._registry.keys())}")
        return cls._registry[key]()

    @classmethod
    def list_models(cls):
        """Retourne la liste des modèles disponibles."""
        if not cls._registry:
            cls._discover_models()
        return list(cls._registry.keys())