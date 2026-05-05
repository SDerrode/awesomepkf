import importlib
import inspect
import pkgutil
import traceback
from pathlib import Path

from prg.models.linear.base_model_linear import BaseModelLinear, LinearAmQ, LinearSigma

p = Path(__file__).parent
__all__ = [f.stem for f in p.glob("*.py") if not f.name.startswith("_")]


class ModelFactoryLinear:
    """Automatic factory: discovers and instantiates all models in the directory."""

    _registry: dict[str, type] = {}

    @classmethod
    def _discover_models(cls) -> None:
        """Scans all modules in this package and registers subclasses of BaseModelLinear."""

        # avoid rescanning if already done
        if cls._registry:
            return

        package_dir = Path(__file__).parent
        package_name = __package__

        for module_info in pkgutil.iter_modules([str(package_dir)]):

            full_module_name = f"{package_name}.{module_info.name}"

            try:
                module = importlib.import_module(full_module_name)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to import module '{full_module_name}': "
                    f"{type(e).__name__}: {e}\n"
                    f"{''.join(traceback.format_exc())}"
                ) from e

            for _, obj in inspect.getmembers(module, inspect.isclass):

                # ignore imported classes
                if obj.__module__ != module.__name__:
                    continue

                if not issubclass(obj, BaseModelLinear):
                    continue

                if obj in (BaseModelLinear, LinearAmQ, LinearSigma):
                    continue

                # ignore abstract classes (non-instantiable intermediates)
                if inspect.isabstract(obj):
                    continue

                cls._registry[obj.MODEL_NAME] = obj

    @classmethod
    def create(cls, name: str) -> BaseModelLinear:
        cls._discover_models()
        key = name.strip()
        if key not in cls._registry:
            raise ValueError(
                f"Unknown model: '{key}'. "
                f"Available: {list(cls._registry.keys())}"
            )
        return cls._registry[key]()

    @classmethod
    def list_models(cls) -> list[str]:
        cls._discover_models()
        return sorted(cls._registry.keys())
