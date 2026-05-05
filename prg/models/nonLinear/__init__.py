import importlib
import inspect
import pkgutil
import traceback
from pathlib import Path

from prg.models.nonLinear.base_model_nonLinear import BaseModelNonLinear

# all modules are importable
p = Path(__file__).parent
__all__ = [f.stem for f in p.glob("*.py") if not f.name.startswith("_")]


class ModelFactoryNonLinear:
    """Automatic factory: discovers and instantiates all models in the directory."""

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

        # avoid rescanning if already done
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

            # search for classes in the module
            for _, obj in inspect.getmembers(module, inspect.isclass):

                # ignore imported classes
                if obj.__module__ != module.__name__:
                    continue

                if not issubclass(obj, BaseModelNonLinear):
                    continue

                if obj is BaseModelNonLinear:
                    continue

                # ignore abstract classes (non-instantiable intermediates)
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
