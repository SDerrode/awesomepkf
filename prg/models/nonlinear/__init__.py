"""
Nonlinear models package.

Two parallel ways of declaring a model:

1. **Registry**: simple models live in :data:`NONLINEAR_CONFIGS`. The
   factory builds a :class:`_FuncModelFxHx` or :class:`_FuncModelGxGy`
   instance from the spec.
2. **Class file**: models that need a custom constructor (parameters),
   a method override, or substitution-based augmented logic stay as
   one-class-per-file. The factory discovers them via package scan,
   exactly as before.

Usage
-----
::

    from prg.models.nonLinear import ModelFactoryNonLinear
    model = ModelFactoryNonLinear.create("model_x1_y1_Sinus_classic")
"""

import importlib
import inspect
import pkgutil
import traceback
from pathlib import Path
from typing import ClassVar

from prg.models.nonLinear.base_model_fxhx import BaseModelFxHx
from prg.models.nonLinear.base_model_gxgy import BaseModelGxGy
from prg.models.nonLinear.base_model_nonLinear import BaseModelNonLinear
from prg.models.nonLinear.configs import NONLINEAR_CONFIGS, NonLinearSpec

# ----------------------------------------------------------------------
# Generic spec-driven model classes
# ----------------------------------------------------------------------


class _FuncModelFxHx(BaseModelFxHx):
    """Generic FxHx model whose ``symbolic_model`` is delegated to a spec callable."""

    def __init__(self, spec: NonLinearSpec) -> None:
        self._spec = spec
        for k, v in spec.attrs.items():
            setattr(self, k, v)

        super().__init__(dim_x=spec.dim_x, dim_y=spec.dim_y, model_type="nonlinear")

        if spec.init_hook is not None:
            spec.init_hook(self)
        else:
            self.mQ, self.mz0, self.Pz0 = self._init_random_params(
                spec.dim_x, spec.dim_y, spec.val_max, seed=None
            )

    def symbolic_model(self, sx, st, su):
        return self._spec.symbolic_model(sx, st, su)


class _FuncModelGxGy(BaseModelGxGy):
    """Generic GxGy model whose ``symbolic_model`` is delegated to a spec callable."""

    def __init__(self, spec: NonLinearSpec) -> None:
        self._spec = spec
        for k, v in spec.attrs.items():
            setattr(self, k, v)

        super().__init__(dim_x=spec.dim_x, dim_y=spec.dim_y, model_type="nonlinear")

        if spec.init_hook is not None:
            spec.init_hook(self)
        else:
            self.mQ, self.mz0, self.Pz0 = self._init_random_params(
                spec.dim_x, spec.dim_y, spec.val_max, seed=None
            )

    def symbolic_model(self, sx, sy, st, su):
        return self._spec.symbolic_model(sx, sy, st, su)


_GENERIC_BUILDERS = {
    "fxhx": _FuncModelFxHx,
    "gxgy": _FuncModelGxGy,
}


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------


# all modules are importable
p = Path(__file__).parent
__all__ = [f.stem for f in p.glob("*.py") if not f.name.startswith("_")] + [
    "ModelFactoryNonLinear", "NONLINEAR_CONFIGS", "NonLinearSpec",
]


class ModelFactoryNonLinear:
    """Factory: registry first (NONLINEAR_CONFIGS), then class discovery."""

    _registry: ClassVar[dict[str, type]] = {}

    _EXCLUDED_MODULES: ClassVar[set[str]] = {
        "base_model_nonLinear",
        "base_model_fxhx",
        "base_model_gxgy",
        "configs",
    }

    @classmethod
    def _discover_models(cls) -> None:
        """Scan the package for class-based models (those NOT in the registry)."""

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

            for _, obj in inspect.getmembers(module, inspect.isclass):
                if obj.__module__ != module.__name__:
                    continue
                if not issubclass(obj, BaseModelNonLinear):
                    continue
                if obj is BaseModelNonLinear:
                    continue
                if inspect.isabstract(obj):
                    continue

                cls._registry[obj.MODEL_NAME] = obj

    @classmethod
    def create(cls, name: str):
        # 1. Registry first
        if name in NONLINEAR_CONFIGS:
            spec = NONLINEAR_CONFIGS[name]
            builder_cls = _GENERIC_BUILDERS[spec.form]
            instance = builder_cls(spec)
            instance.MODEL_NAME = name
            return instance

        # 2. Fall back to class-based discovery
        cls._discover_models()
        if name not in cls._registry:
            raise ValueError(
                f"Unknown model: '{name}'. Available: {cls.list_models()}"
            )
        return cls._registry[name]()

    @classmethod
    def list_models(cls) -> list[str]:
        cls._discover_models()
        return sorted(set(cls._registry.keys()) | set(NONLINEAR_CONFIGS.keys()))
