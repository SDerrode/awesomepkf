"""
Back-compat re-exports for the legacy ``base_model_linear`` module.

The implementation now lives in:
- :mod:`prg.models.linear._base`     — :class:`BaseModelLinear`
- :mod:`prg.models.linear._amq`      — :class:`LinearAmQ`
- :mod:`prg.models.linear._sigma`    — :class:`LinearSigma`
- :mod:`prg.models.linear._dynamics` — :class:`DynamicsMixin`
- :mod:`prg.models.linear._symbolic` — :class:`SymbolicMixin`
- :mod:`prg.models.linear._plotting` — :class:`PlottingMixin`

This thin shim is kept so existing imports
(``from prg.models.linear.base_model_linear import LinearAmQ``) keep
working.
"""

from prg.models.linear._amq import LinearAmQ
from prg.models.linear._base import BaseModelLinear
from prg.models.linear._sigma import LinearSigma

__all__ = ["BaseModelLinear", "LinearAmQ", "LinearSigma"]
