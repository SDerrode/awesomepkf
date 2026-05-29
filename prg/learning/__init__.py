"""Parameter learning from data — currently limited to the 1D linear PMM."""

from prg.learning.pmm_moments import (
    estimate_pmm_params,
    pmm_to_linear_params,
    validate_pmm,
)

__all__ = [
    "estimate_pmm_params",
    "pmm_to_linear_params",
    "validate_pmm",
]
