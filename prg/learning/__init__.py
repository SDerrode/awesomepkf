"""Parameter learning from data.

- Method-of-moments estimator for the 1D linear PMM (:mod:`pmm_moments`).
- Partial EM for the linear-Gaussian pairwise model — noise covariance only,
  transition ``A`` assumed known (:mod:`em_partial_noise`).
"""

from prg.learning.em_partial_noise import EMNoiseResult, estimate_noise_em
from prg.learning.pmm_moments import (
    estimate_pmm_params,
    pmm_to_linear_params,
    validate_pmm,
)

__all__ = [
    "EMNoiseResult",
    "estimate_noise_em",
    "estimate_pmm_params",
    "pmm_to_linear_params",
    "validate_pmm",
]
