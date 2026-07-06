"""Parameter learning from data.

- Method-of-moments estimator for the 1D linear PMM (:mod:`pmm_moments`).
- Partial EM for the linear-Gaussian pairwise model — noise covariance only,
  transition ``A`` assumed known (:mod:`em_partial_noise`).
- Partial EM for the couple dynamics blocks ``A_xy`` (back-action) and ``A_yy``
  (observation memory), with ``A_xx``, ``A_yx``, ``Q`` fixed, plus a
  likelihood-ratio test for back-action (:mod:`em_partial_dynamics`).
"""

from prg.learning.em_partial_dynamics import (
    BackActionLRT,
    EMDynamicsResult,
    back_action_lrt,
    estimate_dynamics_em,
)
from prg.learning.em_partial_noise import EMNoiseResult, estimate_noise_em
from prg.learning.pmm_moments import (
    estimate_pmm_params,
    pmm_to_linear_params,
    validate_pmm,
)

__all__ = [
    "BackActionLRT",
    "EMDynamicsResult",
    "EMNoiseResult",
    "back_action_lrt",
    "estimate_dynamics_em",
    "estimate_noise_em",
    "estimate_pmm_params",
    "pmm_to_linear_params",
    "validate_pmm",
]
