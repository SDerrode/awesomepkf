"""
Registry of linear-model configurations.

Each entry is keyed by the canonical model name and declares one of four
variants:

- ``"AQ_classic"``   — defines (F, C, H, D); A and B are built as
  block matrices, mQ is generated randomly with ``val_max``.
- ``"AQ_pairwise"``  — defines A directly; mQ is generated randomly.
- ``"AQ_augmented"`` — references another pairwise model via
  ``based_on``; the augmented form is built via
  :meth:`LinearAmQ.classic2pairwise`.
- ``"Sigma_pairwise"`` — defines the (sxx, syy, a, b, c, d, e)
  parametrisation directly.

Adding a new model is one entry — no new file or class needed.
"""

import numpy as np

__all__ = ["LINEAR_CONFIGS"]


LINEAR_CONFIGS: dict[str, dict] = {
    # ------------------------------------------------------------------
    # AQ_classic — (F, C, H, D) + block construction
    # ------------------------------------------------------------------
    "model_x1_y1_AQ_classic": {
        "variant": "AQ_classic",
        "dim_x": 1, "dim_y": 1,
        "F": np.array([[0.8]]),
        "C": np.array([[0.25]]),
        "H": np.array([[0.4]]),
        "D": np.array([[0.35]]),
        "val_max": 0.90,
        "diag_only": False,
    },
    "model_x2_y2_AQ_classic": {
        "variant": "AQ_classic",
        "dim_x": 2, "dim_y": 2,
        "F": np.array([[0.6, 0.1], [0.1, 0.5]]),
        "C": np.array([[0.15, 0.05], [0.10, 0.15]]),
        "H": np.array([[0.6, 0.20], [0.15, 0.7]]),
        "D": np.array([[0.2, 0.15], [0.15, 0.25]]),
        "val_max": 0.40,
        "diag_only": True,
    },
    "model_x3_y1_AQ_classic": {
        "variant": "AQ_classic",
        "dim_x": 3, "dim_y": 1,
        "F": np.array([[0.6, 0.1, 0.2], [0.15, 0.6, 0.15], [0.1, 0.2, 0.7]]),
        "C": np.array([[0.2, 0.15, 0.15], [0.15, 0.2, 0.15], [0.15, 0.15, 0.2]]),
        "H": np.array([[0.4, 0.6, 0.4]]),
        "D": np.array([[0.25]]),
        "val_max": 0.35,
        "diag_only": False,
    },

    # ------------------------------------------------------------------
    # AQ_pairwise — A defined directly
    # ------------------------------------------------------------------
    "model_x1_y1_AQ_pairwise": {
        "variant": "AQ_pairwise",
        "dim_x": 1, "dim_y": 1,
        "A": np.array([[0.95, -0.5], [0.3, 0.85]]),
        "val_max": 0.15,
    },
    "model_x2_y2_AQ_pairwise": {
        "variant": "AQ_pairwise",
        "dim_x": 2, "dim_y": 2,
        "A": np.array([
            [0.6323337679269881,  -0.09843546284224247, -0.20273794002607562, 0.1434159061277705],
            [-0.09843546284224246, 0.6323337679269883,   0.14341590612777047, -0.20273794002607565],
            [-0.028683181225554088, -0.009452411994784882, 0.2040417209908735, 0.05019556714471971],
            [-0.009452411994784863, -0.028683181225554123, 0.05019556714471966, 0.20404172099087356],
        ]),
        "val_max": 1.0,
    },
    "model_x3_y1_AQ_pairwise": {
        "variant": "AQ_pairwise",
        "dim_x": 3, "dim_y": 1,
        "A": np.array([
            [0.6399176954732511,  -0.1502057613168724,  0.07818930041152264, -0.18518518518518517],
            [0.26851851851851855,  0.5462962962962961, -0.09259259259259262, -0.08333333333333329],
            [0.22119341563786007,  0.17798353909465012, 0.36625514403292186, -0.06481481481481483],
            [-0.2777777777777778,  0.055555555555555546, -0.11111111111111113, 0.49999999999999994],
        ]),
        "val_max": 0.15,
    },

    # ------------------------------------------------------------------
    # AQ_augmented — derived from a pairwise model
    # ------------------------------------------------------------------
    "model_x1_y1_AQ_augmented": {
        "variant": "AQ_augmented",
        "based_on": "model_x1_y1_AQ_pairwise",
    },
    "model_x2_y2_AQ_augmented": {
        "variant": "AQ_augmented",
        "based_on": "model_x2_y2_AQ_pairwise",
    },
    "model_x3_y1_AQ_augmented": {
        "variant": "AQ_augmented",
        "based_on": "model_x3_y1_AQ_pairwise",
    },

    # ------------------------------------------------------------------
    # Sigma_pairwise — (sxx, syy, a, b, c, d, e)
    # ------------------------------------------------------------------
    "model_x1_y1_Sigma_pairwise": {
        "variant": "Sigma_pairwise",
        "dim_x": 1, "dim_y": 1,
        "sxx": np.array([[1]]),
        "syy": np.array([[1]]),
        "a":   np.array([[0.5]]),
        "b":   np.array([[0.3]]),
        "c":   np.array([[0.04]]),
        "d":   np.array([[0.05]]),
        "e":   np.array([[0.05]]),
    },
    "model_x2_y2_Sigma_pairwise": {
        "variant": "Sigma_pairwise",
        "dim_x": 2, "dim_y": 2,
        "sxx": np.array([[1.0, 0.4], [0.4, 1.0]]),
        "syy": np.array([[1.0, 0.3], [0.3, 1.0]]),
        "a":   np.array([[0.5, 0.2], [0.2, 0.5]]),
        "b":   np.array([[0.6, 0.2], [0.2, 0.6]]),
        "c":   np.array([[0.2, 0.1], [0.1, 0.2]]),
        "d":   np.array([[0.1, 0.05], [0.05, 0.1]]),
        "e":   np.array([[0.2, 0.15], [0.15, 0.2]]),
    },
    "model_x3_y1_Sigma_pairwise": {
        "variant": "Sigma_pairwise",
        "dim_x": 3, "dim_y": 1,
        "sxx": np.array([[1.0, 0.4, 0.4], [0.4, 1.0, 0.4], [0.4, 0.4, 1.0]]),
        "syy": np.array([[1.0]]),
        "a":   np.array([[0.5, 0.1, 0.2], [0.4, 0.6, 0.2], [0.4, 0.4, 0.5]]),
        "b":   np.array([[0.6, 0.2, 0.4]]),
        "c":   np.array([[0.30]]),
        "d":   np.array([[0.0, 0.0, 0.0]]),
        "e":   np.array([[0.20], [0.15], [0.25]]),
    },
}
