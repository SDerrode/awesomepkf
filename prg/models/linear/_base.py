"""BaseModelLinear — abstract base for the linear-model variants.

Composes :class:`DynamicsMixin`, :class:`SymbolicMixin`, and
:class:`PlottingMixin` for the standard evaluation / display surface;
concrete subclasses (:class:`LinearAmQ`, :class:`LinearSigma`) provide
the construction of the (A, B, mQ) matrices.
"""

from prg.models.linear._dynamics import DynamicsMixin
from prg.models.linear._plotting import PlottingMixin
from prg.models.linear._symbolic import SymbolicMixin

__all__ = ["BaseModelLinear"]


class BaseModelLinear(DynamicsMixin, SymbolicMixin, PlottingMixin):
    """
    Base class for linear models.

    Two possible parametrisations:
      1. 'linear_AmQ' : dynamics z_{n+1} = A z_n + B noise, with covariance Q
      2. 'linear_Sigma' : parametrisation via variances sxx, syy and coefficients a, b, c, d, e
    """

    def __init__(self, dim_x, dim_y, model_type, augmented=False, pairwiseModel=True):
        if not isinstance(dim_x, int) or dim_x <= 0:
            raise ValueError("dim_x must be a positive integer")
        if not isinstance(dim_y, int) or dim_y <= 0:
            raise ValueError("dim_y must be a positive integer")
        if model_type not in ("linear_AmQ", "linear_Sigma"):
            raise ValueError("model_type must be 'linear_AmQ' or 'linear_Sigma'")

        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_xy = dim_x + dim_y
        self.model_type = model_type
        self.augmented = augmented
        self.pairwiseModel = pairwiseModel

        # UPKF specific parameters
        self.alpha = 0.25
        self.beta = 2.0
        self.kappa = 0.0
        # Option 1 — classic Wan & Merwe values
        # self.alpha = 1e-3
        # self.beta = 2.0
        # self.kappa = 0.0  # -> lambda_ ≈ -2 + ε, Wm[0] ≈ -1, Wc[0] ≈ 1
        # Option 2 — cubature (symmetric, all positive weights)
        # self.alpha = 1.0
        # self.beta  = 0.0
        # self.kappa = 0.0   # -> lambda_ = 0, Wm = Wc = 1/(2n)

        self.lambda_ = self.alpha**2 * (self.dim_x + self.kappa) - self.dim_x

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        n = cls.__name__
        cls.MODEL_NAME = n[0].lower() + n[1:]

    def __repr__(self):
        return f"{self.__class__.__name__}(dim_x={self.dim_x}, dim_y={self.dim_y}, type={self.model_type})"
