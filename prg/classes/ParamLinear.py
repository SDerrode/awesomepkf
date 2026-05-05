from typing import Any

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from prg.classes.matrix_diagnostics import CovarianceMatrix, StabilityMatrix
from prg.utils.exceptions import CovarianceError, NumericalError, ParamError

__all__ = ["ParamLinear"]


# ----------------------------------------------------------------------
# ParamLinear class
# ----------------------------------------------------------------------


class ParamLinear:
    """
    Manage PKF parameters with optional debug checks.

    Attributes:
        verbose: logging level
        dim_x, dim_y, dim_xy: state and observation dimensions
        kwargs: models parameters
    """

    def __init__(self, verbose: int, dim_x: int, dim_y: int, **kwargs) -> None:
        """
        Initialises the linear PKF filter parameters.

        Parameters
        ----------
        verbose : int
            Verbosity level (0, 1 or 2).
        dim_x : int
            State dimension, must be a strictly positive integer.
        dim_y : int
            Observation dimension, must be a strictly positive integer.
        **kwargs
            Model parameters (matrices A, B, mQ, mz0, Pz0, etc.).

        Raises
        ------
        ParamError
            If ``dim_x``, ``dim_y`` are not strictly positive integers,
            or if ``verbose`` does not belong to ``{0, 1, 2}``.
        ParamError
            If the number of ``kwargs`` parameters does not match any
            known parametrisation (12 or 14 keys).
        CovarianceError
            If a covariance matrix is not positive definite
            during the consistency check.
        """

        if __debug__:
            if not (isinstance(dim_x, int) and dim_x > 0):
                raise ParamError("dim_x must be a strictly positive integer.")
            if not (isinstance(dim_y, int) and dim_y > 0):
                raise ParamError("dim_y must be a strictly positive integer.")
            if verbose not in [0, 1, 2]:
                raise ParamError("verbose must be 0, 1 or 2.")

        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_xy = dim_x + dim_y
        self.verbose = verbose

        # Two ways to construct the object
        if len(kwargs.keys()) == 15:  # parametrization (A, mQ, mz0, Pz0)
            self.constructorFrom_AB_mQ(
                kwargs["A"], kwargs["B"], kwargs["mQ"], kwargs["mz0"], kwargs["Pz0"]
            )
        elif (
            len(kwargs.keys()) == 17
        ):  # parametrization (sxx, syy, a, b, c, d, e) --> Sigma
            self.constructorFrom_Sigma(
                kwargs["sxx"],
                kwargs["syy"],
                kwargs["a"],
                kwargs["b"],
                kwargs["c"],
                kwargs["d"],
                kwargs["e"],
            )
        else:
            raise ParamError(
                f"Le modèle n'est pas bien paramétré : {list(kwargs.keys())}. "
                f"Attendu 12 ou 14 clés, reçu {len(kwargs.keys())}."
            )

        # Common parameters
        self.augmented = kwargs["augmented"]
        self.pairwiseModel = kwargs["pairwiseModel"]
        self.g = kwargs["g"]
        self.f = kwargs["f"]
        self.h = kwargs["h"]

        # UPKF-specific parameters
        self.alpha = kwargs["alpha"]
        self.beta = kwargs["beta"]
        self.kappa = kwargs["kappa"]
        self.lambda_ = kwargs["lambda_"]

        # EPKF-specific parameters
        self.jacobiens_g = kwargs["jacobiens_g"]

        if __debug__:
            self._check_consistency()

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"<ParamLinear(dim_y={self.dim_y}, dim_x={self.dim_x}, "
            f"augmented={self.augmented}, verbose={self.verbose}>"
        )

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    def constructorFrom_AB_mQ(
        self,
        A: np.ndarray,
        B: np.ndarray,
        mQ: np.ndarray,
        mz0: np.ndarray,
        Pz0: np.ndarray,
    ) -> None:
        """
        Constructs the parameters from matrices (A, B, mQ, mz0, Pz0).

        Raises
        ------
        NumericalError
            If matrix ``A`` is not stable (eigenvalues outside the
            unit disc).
        """
        self._A = np.array(A, dtype=float)
        stab = StabilityMatrix(self._A)
        if not stab.is_valid():
            stab.summary()
            raise NumericalError(
                "La matrice A n'est pas stable (valeurs propres hors du disque unité).",
                matrix_name="A",
            )

        self._B = np.array(B, dtype=float)
        self._mQ = np.array(mQ, dtype=float)
        self._mz0 = np.array(mz0, dtype=float)
        self._Pz0 = np.array(Pz0, dtype=float)

        # self._update_Sigma_from_A_B_mQ()

    def constructorFrom_Sigma(
        self,
        sxx: np.ndarray,
        syy: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        d: np.ndarray,
        e: np.ndarray,
    ) -> None:
        """
        Constructs the parameters from the sub-blocks of the Sigma matrix.

        Raises
        ------
        CovarianceError
            If the Cholesky factorisation of ``Q1`` fails during the
        computation of ``A`` and ``mQ``.
        """
        self._sxx, self._syy = np.array(sxx), np.array(syy)
        self._a, self._b, self._c, self._d, self._e = map(np.array, [a, b, c, d, e])

        self._update_A_B_mQ_from_Sigma()

    # ------------------------------------------------------------------
    # Update derived matrices
    # ------------------------------------------------------------------
    def _update_A_B_mQ_from_Sigma(self) -> None:
        """
        Raises
        ------
        CovarianceError
            If the Cholesky factorisation of ``Q1`` fails.
        """
        self._Q1 = np.block([[self._sxx, self._b.T], [self._b, self._syy]])
        self._Q2 = np.block([[self._a, self._e], [self._d, self._c]])
        self._Sigma = np.block([[self._Q1, self._Q2.T], [self._Q2, self._Q1]])

        try:
            c, low = cho_factor(self._Q1)
        except Exception as e:
            raise CovarianceError(
                "Cholesky factorisation of Q1 failed — Q1 may not be positive definite.",
                matrix_name="Q1",
            ) from e

        self._A = self._Q2 @ cho_solve((c, low), np.eye(self.dim_xy))
        self._B = np.eye(self.dim_xy)
        self._mQ = self._Q1 - self._A @ self._Q2.T

        self._mz0 = np.zeros((self.dim_xy, 1))
        self._Pz0 = self._Q1.copy()

    def _update_Sigma_from_A_B_mQ(self) -> None:
        """
        Raises
        ------
        NumericalError
            If the relative error between ``mQ`` and ``Q1 - A @ Q2^T`` exceeds
            the ``EPS_REL`` threshold.
        """

        return

        # self._Q1 = solve_discrete_lyapunov(self._A, self._mQ)
        # self._Q2 = self._A @ self._Q1
        # self._Sigma = np.block([[self._Q1, self._Q2.T], [self._Q2, self._Q1]])

        # if __debug__:
        #     Q_est = self._Q1 - self._A @ self._Q2.T
        #     diff = self._mQ - Q_est
        #     rel_error = np.linalg.norm(diff) / (np.linalg.norm(self._mQ) + EPS_ABS)
        #     if rel_error > EPS_REL:
        #         raise NumericalError(
        #             f"Incohérence détectée : Q ≉ Q1 - A Q2^T "
        #             f"(erreur relative = {rel_error:.2e}).",
        #             matrix_name="mQ",
        #         )

        # # Sub-blocks
        # self._a = self._Sigma[self.dim_xy : self.dim_xy + self.dim_x, : self.dim_x]
        # self._b = self._Sigma[self.dim_x : self.dim_xy, : self.dim_x]
        # self._c = self._Sigma[
        #     self.dim_xy + self.dim_x : 2 * self.dim_xy, self.dim_x : self.dim_xy
        # ]
        # self._d = self._Sigma[self.dim_xy + self.dim_x : 2 * self.dim_xy, : self.dim_x]
        # self._e = self._Sigma[
        #     self.dim_xy : self.dim_xy + self.dim_x, self.dim_x : self.dim_xy
        # ]
        # self._sxx = self._Sigma[: self.dim_x, : self.dim_x]
        # self._syy = self._Sigma[self.dim_x : self.dim_xy, self.dim_x : self.dim_xy]

    # ------------------------------------------------------------------
    # Consistency checks
    # ------------------------------------------------------------------
    def _check_consistency(self) -> None:
        """
        Checks that all covariance matrices are positive definite.

        Raises
        ------
        CovarianceError
            If any of the matrices is not positive semi-definite.
        """

        listMatrix = []
        if not self.augmented:
            listMatrix = [("mQ", self._mQ), ("Pz0", self._Pz0)]

        # if self.augmented:
        #     listMatrix = [("Q1", self._Q1), ("sxx", self._sxx), ("syy", self._syy)]
        # else:
        #     listMatrix = [
        #         # ("Q1", self._Q1),
        #         # ("sxx", self._sxx),
        #         # ("syy", self._syy),
        #         ("mQ", self._mQ),
        #         # ("Sigma", self._Sigma),
        #         ("Pz0", self._Pz0),
        #     ]

        for name, arr in listMatrix:
            report = CovarianceMatrix(arr).check()
            if not report.is_valid:
                raise CovarianceError(
                    f"Matrix {name!r} is not positive semi-definite.",
                    matrix_name=name,
                )

    # ------------------------------------------------------------------
    # Getters / Setters and Properties
    # ------------------------------------------------------------------
    @property
    def A(self) -> np.ndarray:
        return self._A

    @A.setter
    def A(self, new_A: np.ndarray) -> None:
        """
        Raises
        ------
        NumericalError
            If the update of Sigma from A fails.
        CovarianceError
            If the consistency check fails after the update.
        """
        self._A = np.array(new_A, dtype=float)
        self._update_Sigma_from_A_B_mQ()
        if __debug__:
            self._check_consistency()

    @property
    def B(self) -> np.ndarray:
        return self._B

    @B.setter
    def B(self, new_B: np.ndarray) -> None:
        """
        Raises
        ------
        NumericalError
            If the update of Sigma from B fails.
        CovarianceError
            If the consistency check fails after the update.
        """
        self._B = np.array(new_B, dtype=float)
        self._update_Sigma_from_A_B_mQ()
        if __debug__:
            self._check_consistency()

    @property
    def mQ(self) -> np.ndarray:
        return self._mQ

    @mQ.setter
    def mQ(self, new_Q: np.ndarray) -> None:
        """
        Raises
        ------
        NumericalError
            If the update of Sigma from mQ fails.
        CovarianceError
            If the consistency check fails after the update.
        """
        self._mQ = np.array(new_Q, dtype=float)
        self._update_Sigma_from_A_B_mQ()
        if __debug__:
            self._check_consistency()

    @property
    def mz0(self) -> np.ndarray:
        return self._mz0

    @property
    def Pz0(self) -> np.ndarray:
        return self._Pz0

    # @property
    # def sxx(self) -> np.ndarray:
    #     return self._sxx

    # @property
    # def syy(self) -> np.ndarray:
    #     return self._syy

    # @property
    # def a(self) -> np.ndarray:
    #     return self._a

    # @property
    # def b(self) -> np.ndarray:
    #     return self._b

    # @property
    # def c(self) -> np.ndarray:
    #     return self._c

    # @property
    # def d(self) -> np.ndarray:
    #     return self._d

    # @property
    # def e(self) -> np.ndarray:
    #     return self._e

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    def summary(self) -> None:
        def fmt(M: Any) -> str:
            return np.array2string(M, formatter={"float_kind": lambda x: f"{x:6.2f}"})

        print("=== ParamLinear Summary ===")
        print(f"dim_x={self.dim_x}, dim_y={self.dim_y}, verbose={self.verbose}\n")
        print("A:\n", fmt(self.A))
        print("B:\n", fmt(self.B))
        print("mQ:\n", fmt(self.mQ))
        print("mz0:\n", fmt(self.mz0))
        print("Pz0:\n", fmt(self.Pz0))
        print("========================\n")
