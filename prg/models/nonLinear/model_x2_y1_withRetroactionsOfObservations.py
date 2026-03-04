#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from prg.models.nonLinear.base_model_nonLinear import BaseModelNonLinear
from prg.models.Generate_MatrixCov import generate_block_matrix
from prg.exceptions import NumericalError

__all__ = ["ModelX2Y1_withRetroactionsOfObservations"]


class ModelX2Y1_withRetroactionsOfObservations(BaseModelNonLinear):
    """
    Nonlinear model with retro-actions of observations and of states.
    The model includes additive Gaussian process and observation noises.
    """

    MODEL_NAME: str = "x2_y1_withRetroactionsOfObservations"

    def __init__(self) -> None:
        super().__init__(dim_x=2, dim_y=1, model_type="nonlinear")

        self.a, self.b, self.c, self.d, self.e, self.f = 1.0, 0.8, 0.05, 0.9, 0.30, 0.6

        # corrections sur le modèle car le rayon spectrale diverge
        self.a = 0.95  # au lieu de 1.0 — système stable
        self.b = 0.10  # le modèle est contractant car ‖A‖₂ ≈ 0.99
        # ou bien
        # self.a=0.7
        # self.b=0.5

        try:

            Q = np.array([[0.03, 0.0], [0.0, 0.03]])
            R = np.array([[0.03]])
            M = np.zeros(
                (self.dim_x, self.dim_y)
            )  # M=0 : bruits décorrélés (cas standard)
            self.mQ = np.block([[Q, M], [M.T, R]])

            # self.mQ = generate_block_matrix(
            #     self._randMatrices.rng, self.dim_x, self.dim_y, 0.03
            # )

            # self.mQ = np.block([[Q, M], [M.T, R]])

            # # DEBUG temporaire — à supprimer après diagnostic
            # Q = self.mQ[: self.dim_x, : self.dim_x]
            # M = self.mQ[: self.dim_x, self.dim_x :]
            # R = self.mQ[self.dim_x :, self.dim_x :]
            # R_inv = np.linalg.inv(R)
            # P_prime_x = Q - M @ R_inv @ M.T
            # A = np.array([[self.a, self.b], [0.0, self.d]])
            # print(f"[DEBUG mQ] Q:\n{Q}")
            # print(f"[DEBUG mQ] M:\n{M}")
            # print(f"[DEBUG mQ] R: {R}")
            # print(f"[DEBUG mQ] P'_x = Q - M R⁻¹ Mᵀ:\n{P_prime_x}")
            # print(f"[DEBUG mQ] λ_min(P'_x) = {np.linalg.eigvalsh(P_prime_x).min():.4g}")
            # print(f"[DEBUG A]  ‖A‖₂ = {np.linalg.norm(A, 2):.4g}")
            # print(f"[DEBUG A]  valeurs propres A = {np.linalg.eigvals(A)}")
            # # Stabilité globale : norme de la matrice de transition augmentée
            # # avec couplage via MRinv
            # MRinv = M @ R_inv
            # print(f"[DEBUG mQ] MRinv:\n{MRinv}")
            # input("ATTENTE")

            self.mz0 = self._randMatrices.rng.standard_normal((self.dim_xy, 1))
            self.Pz0 = generate_block_matrix(
                self._randMatrices.rng, self.dim_x, self.dim_y, 0.05
            )
        except (ValueError, np.exceptions.AxisError) as e:
            raise NumericalError(
                f"[{self.MODEL_NAME}] Initialization failed: {e}"
            ) from e

    # ------------------------------------------------------------------
    def _gx(self, x, y, t, u, dt):
        """
        Nonlinear state function with retro-action on observation.
        """
        try:
            with np.errstate(all="raise"):
                x1, x2 = x.flatten()
                y1 = y.flatten()[0]
                t1, t2 = t.flatten()

                return np.array(
                    [
                        [self.a * x1 + self.b * x2 + self.c * np.tanh(y1) + t1],
                        [self.d * x2 + self.e * np.sin(y1) + t2],
                    ]
                )
        except FloatingPointError as e:
            raise NumericalError(
                f"[{self.MODEL_NAME}] _gx: floating point error at x={x}, y={y}, t={t}: {e}"
            ) from e
        except (IndexError, ValueError) as e:
            raise NumericalError(
                f"[{self.MODEL_NAME}] _gx: array access error at x={x}, y={y}, t={t}: {e}"
            ) from e

    # ------------------------------------------------------------------
    def _gy(self, x, y, t, u, dt):
        """
        Nonlinear observation function with retro-action on previous observation.
        """
        try:
            with np.errstate(all="raise"):
                # return np.array([[x[0, 0] ** 2 + self.f * y[0, 0] + u[0, 0]]])
                return np.array(
                    [[x[0, 0] ** 2 / (1.0 + x[0, 0] ** 2) + self.f * y[0, 0] + u[0, 0]]]
                )
        except FloatingPointError as e:
            raise NumericalError(
                f"[{self.MODEL_NAME}] _gy: floating point error at x={x}, y={y}, u={u}: {e}"
            ) from e
        except (IndexError, ValueError) as e:
            raise NumericalError(
                f"[{self.MODEL_NAME}] _gy: array access error at x={x}, y={y}, u={u}: {e}"
            ) from e

    # ------------------------------------------------------------------
    def _g(self, x, y, t, u, dt):
        """
        Combined state and observation using Wojciech's formulation.
        """
        if __debug__:
            assert x.shape == (2, 1), f"x must be (2,1), got {x.shape}"
            assert y.shape == (1, 1), f"y must be (1,1), got {y.shape}"
            assert t.shape == (2, 1), f"t must be (2,1), got {t.shape}"
            assert u.shape == (1, 1), f"u must be (1,1), got {u.shape}"
            assert isinstance(dt, (float, int)), "dt must be a float"

        try:
            gx_val = self._gx(x, y, t, u, dt)
            gy_val = self._gy(x, y, t, u, dt)
            return np.vstack((gx_val, gy_val))
        except NumericalError:
            raise
        except ValueError as e:
            raise NumericalError(
                f"[{self.MODEL_NAME}] _g: shape mismatch during vstack: {e}"
            ) from e

    # ------------------------------------------------------------------
    def _jacobiens_g(self, x, y, t, u, dt):
        """
        Jacobians of combined state and observation function.
        """
        if __debug__:
            assert x.shape == (2, 1), f"x must be (2,1), got {x.shape}"
            assert y.shape == (1, 1), f"y must be (1,1), got {y.shape}"
            assert t.shape == (2, 1), f"t must be (2,1), got {t.shape}"
            assert u.shape == (1, 1), f"u must be (1,1), got {u.shape}"
            assert isinstance(dt, (float, int)), "dt must be a float"

        try:
            with np.errstate(all="raise"):
                x1 = x.flatten()[0]
                y1 = y.flatten()[0]

                An = np.array(
                    [
                        [self.a, self.b, self.c * (1.0 - np.tanh(y1) ** 2)],
                        [0.0, self.d, self.e * np.cos(y1)],
                        [2.0 * x1 / (1.0 + x1**2) ** 2, 0.0, self.f],
                    ]
                )
                Bn = np.eye(self.dim_xy)

            return An, Bn

        except FloatingPointError as e:
            raise NumericalError(
                f"[{self.MODEL_NAME}] _jacobiens_g: floating point error at x={x}, y={y}: {e}"
            ) from e
        except (IndexError, ValueError) as e:
            raise NumericalError(
                f"[{self.MODEL_NAME}] _jacobiens_g: array construction error: {e}"
            ) from e
