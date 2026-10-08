"""Classical vs. pairwise smoothing, computed EXACTLY (SPL letter, Sec. IV, rebuilt).

No Monte Carlo enters the reported numbers. For a Gaussian pairwise Markov chain
(GPMC) Z_n = (X_n, Y_n), Z_{n+1} = A Z_n + W_{n+1}, W ~ N(0, R), Z_0 ~ N(0, P0),
the joint law of (x_{0:N}, y_{0:N}) is Gaussian with a covariance we build exactly.
Any model m (A_m, R_m, P0) fed to a fixed-interval smoother returns the linear
estimate  x_hat = K_m y,  K_m = Sig^m_xy (Sig^m_yy)^{-1},  and its own posterior
covariances P^m_n (diagonal blocks of Sig^m_xx - K_m Sig^m_yx). Against the TRUE
law,

    MSE   = 1/(N+1) sum_n tr E_true[e_n e_n^T],          e = x - K_m y,
    NEES  = 1/(N+1) sum_n tr((P^m_n)^{-1} E_true[e_n e_n^T]) / p   (calibrated = 1),

both exact. A short Monte Carlo with the awesomepkf library smoother
(Linear_PKS_VAR) validates the two formulas and the per-step covariances, for the main
models and (`mc_validate_unstable`) for class-C members with unstable observation memory.

Comparison design. Models are
referred to by NAME (the figure panels are (a) MSE penalty and (b) NEES):
  truth   : scalar GPMC, A(rho) = [[0.6, 0.4 rho], [0.3, 0.4]],
            R = [[0.1, 0.05], [0.05, 0.1]] FIXED (correlated noise present and
            representable by the classical class), P0 = I, N = 400.
  pairwise            : the true model.
  classical_projected : KL projection of the true stationary transition onto
                        C = {A^xy = 0} (R, incl. R^xy, and A^yy free). C contains the
                        textbook model y_n = H x_n + v_n; back-action is the only
                        pairwise feature it excludes. Solved by iterated feasible GLS
                        (SUR) and cross-checked against a closed form (C factorizes as
                        p(x'|x) p(y'|x', x, y)) and multi-start BFGS.
  textbook_projected  : KL projection onto A = [[F, 0], [H F, 0]],
                        R = [[Q, Q H^T], [H Q, H Q H^T + V]] (closed form, H in R^{q x p},
                        cross-checked by multi-start BFGS).
  ablated             : the true model with A^xy set to 0, all other blocks kept.
All four models use the prior N(0, I). All main values are exact for N = 400 with that
prior; N -> infinity values (stationary Wiener smoother, frequency domain) are added.

What "projected" means: the transition KL E_stat[-log N(Z'; A_c Z, R_c)] is the
large-sample limit of maximum likelihood with x OBSERVED in training (complete-data,
supervised fit). It is not the limit of EM / ML on y alone. The pairwise smoother uses
the projection onto the full GPMC class, i.e. the truth. The penalty of
classical_projected is therefore the cost of excluding back-action WHEN the classical
model is fitted by that rule, not the cost of the class: the class C itself contains, for
this truth and at every rho, a one-parameter family of models whose N -> infinity smoother
equals the pairwise one exactly (closed form below, `classC_match_member`); with R rescaled
they are also calibrated, and their finite-N penalty is a boundary effect that decays like
1/N. These members fit the transition much worse than the projection (larger KL rate, e.g.
7-21x at rho = 0.5 and 1), and for rho > 0 none reproduces the true y law, so neither a
complete-data nor a y-only likelihood fit converges to them; reaching them needs a
discriminative fit (smoothing error minimised with x observed in training).

Stability. A smoother uses only the model's transition densities (y is observed): in
information form x_hat = -Jxx^-1 Jxy y, posterior covariance Jxx^-1, with J the block-
tridiagonal joint precision, and for N -> infinity H_m(w) = -Jxx(w)^-1 Jxy(w). Both are well
defined for ANY observation memory A^yy once the state dynamics A^xx is stable; no stationary
law of the model is needed. The family's A^yy = b is stable only for rho < 1.2069 (and
b = 3.5 for the x1y1 robustness model with R^yy x4). Capacity is therefore reported for the
class as specified (A^xy = 0; A^xx stable, A^yy free: 'C_free_Ayy') and, for reference, for
its stable sub-class (spectral radius < 1: 'C_stable'), whose gaps appear exactly where b
leaves the unit disc and are approached only at the stability boundary. All capacity members
are evaluated in information form (`model_gain_precision`, `model_symbol`); for stable models
this coincides with the covariance route to rounding (`precision_route_check` in the JSON).
The main comparison (rho grid, robustness metrics, MC check) is unchanged and uses the
covariance route.

Reference-only extras (JSON): the old class {A^xy = R^xy = 0} (letter's original
comparator), the old design in which rho scaled A^xy and R^xy together, a rho = 1
decomposition (back-action only / correlated noise only), rho beyond 1 up to the
stability limit (rho = 2), N -> infinity metrics of every model, the best-in-class
capacity (N -> infinity best member of C, textbook and old class, multi-start local
searches run in a process pool, and a calibrated member evaluated exactly at N = 400 and
1600), and the finite-N MSE-best STABLE member of C and of the textbook class (oracle:
exact N = 400 MSE minimized over the class by local search, so an UPPER bound on the
best-in-class finite-N MSE; mean only, NEES not controlled; needs the true x and, at large
rho, implies a y law far from the truth). The interpretation bullets and caption notes in
design are generated from the results of the run (`derive_text`).

Run (awesomepkf's ``prg`` must be importable, only for the MC check):
    PYTHONPATH=/path/to/awesomePKF python -B experiments/classical_vs_pairwise_exact.py
Options: --no-mc (skip the library MC check), --no-oracle (skip the finite-N MSE-best
         search), --no-capacity (skip the best-in-class capacity section),
         --seeds S (MC seeds, default 200), --workers W (processes for the capacity
         searches, default min(8, cpu count); results do not depend on it),
         --replot (figure only, from the JSON).
Runtime: about 25 min on an Apple-silicon laptop with 8 worker processes (see runtime_* in the JSON);
         with --no-oracle --no-mc --no-capacity about 15 s.
Outputs: experiments/classical_vs_pairwise_results.json,
         figures/classical_vs_pairwise.pdf (+ .png preview).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import scipy
from scipy.linalg import cho_factor, cho_solve, solve_discrete_lyapunov
from scipy.optimize import brentq, least_squares, minimize, minimize_scalar

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
JSON_OUT = HERE / "classical_vs_pairwise_results.json"
FIG_OUT = ROOT / "figures" / "classical_vs_pairwise.pdf"

# ----------------------------------------------------------------------------
# Truth (scalar design)
# ----------------------------------------------------------------------------
P_DIM, Q_DIM = 1, 1
AXX, AXY_BAR, AYX, AYY = 0.6, 0.4, 0.3, 0.4
R_TRUE = np.array([[0.10, 0.05], [0.05, 0.10]])
N_STEPS = 400                       # N + 1 = 401 states
RHOS = np.linspace(0.0, 1.0, 9)
RHOS_EXT = [1.25, 1.5, 1.75, 1.9]   # stability limit of A(rho) is rho = 2


def A_true(rho: float) -> np.ndarray:
    return np.array([[AXX, AXY_BAR * rho], [AYX, AYY]])


# Robustness models: the three of awesomePKF/experiments/classical_vs_couple_multi.py,
# A^xy at its full value, noise cross block S kept, R^yy optionally x4 (R^xy unchanged).
MODELS = {
    "x1y1": {"dx": 1, "dy": 1,
             "Axx": [[0.6]], "Ayx": [[0.3]], "Ayy": [[0.4]], "Axy": [[0.4]],
             "Qxx": [[0.10]], "Qyy": [[0.10]], "S": [[0.05]]},
    "x2y2": {"dx": 2, "dy": 2,
             "Axx": [[0.5, 0.1], [0.0, 0.45]], "Ayx": [[0.3, 0.0], [0.1, 0.2]],
             "Ayy": [[0.35, 0.05], [0.0, 0.3]], "Axy": [[0.35, 0.1], [0.05, 0.3]],
             "Qxx": [[0.10, 0.0], [0.0, 0.10]], "Qyy": [[0.10, 0.0], [0.0, 0.10]],
             "S": [[0.04, 0.0], [0.0, 0.04]]},
    "x2y1": {"dx": 2, "dy": 1,
             "Axx": [[0.5, 0.1], [0.0, 0.45]], "Ayx": [[0.3, 0.1]], "Ayy": [[0.4]],
             "Axy": [[0.4], [0.2]], "Qxx": [[0.10, 0.0], [0.0, 0.10]],
             "Qyy": [[0.10]], "S": [[0.05], [0.03]]},
}


def model_from_cfg(cfg: dict, ryy_scale: float):
    Axx, Ayx = np.array(cfg["Axx"], float), np.array(cfg["Ayx"], float)
    Ayy, Axy = np.array(cfg["Ayy"], float), np.array(cfg["Axy"], float)
    S = np.array(cfg["S"], float)
    A = np.block([[Axx, Axy], [Ayx, Ayy]])
    R = np.block([[np.array(cfg["Qxx"], float), S],
                  [S.T, ryy_scale * np.array(cfg["Qyy"], float)]])
    return A, R, cfg["dx"], cfg["dy"]


# ----------------------------------------------------------------------------
# Exact Gaussian machinery
# ----------------------------------------------------------------------------
def sym(M: np.ndarray) -> np.ndarray:
    return 0.5 * (M + M.T)


def joint_cov(A: np.ndarray, R: np.ndarray, P0: np.ndarray, N: int) -> np.ndarray:
    """Covariance of Z_{0:N} (time-major, (N+1)d x (N+1)d): Cov(Z_n, Z_m) = A^{n-m} Sig_m."""
    d, T = A.shape[0], N + 1
    Sig = np.empty((T, d, d))
    Sig[0] = sym(P0)
    for n in range(N):
        Sig[n + 1] = sym(A @ Sig[n] @ A.T + R)
    Apow = np.empty((T, d, d))
    Apow[0] = np.eye(d)
    for k in range(1, T):
        Apow[k] = A @ Apow[k - 1]
    G = np.empty((T, d, T, d))
    for m in range(T):
        blocks = Apow[:T - m] @ Sig[m]                    # Cov(Z_n, Z_m), n = m..N
        G[m:, :, m, :] = blocks
        G[m, :, m:, :] = blocks.transpose(2, 0, 1)        # Cov(Z_m, Z_n) = blocks^T
    return G.reshape(T * d, T * d)


def split_xy(G: np.ndarray, p: int, q: int, T: int):
    d = p + q
    G4 = G.reshape(T, d, T, d)
    xx = G4[:, :p, :, :p].reshape(T * p, T * p)
    xy = G4[:, :p, :, p:].reshape(T * p, T * q)
    yy = G4[:, p:, :, p:].reshape(T * q, T * q)
    return xx, xy, yy


def exact_metrics(Gt, Gm, p: int, q: int, N: int, keep_blocks: bool = False) -> dict:
    """Exact MSE / expected NEES of the smoother built on model m, under the true law."""
    T = N + 1
    txx, txy, tyy = split_xy(Gt, p, q, T)
    mxx, mxy, myy = split_xy(Gm, p, q, T)
    c = cho_factor(myy)
    K = cho_solve(c, mxy.T).T                                    # (Tp, Tq)
    K3 = K.reshape(T, p, T * q)
    # model's own posterior covariances (diagonal blocks)
    mxx_b = mxx.reshape(T, p, T, p)[np.arange(T), :, np.arange(T), :]
    mxy3 = mxy.reshape(T, p, T * q)
    Pm = mxx_b - np.einsum("nia,nja->nij", K3, mxy3)
    Pm = 0.5 * (Pm + Pm.transpose(0, 2, 1))
    # true error covariance blocks E_true[e_n e_n^T]
    txx_b = txx.reshape(T, p, T, p)[np.arange(T), :, np.arange(T), :]
    txy3 = txy.reshape(T, p, T * q)
    KT3 = (K @ tyy).reshape(T, p, T * q)
    cross = np.einsum("nia,nja->nij", K3, txy3)
    E = txx_b - cross - cross.transpose(0, 2, 1) + np.einsum("nia,nja->nij", KT3, K3)
    E = 0.5 * (E + E.transpose(0, 2, 1))
    mse_n = np.trace(E, axis1=1, axis2=2)
    nees_n = np.trace(np.linalg.solve(Pm, E), axis1=1, axis2=2) / p
    out = {"mse": float(mse_n.mean()), "nees": float(nees_n.mean()),
           "mean_tr_P_model": float(np.trace(Pm, axis1=1, axis2=2).mean()),
           "cond_Sig_yy_model": float(np.linalg.cond(myy))}
    if keep_blocks:
        out["P_blocks"], out["E_blocks"] = Pm, E
    return out


# ----------------------------------------------------------------------------
# KL projections of the true stationary transition
# ----------------------------------------------------------------------------
def stationary(At: np.ndarray, Rt: np.ndarray):
    S = sym(solve_discrete_lyapunov(At, Rt))
    return S, At @ S                                  # Sig, Gamma1 = E[Z' Z^T]


def C_of(Ac: np.ndarray, S: np.ndarray, G1: np.ndarray) -> np.ndarray:
    """E_stat[(Z' - Ac Z)(Z' - Ac Z)^T] = R_t + (A_t - Ac) Sig (A_t - Ac)^T."""
    return sym(S - G1 @ Ac.T - Ac @ G1.T + Ac @ S @ Ac.T)


def kl_rate(At, Rt, Ac, Rc) -> float:
    """E_stat KL( N(A_t z, R_t) || N(A_c z, R_c) ), nats per step."""
    S, G1 = stationary(At, Rt)
    d = At.shape[0]
    C = C_of(Ac, S, G1)
    return float(0.5 * (np.linalg.slogdet(Rc)[1] - np.linalg.slogdet(Rt)[1]
                        + np.trace(np.linalg.solve(Rc, C)) - d))


def classC_mask(p: int, q: int) -> np.ndarray:
    mask = np.ones((p + q, p + q), bool)
    mask[:p, p:] = False                              # A^xy = 0
    return mask


def project_classC_fgls(At, Rt, p, q, tol=1e-15, maxit=5000) -> dict:
    """argmin_{A in C, R>0} E_stat[-log N(Z'; A Z, R)]: iterated feasible GLS (SUR)."""
    S, G1 = stationary(At, Rt)
    d = p + q
    free = classC_mask(p, q).flatten(order="F")
    W = np.eye(d)
    A = np.zeros((d, d))
    it = 0
    for it in range(1, maxit + 1):
        H = np.kron(S, W)                             # (Sig (x) W) vec(A)
        g = (W @ G1).flatten(order="F")
        a = np.zeros(d * d)
        a[free] = np.linalg.solve(H[np.ix_(free, free)], g[free])
        Anew = a.reshape(d, d, order="F")
        W = np.linalg.inv(C_of(Anew, S, G1))
        step = float(np.max(np.abs(Anew - A)))
        A = Anew
        if step < tol and it > 2:
            break
    C = C_of(A, S, G1)
    grad = 2.0 * np.linalg.solve(C, A @ S - G1)       # d logdet C / dA
    return {"A": A, "R": C, "iters": it, "last_step": step,
            "grad_free_max": float(np.max(np.abs(grad[classC_mask(p, q)]))),
            "objective_logdetC": float(np.linalg.slogdet(C)[1])}


def project_classC_closed(At, Rt, p, q):
    """Closed form: C = {p(x'|x) linear-Gaussian} x {p(y'|x',x,y) linear-Gaussian}."""
    S, G1 = stationary(At, Rt)
    Sxx, G1xx = S[:p, :p], G1[:p, :p]
    Axx = G1xx @ np.linalg.inv(Sxx)
    Rxx = sym(Sxx - Axx @ G1xx.T)
    # regress y' on w = (x', z)
    Cww = np.block([[S[:p, :p], G1[:p, :]], [G1[:p, :].T, S]])
    Cyw = np.hstack([S[p:, :p], G1[p:, :]])
    B = Cyw @ np.linalg.inv(Cww)
    V = sym(S[p:, p:] - B @ Cyw.T)
    c, bx, by = B[:, :p], B[:, p:2 * p], B[:, 2 * p:]
    A = np.zeros((p + q, p + q))
    A[:p, :p] = Axx
    A[p:, :p] = c @ Axx + bx
    A[p:, p:] = by
    R = np.block([[Rxx, Rxx @ c.T], [c @ Rxx, c @ Rxx @ c.T + V]])
    return A, sym(R)


def _chol_params(L_flat, k):
    L = np.zeros((k, k))
    L[np.tril_indices(k)] = L_flat
    L[np.diag_indices(k)] = np.exp(np.diag(L))
    return L


def check_classC_multistart(At, Rt, p, q, ref_obj, n_starts=20, seed=0) -> dict:
    """Multi-start BFGS on f(A) = logdet C(A) over the free entries (profile in R)."""
    S, G1 = stationary(At, Rt)
    mask = classC_mask(p, q)
    rng = np.random.default_rng(seed)

    def f(a):
        A = np.zeros_like(At)
        A[mask] = a
        return np.linalg.slogdet(C_of(A, S, G1))[1]

    def g(a):
        A = np.zeros_like(At)
        A[mask] = a
        C = C_of(A, S, G1)
        return (2.0 * np.linalg.solve(C, A @ S - G1))[mask]

    best = []
    for _ in range(n_starts):
        a0 = rng.uniform(-1.5, 1.5, mask.sum())
        r = minimize(f, a0, jac=g, method="BFGS", options={"gtol": 1e-12, "maxiter": 5000})
        best.append((r.fun, r.x))
    objs = np.array([b[0] for b in best])
    return {"n_starts": n_starts, "min_obj": float(objs.min()),
            "max_obj": float(objs.max()),
            "min_obj_minus_fgls": float(objs.min() - ref_obj),
            "all_starts_within_1e-9": bool(np.all(objs - ref_obj < 1e-9))}


def hessian_eigs_classC(At, Rt, p, q, A_opt, h=1e-6) -> list:
    S, G1 = stationary(At, Rt)
    mask = classC_mask(p, q)

    def g(a):
        A = np.zeros_like(At)
        A[mask] = a
        C = C_of(A, S, G1)
        return (2.0 * np.linalg.solve(C, A @ S - G1))[mask]

    a0 = A_opt[mask]
    k = a0.size
    H = np.zeros((k, k))
    for i in range(k):
        e = np.zeros(k)
        e[i] = h
        H[:, i] = (g(a0 + e) - g(a0 - e)) / (2 * h)
    return np.linalg.eigvalsh(sym(H)).tolist()


def textbook_AR(F, H, Qx, V):
    A = np.block([[F, np.zeros((F.shape[0], H.shape[0]))],
                  [H @ F, np.zeros((H.shape[0], H.shape[0]))]])
    R = np.block([[Qx, Qx @ H.T], [H @ Qx, H @ Qx @ H.T + V]])
    return A, sym(R)


def project_textbook_closed(At, Rt, p, q):
    """KL projection onto x'=Fx+w, y=Hx+v (same time): two stationary regressions."""
    S, G1 = stationary(At, Rt)
    Sxx, G1xx = S[:p, :p], G1[:p, :p]
    F = G1xx @ np.linalg.inv(Sxx)
    Qx = sym(Sxx - F @ G1xx.T)
    H = S[p:, :p] @ np.linalg.inv(Sxx)
    V = sym(S[p:, p:] - H @ S[:p, p:])
    A, R = textbook_AR(F, H, Qx, V)
    return A, R, {"F": F, "H": H, "Q": Qx, "V": V}


def textbook_objective(theta, p, q, S, G1):
    F = theta[:p * p].reshape(p, p)
    H = theta[p * p:p * p + q * p].reshape(q, p)
    o = p * p + q * p
    nq, nv = p * (p + 1) // 2, q * (q + 1) // 2
    Lq = _chol_params(theta[o:o + nq], p)
    Lv = _chol_params(theta[o + nq:o + nq + nv], q)
    A, R = textbook_AR(F, H, Lq @ Lq.T, Lv @ Lv.T)
    C = C_of(A, S, G1)
    return float(np.linalg.slogdet(R)[1] + np.trace(np.linalg.solve(R, C)))


def check_textbook_multistart(At, Rt, p, q, A_cl, R_cl, n_starts=20, seed=1) -> dict:
    S, G1 = stationary(At, Rt)
    ref = float(np.linalg.slogdet(R_cl)[1] + np.trace(np.linalg.solve(R_cl, C_of(A_cl, S, G1))))
    rng = np.random.default_rng(seed)
    k = p * p + q * p + p * (p + 1) // 2 + q * (q + 1) // 2
    objs = []
    for _ in range(n_starts):
        th0 = rng.uniform(-1.0, 1.0, k)
        r = minimize(textbook_objective, th0, args=(p, q, S, G1), method="BFGS",
                     options={"gtol": 1e-10, "maxiter": 20000})
        objs.append(r.fun)
    objs = np.array(objs)
    return {"n_starts": n_starts, "closed_form_obj": ref, "min_obj": float(objs.min()),
            "min_obj_minus_closed": float(objs.min() - ref),
            "n_starts_reaching_closed_within_1e-8": int(np.sum(objs - ref < 1e-8))}


def project_old_class(At, Rt, p, q):
    """Letter's original class {A^xy = 0, R^xy = 0}: two separate OLS regressions."""
    S, G1 = stationary(At, Rt)
    d = p + q
    A = np.zeros((d, d))
    A[:p, :p] = G1[:p, :p] @ np.linalg.inv(S[:p, :p])
    A[p:, :] = G1[p:, :] @ np.linalg.inv(S)
    R = np.zeros((d, d))
    R[:p, :p] = S[:p, :p] - A[:p, :p] @ G1[:p, :p].T
    R[p:, p:] = S[p:, p:] - A[p:, :] @ G1[p:, :].T
    return A, sym(R)


def project_textbook_memory(At, Rt, p, q):
    """Reference only: x'=Fx+w, y'=Hx'+Dy+v (textbook + observation memory)."""
    S, G1 = stationary(At, Rt)
    Sxx, G1xx = S[:p, :p], G1[:p, :p]
    F = G1xx @ np.linalg.inv(Sxx)
    Qx = sym(Sxx - F @ G1xx.T)
    # regress y' on (x', y)
    Cww = np.block([[S[:p, :p], G1[:p, p:]], [G1[:p, p:].T, S[p:, p:]]])
    Cyw = np.hstack([S[p:, :p], G1[p:, p:]])
    B = Cyw @ np.linalg.inv(Cww)
    V = sym(S[p:, p:] - B @ Cyw.T)
    H, D = B[:, :p], B[:, p:]
    A = np.block([[F, np.zeros((p, q))], [H @ F, D]])
    R = np.block([[Qx, Qx @ H.T], [H @ Qx, H @ Qx @ H.T + V]])
    return A, sym(R)


def ablate(At, Rt, p):
    A = At.copy()
    A[:p, p:] = 0.0
    return A, Rt.copy()


# ----------------------------------------------------------------------------
# Evaluate a truth against a set of comparator models
# ----------------------------------------------------------------------------
def evaluate(At, Rt, p, q, N, comparators: dict, keep_blocks=False) -> dict:
    P0 = np.eye(p + q)
    Gt = joint_cov(At, Rt, P0, N)
    out = {"pairwise": exact_metrics(Gt, Gt, p, q, N, keep_blocks)}
    out["pairwise"]["kl_rate"] = 0.0
    for name, (Am, Rm) in comparators.items():
        Gm = joint_cov(Am, Rm, P0, N)
        out[name] = exact_metrics(Gt, Gm, p, q, N, keep_blocks)
        out[name]["kl_rate"] = kl_rate(At, Rt, Am, Rm)
    base = out["pairwise"]["mse"]
    for name in out:
        out[name]["penalty_pct"] = 100.0 * (out[name]["mse"] / base - 1.0)
    return out


def comparators_main(At, Rt, p, q) -> tuple[dict, dict]:
    fg = project_classC_fgls(At, Rt, p, q)
    Acl, Rcl = project_classC_closed(At, Rt, p, q)
    Atb, Rtb, tb = project_textbook_closed(At, Rt, p, q)
    Aab, Rab = ablate(At, Rt, p)
    Aold, Rold = project_old_class(At, Rt, p, q)
    diag = {
        "classC_fgls_iters": fg["iters"],
        "classC_fgls_grad_free_max": fg["grad_free_max"],
        "classC_fgls_vs_closed_maxabs_A": float(np.max(np.abs(fg["A"] - Acl))),
        "classC_fgls_vs_closed_maxabs_R": float(np.max(np.abs(fg["R"] - Rcl))),
        "classC_A": fg["A"].tolist(), "classC_R": fg["R"].tolist(),
        "textbook_params": {k: v.tolist() for k, v in tb.items()},
        "textbook_A": Atb.tolist(), "textbook_R": Rtb.tolist(),
        "oldclass_A": Aold.tolist(), "oldclass_R": Rold.tolist(),
    }
    comps = {"classical_projected": (fg["A"], fg["R"]),
             "textbook_projected": (Atb, Rtb),
             "ablated": (Aab, Rab),
             "oldclass_projected": (Aold, Rold)}
    return comps, diag


# ----------------------------------------------------------------------------
# N -> infinity: stationary (interior) Wiener smoother, frequency domain
# ----------------------------------------------------------------------------
def spectra(A: np.ndarray, R: np.ndarray, nw: int) -> np.ndarray:
    """S(w) = G R G^H with G = (I - A e^{-iw})^{-1}, on the nw-point midpoint grid."""
    w = 2.0 * np.pi * (np.arange(nw) + 0.5) / nw - np.pi
    z = np.exp(-1j * w)
    d = A.shape[0]
    G = np.linalg.inv(np.eye(d)[None] - A[None] * z[:, None, None])
    return G @ R @ np.conj(np.transpose(G, (0, 2, 1)))


def _ct(M: np.ndarray) -> np.ndarray:
    return np.conj(np.transpose(M, (0, 2, 1)))


def _asym_blocks(St, Sm, p):
    """Grid means of the true error spectrum and of the model's own posterior spectrum."""
    Hm = Sm[:, :p, p:] @ np.linalg.inv(Sm[:, p:, p:])
    Sxy = St[:, :p, p:]
    E = St[:, :p, :p] - Hm @ _ct(Sxy) - Sxy @ _ct(Hm) + Hm @ St[:, p:, p:] @ _ct(Hm)
    Pm = Sm[:, :p, :p] - Hm @ Sm[:, p:, :p]
    return E.mean(0), Pm.mean(0)


def asym_metrics(At, Rt, Am, Rm, p, nw0=1024, tol=1e-14, nw_max=2 ** 17) -> dict:
    """Interior (N -> inf) smoother of model m under the true law: MSE = tr E_inf and
    NEES = tr(P_inf^-1 E_inf)/p, with E_inf = (1/2pi) int E(w) dw and P_inf the model's
    own steady smoothing covariance. Midpoint rule (spectrally accurate for these analytic
    periodic integrands), doubled until the blocks move by < tol."""
    nw = nw0
    Eo, Po = _asym_blocks(spectra(At, Rt, nw), spectra(Am, Rm, nw), p)
    while True:
        nw *= 2
        En, Pn = _asym_blocks(spectra(At, Rt, nw), spectra(Am, Rm, nw), p)
        change = float(max(np.max(np.abs(En - Eo)), np.max(np.abs(Pn - Po))))
        Eo, Po = En, Pn
        if change < tol or nw >= nw_max:
            break
    E, P = sym(Eo.real), sym(Po.real)
    return {"mse": float(np.trace(E)), "nees": float(np.trace(np.linalg.solve(P, E)) / p),
            "nw": nw, "grid_change": change,
            "max_imag": float(max(np.abs(Eo.imag).max(), np.abs(Po.imag).max())),
            "E": E, "P": P}


def evaluate_asym(At, Rt, p, comparators: dict, ev_exact: dict | None = None,
                  N: int | None = None) -> dict:
    """N -> inf metrics of the pairwise smoother and of each comparator. If the exact
    finite-N blocks are given, also checks them at mid-record (n = N/2) against E_inf,
    P_inf (boundary effects decay geometrically, so they must agree to rounding)."""
    raw = {"pairwise": asym_metrics(At, Rt, At, Rt, p)}
    for name, (Am, Rm) in comparators.items():
        raw[name] = asym_metrics(At, Rt, Am, Rm, p)
    base = raw["pairwise"]["mse"]
    out = {}
    for name, a in raw.items():
        r = {"mse": a["mse"], "nees": a["nees"], "penalty_pct": 100.0 * (a["mse"] / base - 1.0),
             "nw": a["nw"], "grid_change": a["grid_change"]}
        if ev_exact is not None and "E_blocks" in ev_exact.get(name, {}):
            mid = N // 2
            r["mid_record_vs_asym_maxabs"] = float(max(
                np.max(np.abs(ev_exact[name]["E_blocks"][mid] - a["E"])),
                np.max(np.abs(ev_exact[name]["P_blocks"][mid] - a["P"]))))
        out[name] = r
    return out


# ----------------------------------------------------------------------------
# Model side in PRECISION (information) form: valid for ANY A, stable or not
# ----------------------------------------------------------------------------
# The smoother of a model m only uses its transition densities: with the joint precision J
# of Z_{0:N} (block tridiagonal: J_00 = P0^-1 + A^T R^-1 A, J_nn = R^-1 + A^T R^-1 A,
# J_NN = R^-1, J_{n+1,n} = -R^-1 A), xhat = K y with K = -Jxx^-1 Jxy and posterior covariance
# Jxx^-1. No matrix powers or stationary law are needed, so a model whose observation memory
# A^yy is unstable (y observed, so its explosion is never simulated) has a well-defined
# smoother. Likewise, for N -> inf, the interior smoother is H_m(w) = -Jxx(w)^-1 Jxy(w) with the
# symbol J(w) = M^H R^-1 M, M = I - A e^{-iw}; Jxx(w) is PD whenever A^xx has no unit-circle
# eigenvalue, whatever A^yy. For a stable model this equals S_xy S_yy^-1 (S = J^-1).
def model_gain_precision(Am, Rm, p, q, N, P0=None):
    """Exact finite-N smoother of model m: (K, diagonal blocks of the posterior covariance)."""
    d, T = p + q, N + 1
    P0 = np.eye(d) if P0 is None else P0
    Ri = sym(np.linalg.inv(sym(Rm)))
    AtRiA = Am.T @ Ri @ Am
    D = np.empty((T, d, d))
    D[:] = Ri + AtRiA
    D[0] = np.linalg.inv(P0) + AtRiA
    D[-1] = Ri
    L = -Ri @ Am                                     # J_{n+1,n}
    t, t1 = np.arange(T), np.arange(N)
    Jxx = np.zeros((T, p, T, p))
    Jxy = np.zeros((T, p, T, q))
    Jxx[t, :, t, :] = D[:, :p, :p]
    Jxy[t, :, t, :] = D[:, :p, p:]
    Jxx[t1 + 1, :, t1, :] = L[:p, :p]
    Jxx[t1, :, t1 + 1, :] = L[:p, :p].T
    Jxy[t1 + 1, :, t1, :] = L[:p, p:]
    Jxy[t1, :, t1 + 1, :] = L[p:, :p].T
    c = cho_factor(Jxx.reshape(T * p, T * p))
    K = -cho_solve(c, Jxy.reshape(T * p, T * q))
    Pfull = cho_solve(c, np.eye(T * p)).reshape(T, p, T, p)
    Pm = sym3(Pfull[t, :, t, :])
    return K, Pm


def sym3(M: np.ndarray) -> np.ndarray:
    return 0.5 * (M + np.swapaxes(M, -1, -2))


def model_symbol(A, R, p, z):
    """N -> inf model smoother on the points z (batched over leading dims of A, R):
    H_m = -Jxx^-1 Jxy and P_m = Jxx^-1, J = M^H R^-1 M, M = I - A z."""
    d = A.shape[-1]
    M = np.eye(d) - A[..., None, :, :] * z[:, None, None]
    WM = np.linalg.inv(R)[..., None, :, :] @ M
    MxH = np.conj(np.swapaxes(M[..., :, :p], -1, -2))
    Pm = np.linalg.inv(MxH @ WM[..., :, :p])
    return -Pm @ (MxH @ WM[..., :, p:]), Pm


def zgrid(nw: int) -> np.ndarray:
    w = 2.0 * np.pi * (np.arange(nw) + 0.5) / nw - np.pi     # same midpoint grid as `spectra`
    return np.exp(-1j * w)


def asym_metrics_any(At, Rt, Am, Rm, p, nw0=1024, tol=1e-14, nw_max=2 ** 17) -> dict:
    """As `asym_metrics`, with the model side in precision form (valid for unstable A^yy);
    the truth is stable. Equal to `asym_metrics` for stable models (checked in the JSON)."""
    def blocks(nw):
        St = spectra(At, Rt, nw)
        Hm, Pm = model_symbol(Am, Rm, p, zgrid(nw))
        Sxy = St[:, :p, p:]
        E = St[:, :p, :p] - Hm @ _ct(Sxy) - Sxy @ _ct(Hm) + Hm @ St[:, p:, p:] @ _ct(Hm)
        return E.mean(0), Pm.mean(0)
    nw = nw0
    Eo, Po = blocks(nw)
    while True:
        nw *= 2
        En, Pn = blocks(nw)
        change = float(max(np.max(np.abs(En - Eo)), np.max(np.abs(Pn - Po))))
        Eo, Po = En, Pn
        if change < tol or nw >= nw_max:
            break
    E, P = sym(Eo.real), sym(Po.real)
    return {"mse": float(np.trace(E)), "nees": float(np.trace(np.linalg.solve(P, E)) / p),
            "nw": nw, "grid_change": change, "E": E, "P": P}


class ExactTruth:
    """Caches the exact joint law of the (stable) truth for one (A_t, R_t, N); `eval` builds the
    model's smoother in precision form, so it accepts any model (stable or not)."""

    def __init__(self, At, Rt, p, q, N):
        self.p, self.q, self.N = p, q, N
        T = N + 1
        Gt = joint_cov(At, Rt, np.eye(p + q), N)
        txx, self.txy, self.tyy = split_xy(Gt, p, q, T)
        self.txx_b = txx.reshape(T, p, T, p)[np.arange(T), :, np.arange(T), :]
        del Gt, txx
        self.base = self._eval(At, Rt)["mse"]

    def _eval(self, Am, Rm, mid=False) -> dict:
        p, q, T = self.p, self.q, self.N + 1
        K, Pm = model_gain_precision(Am, Rm, p, q, self.N)
        K3 = K.reshape(T, p, T * q)
        cross = np.einsum("nia,nja->nij", K3, self.txy.reshape(T, p, T * q))
        KT3 = (K @ self.tyy).reshape(T, p, T * q)
        E = sym3(self.txx_b - cross - cross.transpose(0, 2, 1)
                 + np.einsum("nia,nja->nij", KT3, K3))
        out = {"mse": float(np.trace(E, axis1=1, axis2=2).mean()),
               "nees": float((np.trace(np.linalg.solve(Pm, E), axis1=1, axis2=2) / p).mean())}
        if mid:
            out["E_mid"], out["P_mid"] = E[self.N // 2], Pm[self.N // 2]
        return out

    def eval(self, Am, Rm, mid_vs: dict | None = None) -> dict:
        """Exact MSE / NEES / penalty of model m; with mid_vs = asym_metrics_any(...) of the same
        model, also max |E_n - E_inf|, |P_n - P_inf| at n = N/2 (interior check)."""
        m = self._eval(Am, Rm, mid=mid_vs is not None)
        out = {"mse": m["mse"], "nees": m["nees"],
               "penalty_pct": 100.0 * (m["mse"] / self.base - 1.0)}
        if mid_vs is not None:
            out["mid_record_vs_asym_maxabs"] = float(max(
                np.max(np.abs(m["E_mid"] - mid_vs["E"])), np.max(np.abs(m["P_mid"] - mid_vs["P"]))))
        return out


def y_autocov(A, R, p, lags=4) -> list:
    """Implied stationary autocovariances of y, lags 0..lags-1 (y blocks of A^k Sig)."""
    S = sym(solve_discrete_lyapunov(A, R))
    out, M = [], S
    for _ in range(lags):
        out.append(M[p:, p:].tolist())
        M = A @ M
    return out


def spectral_radius(X: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(X)))) if X.size else 0.0


def y_var_ratio(Am, Rm, At, Rt, p):
    """tr Var(y) of the model's stationary law over the truth's; None if the model has no
    stationary law (spectral radius >= 1)."""
    if spectral_radius(Am) >= 1.0:
        return None
    return float(np.trace(np.array(y_autocov(Am, Rm, p, 1)[0]))
                 / np.trace(np.array(y_autocov(At, Rt, p, 1)[0])))


# ----------------------------------------------------------------------------
# Best-in-class CAPACITY (N -> inf): what the class can do, whatever the fitting rule
# ----------------------------------------------------------------------------
def scalar_laurent(A, R):
    """p = q = 1: on |z| = 1 (z = e^{-iw}) the smoother transfer function is
    S_xy/S_yy = N_xy/N_yy with N_xy = t_m1/z + t_0 + t_1 z and N_yy = n_0 + n_1 (z + 1/z),
    built from the rows of adj(I - A z): u1 = (1 - a22 z, a12 z), u2 = (a21 z, 1 - a11 z).
    (Equivalently -Jxx^-1 Jxy of the precision symbol, so it also holds for unstable A.)"""
    u1 = np.array([[1.0, -A[1, 1]], [0.0, A[0, 1]]])     # row j = (coef z^0, coef z^1)
    u2 = np.array([[0.0, A[1, 0]], [1.0, -A[0, 0]]])

    def coeffs(u):
        return (float(u[:, 0] @ R @ u2[:, 1]),                           # z^-1
                float(u[:, 0] @ R @ u2[:, 0] + u[:, 1] @ R @ u2[:, 1]),  # z^0
                float(u[:, 1] @ R @ u2[:, 0]))                           # z^+1
    t = coeffs(u1)
    _, n0, n1 = coeffs(u2)
    return t, (n0, n1)


def classC_match_member(At, Rt, a: float):
    """p = q = 1. The class-C model A = [[a, 0], [c, b]], R whose smoother transfer
    function equals the truth's, from N^m_xy = N^t_xy and N^m_yy = N^t_yy (necessary
    when N^t_xy, N^t_yy are coprime; R's overall scale is then free). With a12 = 0,
    N^m_xy = (1 - b z)(r12 + s/z), s = r11 c - r12 a, which gives
    s = t_m1, r12 - b s = t_0, -b r12 = t_1; N^m_yy then is linear in (c, r22).
    Returns (A, R, feasible, stable) or None. feasible = R > 0 and |a| < 1 (the state
    dynamics A^xx stable); the observation memory b = A^yy is NOT required to be stable
    (the smoother is defined in precision form for any b); stable = feasible and |b| < 1."""
    (tm1, t0, t1), (n0, n1) = scalar_laurent(At, Rt)
    if abs(tm1) <= 1e-14 * max(abs(t0), abs(t1)):
        if t0 == 0.0:
            return None
        b, tm1 = -t1 / t0, 0.0
    else:
        disc = t0 * t0 - 4.0 * tm1 * t1
        if disc < 0:
            return None
        b = min(((-t0 + s * np.sqrt(disc)) / (2.0 * tm1) for s in (1.0, -1.0)), key=abs)
    r12 = t0 + b * tm1
    M = np.array([[tm1 - a * r12, 1.0 + a * a], [r12, -a]])
    if abs(np.linalg.det(M)) < 1e-300:
        return None
    c, r22 = np.linalg.solve(M, [n0, n1])
    if c == 0.0:
        return None
    r11 = (tm1 + a * r12) / c
    A = np.array([[a, 0.0], [c, b]])
    R = np.array([[r11, r12], [r12, r22]])
    feasible = bool(np.all(np.linalg.eigvalsh(R) > 0) and abs(a) < 1.0)
    return A, R, feasible, bool(feasible and abs(b) < 1.0)


A_SAMPLE = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)


def classC_match_family(At, Rt, truths: dict) -> dict:
    """Scan a for feasible exact-match members; report the transfer-function match, the
    N -> inf penalty, and calibrated members (R scaled so that NEES_inf = 1) evaluated
    exactly at each finite N in `truths` ({N: ExactTruth}, precision form); also the
    calibrated member that minimises the exact MSE at the smallest N (bounded 1-D search)."""
    agrid = np.linspace(-0.999, 0.999, 1999)
    feas = [a for a in agrid
            if (m := classC_match_member(At, Rt, a)) is not None and m[2]]
    (tm1, t0, t1), _ = scalar_laurent(At, Rt)
    if not feas:
        return {"exists": False, "laurent_t": [tm1, t0, t1],
                "note": "no feasible exact-match member (R not PD for every |a| < 1)"}
    lo, hi = float(min(feas)), float(max(feas))
    step = agrid[1] - agrid[0]
    contiguous = len(feas) == int(round((hi - lo) / step)) + 1
    nw_chk = 4096
    St = spectra(At, Rt, nw_chk)
    Ht = St[:, 0, 1] / St[:, 1, 1]
    base_inf = asym_metrics(At, Rt, At, Rt, 1)["mse"]
    Ns = sorted(truths)
    Ac_kl, Rc_kl = project_classC_closed(At, Rt, 1, 1)
    kl_proj = kl_rate(At, Rt, Ac_kl, Rc_kl)

    def member(a, mid=True):
        A, R, ok, stable = classC_match_member(At, Rt, a)
        am = asym_metrics_any(At, Rt, A, R, 1)
        kappa = am["nees"]             # H invariant to R -> kappa R; NEES_inf scales 1/kappa
        am_cal = asym_metrics_any(At, Rt, A, kappa * R, 1)
        Hm, _ = model_symbol(A, R, 1, zgrid(nw_chk))
        kl = kl_rate(At, Rt, A, kappa * R)
        rec = {"a": float(a), "b": float(A[1, 1]), "stable": stable,
               "A": A.tolist(), "R_over_R11": (R / R[0, 0]).tolist(), "feasible": ok,
               "max_abs_H_m_minus_H_t": float(np.max(np.abs(Hm[:, 0, 0] - Ht))),
               "penalty_inf_pct": 100.0 * (am["mse"] / base_inf - 1.0),
               "nees_inf_unscaled": am["nees"], "kappa": float(kappa),
               "R_calibrated": (kappa * R).tolist(),
               "kl_rate_calibrated": kl,
               # undefined at rho = 0, where the projection is the truth (KL rate 0)
               "kl_rate_ratio_to_projection": kl / kl_proj if kl_proj > 1e-12 else None,
               "y_var_ratio_calibrated": y_var_ratio(A, kappa * R, At, Rt, 1)}
        for N in Ns:
            rec[f"N{N}"] = truths[N].eval(A, kappa * R, mid_vs=am_cal if (mid and N == Ns[-1])
                                          else None)
        return rec

    n0 = Ns[0]

    def mse_calibrated(a):
        A, R, _, _ = classC_match_member(At, Rt, a)
        return truths[n0].eval(A, asym_metrics_any(At, Rt, A, R, 1)["nees"] * R)["mse"]

    sample = [member(a) for a in A_SAMPLE if lo < a < hi]
    opt = minimize_scalar(mse_calibrated, bounds=(lo, hi), method="bounded",
                          options={"xatol": 1e-6})
    best = member(float(opt.x))
    # keep the better of the bounded search and the sample (guards a non-unimodal curve)
    cand = min(sample + [best], key=lambda r: r[f"N{n0}"]["mse"])
    allm = sample + [best]
    return {"exists": True, "b": float(cand["b"]), "stable": bool(cand["stable"]),
            "a_interval": [lo, hi], "a_interval_contiguous": contiguous,
            "laurent_t": [tm1, t0, t1],
            "max_abs_H_m_minus_H_t_over_family": max(r["max_abs_H_m_minus_H_t"] for r in allm),
            "max_abs_penalty_inf_pct_over_family": max(abs(r["penalty_inf_pct"]) for r in allm),
            "max_mid_record_vs_asym_N%d" % Ns[-1]: max(r[f"N{Ns[-1]}"]["mid_record_vs_asym_maxabs"]
                                                    for r in allm),
            "kl_rate_classical_projected": kl_proj,
            "kl_rate_ratio_range_over_sample": (
                [min(r["kl_rate_ratio_to_projection"] for r in sample),
                 max(r["kl_rate_ratio_to_projection"] for r in sample)] if kl_proj > 1e-12 else None),
            "sample": sample,
            "best_calibrated_at_N": n0, "best_calibrated": cand,
            "range_over_sample_penalty_pct": {
                f"N{N}": [min(r[f"N{N}"]["penalty_pct"] for r in sample),
                          max(r[f"N{N}"]["penalty_pct"] for r in sample)] for N in Ns},
            "range_over_sample_nees": {
                f"N{N}": [min(r[f"N{N}"]["nees"] for r in sample),
                          max(r[f"N{N}"]["nees"] for r in sample)] for N in Ns},
            "note": ("every member reproduces the pairwise N->inf smoother (gain and, after "
                     "R -> kappa R, calibration); members differ only at finite N. b = A^yy is "
                     "the same for all members (b = -t_1/t_0 when t_m1 = 0); 'stable' = |b| < 1. "
                     "Unstable-b members are evaluated in precision form (exact at finite N; "
                     "mid-record check vs N->inf in max_mid_record_vs_asym_*). Near the lower end "
                     "of the a-interval the member degenerates (R11 -> inf), so the sample is "
                     "restricted to a in A_SAMPLE inside the interval. best_calibrated comes from a "
                     "bounded search over the whole interval; when it ends at the upper end "
                     "(a = 0.999) the finite-N infimum is approached in the near-unit-root limit of "
                     "x.")}


# Capacity search settings. Two regions are searched for the classes with observation memory:
#   'stable'   : spectral radius of A < 1 (enforced as below), the usual stationary-model class;
#   'free_Ayy' : only the state dynamics A^xx (resp. F) stable; A^yy unrestricted (the smoother is
#                well defined in precision form for any A^yy; see model_gain_precision).
# The constrained radius (sr(A) for 'stable', sr(A^xx) for 'free_Ayy') carries an exterior
# penalty BAR_W * max(0, radius - SR_CAP_OPT) and a hard wall at SR_HARD, so the trust-region
# solver can slide along the cap; optima on the cap are continued to SR_CAP_CONT (finer grid).
SR_CAP_OPT, SR_HARD, BAR_W = 0.995, 0.9999, 100.0
SR_CAP_CONT, SR_HARD_CONT = 0.9999, 0.99999
# In information form the model's smoother H_m = -Jxx^-1 Jxy has no pole at the eigenvalues of
# A^yy (Jxx depends on A^xx, A^yx, R only), so the integrand stays smooth even when A^yy nears
# the unit circle: a 512-point grid gives ~12 digits here (final values are re-evaluated on a
# converged grid, and the search-grid error is reported as 'search_grid_abs_error_pct').
NW_OPT, NW_CONT = 512, 1024
MAX_NFEV, MAX_NFEV_CONT = 1000, 500


def _chol_free(v, k, fix_first):
    L = np.zeros((k, k))
    L[np.tril_indices(k)] = np.concatenate([[0.0], v]) if fix_first else v
    with np.errstate(over="ignore"):           # overflow -> inf -> rejected as infeasible
        L[np.diag_indices(k)] = np.exp(np.diag(L))
    return L


def _chol_pack(R, fix_first):
    L = np.linalg.cholesky(sym(R))
    L[np.diag_indices(R.shape[0])] = np.log(np.diag(L))
    v = L[np.tril_indices(R.shape[0])]
    return v[1:] if fix_first else v


class ModelClass:
    """Unconstrained parameterisation of a model class, overall R scale removed
    (the N -> inf smoother is invariant to R -> kappa R): 'C' = {A^xy = 0},
    'textbook' = {A = [[F,0],[HF,0]], R = [[Q,QH^T],[HQ,HQH^T+V]]}, 'old' = {A^xy = 0, R^xy = 0}."""

    def __init__(self, kind, p, q):
        self.kind, self.p, self.q, self.d = kind, p, q, p + q
        self.mask = classC_mask(p, q)

    def unpack(self, th):
        p, q, d = self.p, self.q, self.d
        if self.kind in ("C", "old"):
            nA = int(self.mask.sum())
            A = np.zeros((d, d))
            A[self.mask] = th[:nA]
            if self.kind == "C":
                L = _chol_free(th[nA:], d, True)
                return A, L @ L.T
            nx = p * (p + 1) // 2 - 1
            Lx, Ly = _chol_free(th[nA:nA + nx], p, True), _chol_free(th[nA + nx:], q, False)
            R = np.zeros((d, d))
            R[:p, :p], R[p:, p:] = Lx @ Lx.T, Ly @ Ly.T
            return A, R
        F = th[:p * p].reshape(p, p)
        H = th[p * p:p * p + q * p].reshape(q, p)
        o, nq = p * p + q * p, p * (p + 1) // 2 - 1
        Lq, Lv = _chol_free(th[o:o + nq], p, True), _chol_free(th[o + nq:], q, False)
        return textbook_AR(F, H, Lq @ Lq.T, Lv @ Lv.T)

    def pack(self, A, R):
        p = self.p
        R = R / R[0, 0]
        if self.kind == "C":
            return np.concatenate([A[self.mask], _chol_pack(R, True)])
        if self.kind == "old":
            return np.concatenate([A[self.mask], _chol_pack(R[:p, :p], True),
                                   _chol_pack(R[p:, p:], False)])
        Qx = R[:p, :p]
        H = np.linalg.solve(Qx, R[:p, p:]).T
        V = R[p:, p:] - H @ Qx @ H.T
        return np.concatenate([A[:p, :p].ravel(), H.ravel(), _chol_pack(Qx, True),
                               _chol_pack(V, False)])

    def radius(self, A, region):
        """The constrained spectral radius: of A ('stable') or of the state block ('free_Ayy')."""
        return spectral_radius(A) if region == "stable" else spectral_radius(A[:self.p, :self.p])


class CapProblem:
    """N -> inf penalty of a class member as a nonlinear least-squares residual:
    mean_w ||(H_m - H_t) L_yy||_F^2 / MSE_inf (S_yy = L_yy L_yy^H), H_m in precision form,
    plus one exterior-penalty residual on the constrained spectral radius."""

    def __init__(self, At, Rt, p, q, kind, region, nw, cap, hard):
        self.cls = ModelClass(kind, p, q)
        self.p, self.region, self.cap, self.hard = p, region, cap, hard
        St = spectra(At, Rt, nw)
        self.Ht = St[:, :p, p:] @ np.linalg.inv(St[:, p:, p:])
        self.Lyy = np.linalg.cholesky(St[:, p:, p:])
        self.scale = 1.0 / np.sqrt(nw * float(np.trace(_asym_blocks(St, St, p)[0]).real))
        self.nres = 2 * nw * p * q + 1
        self.bad = np.full(self.nres, np.sqrt(1e3 / self.nres))
        self.z = zgrid(nw)

    def resid_batch(self, TH):
        out = np.empty((TH.shape[0], self.nres))
        As, Rs, idx, bars = [], [], [], []
        for i, th in enumerate(TH):
            A, R = self.cls.unpack(th)
            if not (np.all(np.isfinite(A)) and np.all(np.isfinite(R))) or np.linalg.cond(R) > 1e12:
                out[i] = 2.0 * self.bad          # R must stay PD (information form uses R^-1)
                continue
            rad = self.cls.radius(A, self.region)
            if rad >= self.hard:
                out[i] = self.bad * (1.0 + rad)
                continue
            As.append(A)
            Rs.append(R)
            idx.append(i)
            bars.append(BAR_W * max(0.0, rad - self.cap))
        if idx:
            try:
                Hm, _ = model_symbol(np.array(As), np.array(Rs), self.p, self.z)
            except np.linalg.LinAlgError:      # one singular member: evaluate one by one
                for k, i in enumerate(idx):
                    out[i] = self.resid_batch_one(As[k], Rs[k], bars[k])
                return out
            D = ((Hm - self.Ht[None]) @ self.Lyy[None]).reshape(len(idx), -1) * self.scale
            res = np.concatenate([D.real, D.imag, np.array(bars)[:, None]], axis=1)
            for k, i in enumerate(idx):
                out[i] = res[k] if np.all(np.isfinite(res[k])) else self.bad
        return out

    def resid_batch_one(self, A, R, bar):
        try:
            Hm, _ = model_symbol(A[None], R[None], self.p, self.z)
        except np.linalg.LinAlgError:
            return self.bad
        D = ((Hm[0] - self.Ht) @ self.Lyy).reshape(-1) * self.scale
        r = np.concatenate([D.real, D.imag, [bar]])
        return r if np.all(np.isfinite(r)) else self.bad

    def fun(self, th):
        return self.resid_batch(th[None])[0]

    def jac(self, th):           # forward differences, all columns in one batched evaluation
        h = 1.4901161193847656e-08 * np.maximum(1.0, np.abs(th))
        R = self.resid_batch(np.vstack([th[None], th[None] + np.diag(h)]))
        return ((R[1:] - R[0]) / h[:, None]).T


def _cap_worker(args):
    """One trust-region least-squares run (module level so that it can run in a process pool)."""
    At, Rt, p, q, kind, region, th0, nw, cap, hard, max_nfev, x_scale = args
    pr = CapProblem(At, Rt, p, q, kind, region, nw, cap, hard)
    r = least_squares(pr.fun, th0, jac=pr.jac, method="trf", x_scale=x_scale, xtol=1e-15,
                      ftol=1e-15, gtol=1e-15, max_nfev=max_nfev)
    return r.x, 100.0 * float(np.sum(r.fun[:-1] ** 2)), int(r.nfev)


def _random_member(rng, kind, region, p, q):
    """Random start: stable state block, A^yy stable ('stable') or unrestricted ('free_Ayy')."""
    d = p + q
    while True:
        Axx = rng.uniform(-0.95, 0.95, (p, p)) / np.sqrt(p)
        if spectral_radius(Axx) < 0.99:
            break
    if kind == "textbook":
        X, Y = rng.normal(0.0, 0.4, (p, p)), rng.normal(0.0, 0.4, (q, q))
        return textbook_AR(Axx, rng.normal(0.0, 1.0, (q, p)), X @ X.T + 0.02 * np.eye(p),
                           Y @ Y.T + 0.02 * np.eye(q))
    if region == "stable":
        while True:
            Ayy = rng.uniform(-0.95, 0.95, (q, q)) / np.sqrt(q)
            if spectral_radius(Ayy) < 0.99:
                break
    else:
        Ayy = rng.normal(0.0, 1.5, (q, q))
    A = np.zeros((d, d))
    A[:p, :p], A[p:, p:], A[p:, :p] = Axx, Ayy, rng.normal(0.0, 0.7, (q, p))
    X = rng.normal(0.0, 0.4, (d, d))
    R = X @ X.T + 0.02 * np.eye(d)
    if kind == "old":
        R[:p, p:], R[p:, :p] = 0.0, 0.0
    return A, R


def capacity_search(At, Rt, p, q, kind, region, starts: dict, truths: dict, pool,
                    n_rand=8, n_pert=4, seed=0, continue_on_cap=True) -> dict:
    """N -> inf best member of a class within `region`, by trust-region least squares from the
    given starts, n_pert Gaussian perturbations of the first one, n_rand random members and (for
    'free_Ayy') the starts with A^yy reflected to inv(A^yy) (eigenvalues 1/lambda); structured
    starts run with both unit and Jacobian parameter scaling, random ones alternate. Runs in
    `pool`. Local search: the result is an UPPER bound on the class infimum over the region.
    If the best run sits on the cap, the 3 best runs are continued to SR_CAP_CONT on a finer
    grid ('continuation'). The best member, with R rescaled so that NEES_inf = 1, is evaluated
    exactly at each N of `truths` (one calibrated member, not a finite-N optimum)."""
    cls = ModelClass(kind, p, q)
    rng = np.random.default_rng(seed)
    probe = CapProblem(At, Rt, p, q, kind, region, 64, SR_CAP_OPT, SR_HARD)

    def ok(th):
        return probe.fun(th)[-1] == 0.0 and np.all(np.isfinite(probe.fun(th)))

    th_starts = [(name, cls.pack(A0, R0)) for name, (A0, R0) in starts.items()]
    if region == "free_Ayy" and kind != "textbook":
        for name, (A0, R0) in starts.items():
            Ayy = A0[p:, p:]
            if np.linalg.cond(Ayy) < 1e6 and spectral_radius(Ayy) > 1e-3:
                A1 = A0.copy()
                A1[p:, p:] = np.linalg.inv(Ayy)
                th_starts.append((name + "_reflected", cls.pack(A1, R0)))
    base_th = th_starts[0][1]
    for i in range(n_pert):
        while not ok(th := base_th + rng.normal(0.0, 0.3, base_th.size)):
            pass
        th_starts.append((f"pert{i}", th))
    # structured starts are run with both parameter scalings (unit and Jacobian-based: neither
    # dominates on these rugged landscapes); random starts alternate between them
    th_starts = [(f"{name}|{xs}", th, xs) for name, th in th_starts for xs in ("unit", "jac")]
    for i in range(n_rand):
        th_starts.append((f"rand{i}|{('unit', 'jac')[i % 2]}",
                          cls.pack(*_random_member(rng, kind, region, p, q)), ("unit", "jac")[i % 2]))
    jobs = [(At, Rt, p, q, kind, region, th0, NW_OPT, SR_CAP_OPT, SR_HARD, MAX_NFEV,
             1.0 if xs == "unit" else "jac") for _, th0, xs in th_starts]
    runs = [{"start": name, "end_penalty_pct": pen, "nfev": nfev, "th": x,
             "x_scale": 1.0 if xs == "unit" else "jac"}
            for (name, _, xs), (x, pen, nfev) in zip(th_starts, pool.map(_cap_worker, jobs))]
    runs.sort(key=lambda t: t["end_penalty_pct"])
    base_inf = asym_metrics(At, Rt, At, Rt, p)["mse"]
    Ns = sorted(truths)

    def describe(th, finite=()):
        A, R = cls.unpack(th)
        am = asym_metrics_any(At, Rt, A, R, p)
        kappa = am["nees"]          # H invariant to R -> kappa R, NEES_inf scales as 1/kappa
        rad = cls.radius(A, region)
        rec = {"penalty_inf_pct": 100.0 * (am["mse"] / base_inf - 1.0),
               "nees_inf_unscaled": am["nees"], "nw_final": am["nw"],
               "spectral_radius": spectral_radius(A),
               "spectral_radius_state_block": spectral_radius(A[:p, :p]),
               "eig_moduli_Ayy": sorted(np.abs(np.linalg.eigvals(A[p:, p:])).tolist()),
               "constrained_radius": rad, "at_stability_cap": bool(rad > SR_CAP_OPT - 1e-3),
               "stable": bool(spectral_radius(A) < 1.0),
               "A": A.tolist(), "R_over_R11": (R / R[0, 0]).tolist(), "kappa": float(kappa),
               "R_calibrated": (kappa * R).tolist(),
               "kl_rate_calibrated": kl_rate(At, Rt, A, kappa * R),
               "y_var_ratio_calibrated": y_var_ratio(A, kappa * R, At, Rt, p)}
        for N in finite:
            rec[f"N{N}"] = truths[N].eval(A, kappa * R)
        return rec, (A, R)

    best, best_AR = describe(runs[0]["th"])
    best["search_grid_abs_error_pct"] = abs(best["penalty_inf_pct"] - runs[0]["end_penalty_pct"])
    out = {"class": kind, "region": region, **best, "n_starts": len(runs),
           "starts_end_penalty_pct_sorted": [r_["end_penalty_pct"] for r_ in runs],
           "starts_nfev_sorted_as_above": [r_["nfev"] for r_ in runs],
           "best_start": runs[0]["start"]}
    if continue_on_cap and best["at_stability_cap"] and best["penalty_inf_pct"] > 1e-6:
        jobs = [(At, Rt, p, q, kind, region, r_["th"], NW_CONT, SR_CAP_CONT, SR_HARD_CONT,
                 MAX_NFEV_CONT, r_["x_scale"]) for r_ in runs[:3]]
        cont = sorted(pool.map(_cap_worker, jobs), key=lambda t: t[1])
        cbest, (Acb, _) = describe(cont[0][0])
        out["continuation"] = {
            "note": (f"3 best runs continued with the cap raised to {SR_CAP_CONT} (grid "
                     f"{NW_CONT}): the class infimum over the region is approached at the "
                     "boundary; value re-evaluated on the converged grid"),
            "penalty_inf_pct": cbest["penalty_inf_pct"], "constrained_radius": cbest[
                "constrained_radius"], "eig_moduli_Ayy": cbest["eig_moduli_Ayy"],
            "A": cbest["A"], "R_over_R11": cbest["R_over_R11"]}
        if cbest["penalty_inf_pct"] < best["penalty_inf_pct"]:
            out["best_value_pct"] = cbest["penalty_inf_pct"]
    out.setdefault("best_value_pct", best["penalty_inf_pct"])
    # the cap binds only if raising it lowers the value (an optimum can sit on the cap of a
    # flat valley without the cap mattering, e.g. the textbook class here)
    out["cap_active"] = bool("continuation" in out and best["penalty_inf_pct"]
                             - out["continuation"]["penalty_inf_pct"] > 1e-4)
    # finite-N calibrated member: among runs within 0.05 pp of the best N->inf penalty,
    # the one with the lowest exact MSE at the smallest N (the N->inf optimum is often a
    # flat set; this only picks a better representative, it is not a finite-N optimum)
    near = [r_ for r_ in runs if r_["end_penalty_pct"] <= runs[0]["end_penalty_pct"] + 0.05]
    cands = [describe(r_["th"], finite=Ns[:1]) for r_ in near]
    cal, (Ac, Rc) = min(cands, key=lambda c: c[0][f"N{Ns[0]}"]["mse"])
    for N in Ns[1:]:
        cal[f"N{N}"] = truths[N].eval(Ac, cal["kappa"] * Rc)
    cal["selection"] = (f"lowest exact N={Ns[0]} MSE among the {len(near)} runs within 0.05 pp "
                        "of the best N->inf penalty, R scaled so that NEES_inf = 1")
    out["calibrated_member"] = cal
    out["_AR"] = best_AR
    return out


def capacity_all(At, Rt, p, q, comps: dict, truths: dict, pool, with_old=True,
                 n_rand=8) -> dict:
    """Capacity of the textbook class (F stable), and of C (and the old class) over the
    'free_Ayy' and 'stable' regions. C is warm-started from the projections of all classes and
    from the textbook / old optima (which lie in C); the stable search also from the free
    optimum with A^yy shrunk into the unit disc or reflected (inv(A^yy)) when that is stable.
    Since C_stable is a subset of C_free_Ayy, C_free_Ayy.best_value_pct is the smaller of the two
    searches ('best_value_source'), and 'calibrated_member_best_of_C' is the lower-N=400 of the
    two calibrated members."""
    res = {"textbook": capacity_search(At, Rt, p, q, "textbook", "stable",
                                       {"textbook_projected": comps["textbook_projected"]},
                                       truths, pool, n_rand=n_rand, seed=1)}
    starts_C = {"classical_projected": comps["classical_projected"], "ablated": comps["ablated"],
                "textbook_projected": comps["textbook_projected"],
                "oldclass_projected": comps["oldclass_projected"],
                "textbook_best": res["textbook"]["_AR"]}
    if with_old:
        res["old_stable"] = capacity_search(At, Rt, p, q, "old", "stable",
                                            {"oldclass_projected": comps["oldclass_projected"]},
                                            truths, pool, n_rand=n_rand, seed=2)
        res["old_free_Ayy"] = capacity_search(
            At, Rt, p, q, "old", "free_Ayy",
            {"oldclass_projected": comps["oldclass_projected"],
             "old_stable_best": res["old_stable"]["_AR"]}, truths, pool, n_rand=n_rand, seed=4)
        starts_C["old_best"] = res["old_stable"]["_AR"]
    res["C_free_Ayy"] = capacity_search(At, Rt, p, q, "C", "free_Ayy", starts_C, truths, pool,
                                        n_rand=n_rand, seed=5)
    Af, Rf = res["C_free_Ayy"]["_AR"]
    starts_S = dict(starts_C)
    if spectral_radius(Af[p:, p:]) > 1e-3:
        A1 = Af.copy()
        A1[p:, p:] *= min(1.0, 0.98 / spectral_radius(Af[p:, p:]))
        starts_S["C_free_best_Ayy_shrunk"] = (A1, Rf)
        if np.linalg.cond(Af[p:, p:]) < 1e6:
            A2 = Af.copy()
            A2[p:, p:] = np.linalg.inv(Af[p:, p:])
            if spectral_radius(A2) < SR_CAP_OPT:
                starts_S["C_free_best_Ayy_reflected"] = (A2, Rf)
    res["C_stable"] = capacity_search(At, Rt, p, q, "C", "stable", starts_S, truths, pool,
                                      n_rand=3 * n_rand, seed=3)
    cf, cs = res["C_free_Ayy"], res["C_stable"]
    cf["best_value_source"] = "C_free_Ayy search"
    if cs["best_value_pct"] < cf["best_value_pct"]:
        cf["best_value_pct"], cf["best_value_source"] = cs["best_value_pct"], "C_stable search"
    cands = [("C_free_Ayy", cf["calibrated_member"]), ("C_stable", cs["calibrated_member"])]
    src, cm = min(cands, key=lambda t: t[1][f"N{min(truths)}"]["mse"])
    cf["calibrated_member_best_of_C"] = {"source": src, **{
        k: cm[k] for k in cm if k.startswith("N") or k in ("penalty_inf_pct", "kappa")}}
    for v in res.values():
        v.pop("_AR")
    return res


# ----------------------------------------------------------------------------
# Oracle: MSE-best member of a class (scalar case only; reference)
# ----------------------------------------------------------------------------
def _mse_only(Gt, Am, Rm, p, q, N):
    Gm = joint_cov(Am, Rm, np.eye(p + q), N)
    return exact_metrics(Gt, Gm, p, q, N)


EIG_CAP = 0.9999   # stability enforced smoothly: a = EIG_CAP * tanh(u)


def _atanh_cap(a: float) -> float:
    return float(np.arctanh(np.clip(a / EIG_CAP, -0.999999, 0.999999)))


def oracle_classC_scalar(At, Rt, N, starts: dict) -> dict:
    """MSE-best member of C = {A^xy = 0} (p=q=1): exact MSE minimized by BFGS from
    several starts. A = [[a, 0], [b, c]] is lower triangular, so its eigenvalues are
    a and c; stability is enforced by a = cap*tanh(u), c = cap*tanh(w)."""
    Gt = joint_cov(At, Rt, np.eye(2), N)
    base = exact_metrics(Gt, Gt, 1, 1, N)["mse"]

    def unpack(th):
        A = np.array([[EIG_CAP * np.tanh(th[0]), 0.0], [th[1], EIG_CAP * np.tanh(th[2])]])
        L = _chol_params(th[3:6], 2)
        return A, L @ L.T

    def pack(A, R):
        L = np.linalg.cholesky(R)
        return np.array([_atanh_cap(A[0, 0]), A[1, 0], _atanh_cap(A[1, 1]),
                         np.log(L[0, 0]), L[1, 0], np.log(L[1, 1])])

    def f(th):
        A, R = unpack(th)
        return _mse_only(Gt, A, R, 1, 1, N)["mse"]

    per_start, best = {}, None
    for name, (A0, R0) in starts.items():
        th0 = pack(A0, R0)
        r = minimize(f, th0, method="BFGS", options={"gtol": 1e-10, "maxiter": 400})
        per_start[name] = {"start_penalty_pct": 100 * (f(th0) / base - 1),
                           "end_penalty_pct": 100 * (r.fun / base - 1), "nfev": int(r.nfev)}
        if best is None or r.fun < best.fun:
            best = r
    A, R = unpack(best.x)
    m = _mse_only(Gt, A, R, 1, 1, N)
    return {"mse": m["mse"], "nees": m["nees"], "A": A.tolist(), "R": R.tolist(),
            "kl_rate": kl_rate(At, Rt, A, R), "per_start": per_start}


def oracle_textbook_scalar(At, Rt, N, starts: dict) -> dict:
    """MSE-best textbook model (p=q=1), F = cap*tanh(u), Q, V > 0, multi-start BFGS."""
    Gt = joint_cov(At, Rt, np.eye(2), N)
    base = exact_metrics(Gt, Gt, 1, 1, N)["mse"]

    def unpack(th):
        return textbook_AR(np.array([[EIG_CAP * np.tanh(th[0])]]), np.array([[th[1]]]),
                           np.array([[np.exp(2 * th[2])]]), np.array([[np.exp(2 * th[3])]]))

    def f(th):
        A, R = unpack(th)
        return _mse_only(Gt, A, R, 1, 1, N)["mse"]

    per_start, best = {}, None
    for name, (F, H, Qx, V) in starts.items():
        th0 = np.array([_atanh_cap(F), H, 0.5 * np.log(Qx), 0.5 * np.log(V)])
        r = minimize(f, th0, method="BFGS", options={"gtol": 1e-10, "maxiter": 400})
        per_start[name] = {"start_penalty_pct": 100 * (f(th0) / base - 1),
                           "end_penalty_pct": 100 * (r.fun / base - 1), "nfev": int(r.nfev)}
        if best is None or r.fun < best.fun:
            best = r
    A, R = unpack(best.x)
    m = _mse_only(Gt, A, R, 1, 1, N)
    return {"mse": m["mse"], "nees": m["nees"],
            "F": float(EIG_CAP * np.tanh(best.x[0])), "H": float(best.x[1]),
            "Q": float(np.exp(2 * best.x[2])), "V": float(np.exp(2 * best.x[3])),
            "kl_rate": kl_rate(At, Rt, A, R), "per_start": per_start}


# ----------------------------------------------------------------------------
# Monte-Carlo validation with the awesomepkf library smoother
# ----------------------------------------------------------------------------
def mc_validate(At, Rt, p, q, N, models: dict, exact: dict, seeds: int) -> dict:
    try:
        from prg.classes.linear_pkf import Linear_PKF
        from prg.classes.linear_pks import Linear_PKS_VAR
        from prg.classes.param_linear import ParamLinear
        from prg.models.linear._amq import LinearAmQ
    except ImportError as e:  # pragma: no cover
        raise SystemExit("awesomepkf's prg package is not importable; set PYTHONPATH "
                         "or run with --no-mc") from e

    def param(A, R):
        d = p + q
        m = LinearAmQ(p, q, A=np.array(A, float), mQ=np.array(R, float),
                      mz0=np.zeros((d, 1)), Pz0=np.eye(d), pairwiseModel=True)
        pr = m.get_params().copy()
        pr.pop("dim_x")
        pr.pop("dim_y")
        return ParamLinear(0, p, q, **pr)

    p_true = param(At, Rt)
    params = {name: param(*AR) for name, AR in models.items()}
    mse = {k: [] for k in models}
    nees = {k: [] for k in models}
    cov_gap = {k: 0.0 for k in models}
    t0 = time.time()
    for s in range(seeds):
        data = Linear_PKF(p_true, sKey=s).simulate_N_data(N)
        X = np.array([np.asarray(x, float).reshape(p) for (_k, x, _y) in data])
        for name, prm in params.items():
            sm = Linear_PKS_VAR(prm)
            sm.process_N_data_smoother(N=len(data) - 1, data_generator=iter(data))
            xh = np.array([np.asarray(h["Xkp1_smooth"], float).reshape(p) for h in sm.history])
            P = np.array([np.asarray(h["PXXkp1_smooth"], float).reshape(p, p)
                          for h in sm.history])
            e = X - xh
            mse[name].append(float(np.mean(np.sum(e * e, axis=1))))
            nees[name].append(float(np.mean(np.einsum("ni,ni->n", e,
                                                      np.linalg.solve(P, e[:, :, None])[:, :, 0])) / p))
            if s == 0:
                cov_gap[name] = float(np.max(np.abs(P - exact[name]["P_blocks"])))
    out = {"seeds": seeds, "N": N, "smoother": "Linear_PKS_VAR",
           "runtime_s": time.time() - t0, "models": {}}
    for name in models:
        m, n_ = np.array(mse[name]), np.array(nees[name])
        se_m, se_n = m.std(ddof=1) / np.sqrt(seeds), n_.std(ddof=1) / np.sqrt(seeds)
        out["models"][name] = {
            "mc_mse": float(m.mean()), "mc_mse_se": float(se_m),
            "exact_mse": exact[name]["mse"],
            "z_mse": float((m.mean() - exact[name]["mse"]) / se_m),
            "mc_nees": float(n_.mean()), "mc_nees_se": float(se_n),
            "exact_nees": exact[name]["nees"],
            "z_nees": float((n_.mean() - exact[name]["nees"]) / se_n) if se_n > 0 else 0.0,
            "max_abs_gap_library_vs_exact_posterior_cov": cov_gap[name],
        }
    # paired penalty estimates (ratio of means, delta-method s.e.)
    base = np.array(mse["pairwise"])
    for name in models:
        if name == "pairwise":
            continue
        x = np.array(mse[name])
        r = x.mean() / base.mean()
        cov = np.cov(np.vstack([x, base]), ddof=1) / seeds
        var_r = (cov[0, 0] - 2 * r * cov[0, 1] + r * r * cov[1, 1]) / base.mean() ** 2
        out["models"][name]["mc_penalty_pct"] = float(100 * (r - 1))
        out["models"][name]["mc_penalty_se_pct"] = float(100 * np.sqrt(max(var_r, 0.0)))
        out["models"][name]["exact_penalty_pct"] = float(
            100 * (exact[name]["mse"] / exact["pairwise"]["mse"] - 1))
    return out


LIB_SMOOTHERS = ("Linear_PKS_RTS", "Linear_PKS_BF", "Linear_PKS_MBF", "Linear_PKS_MF",
                 "Linear_PKS_DWY", "Linear_PKS_VAR")


def mc_validate_unstable(seeds: int, N: int = N_STEPS) -> dict:
    """Library check on exact-match class-C members whose observation memory is UNSTABLE
    (|b| > 1): data from the (stable) truth, smoother built on the member (R calibrated).
    awesomepkf's ParamLinear refuses an unstable A at construction (a library-level guard; its
    public A setter does not check), so the member is built as follows: LinearAmQ (no stability
    check) builds the symbolic transition from the member's own A; ParamLinear's constructor gets
    the truth's A as a placeholder; the member's A is then set through the public setter, before
    any filter object is created (Linear_PKF reads param.A at construction). Reports, on seed 0,
    max |xhat_lib - K y| and |P_lib - P_exact| for the six library smoothers against the exact
    information-form smoother (K = -Jxx^-1 Jxy), and a Monte Carlo of Linear_PKS_VAR (`seeds`
    records) against the exact MSE / NEES."""
    try:
        import prg.classes.linear_pks as lib_pks
        from prg.classes.linear_pkf import Linear_PKF
        from prg.classes.param_linear import ParamLinear
        from prg.models.linear._amq import LinearAmQ
    except ImportError as e:  # pragma: no cover
        raise SystemExit("awesomepkf's prg package is not importable; set PYTHONPATH "
                         "or run with --no-mc") from e

    def param(A, R, A_placeholder=None):
        m = LinearAmQ(1, 1, A=np.array(A, float), mQ=np.array(R, float), mz0=np.zeros((2, 1)),
                      Pz0=np.eye(2), pairwiseModel=True)
        pr = m.get_params().copy()
        pr.pop("dim_x")
        pr.pop("dim_y")
        if A_placeholder is not None:
            pr["A"] = np.array(A_placeholder, float)
        prm = ParamLinear(0, 1, 1, **pr)
        prm.A = np.array(A, float)
        return prm

    def run(prm, data, name):
        sm = getattr(lib_pks, name)(prm)
        sm.process_N_data_smoother(N=len(data) - 1, data_generator=iter(data))
        xh = np.array([float(np.asarray(h["Xkp1_smooth"]).ravel()[0]) for h in sm.history])
        P = np.array([float(np.asarray(h["PXXkp1_smooth"]).ravel()[0]) for h in sm.history])
        return xh, P

    cases = {"rho1.5_a0.9": (A_true(1.5), R_TRUE, 0.9),
             "x1y1_Ryy_x4_a0.9": (*model_from_cfg(MODELS["x1y1"], 4.0)[:2], 0.9)}
    out = {"seeds": seeds, "N": N, "mc_smoother": "Linear_PKS_VAR", "cases": {}}
    t0 = time.time()
    for tag, (At, Rt, a) in cases.items():
        A, R, _, stable = classC_match_member(At, Rt, a)
        Rc = asym_metrics_any(At, Rt, A, R, 1)["nees"] * R
        exact = ExactTruth(At, Rt, 1, 1, N).eval(A, Rc)
        K, Pm = model_gain_precision(A, Rc, 1, 1, N)
        p_true, p_mem = param(At, Rt), param(A, Rc, A_placeholder=At)
        data0 = Linear_PKF(p_true, sKey=0).simulate_N_data(N)
        y0 = np.array([float(np.asarray(y).ravel()[0]) for (_k, _x, y) in data0])
        seed0 = {}
        for name in LIB_SMOOTHERS:
            try:
                xh, P = run(p_mem, data0, name)
                seed0[name] = {"max_abs_xhat_minus_Ky": float(np.max(np.abs(xh - K @ y0))),
                               "max_abs_P_minus_exact": float(np.max(np.abs(P - Pm[:, 0, 0])))}
            except Exception as e:      # 2F / DWY build stationary quantities: may fail
                seed0[name] = {"error": type(e).__name__}
        mse, nees, mse_pw = [], [], []
        for sd in range(seeds):
            data = data0 if sd == 0 else Linear_PKF(p_true, sKey=sd).simulate_N_data(N)
            X = np.array([float(np.asarray(x).ravel()[0]) for (_k, x, _y) in data])
            xh, P = run(p_mem, data, "Linear_PKS_VAR")
            xt, _ = run(p_true, data, "Linear_PKS_VAR")
            mse.append(float(np.mean((X - xh) ** 2)))
            nees.append(float(np.mean((X - xh) ** 2 / P)))
            mse_pw.append(float(np.mean((X - xt) ** 2)))
        mse, nees, mse_pw = np.array(mse), np.array(nees), np.array(mse_pw)
        se_m, se_n = mse.std(ddof=1) / np.sqrt(seeds), nees.std(ddof=1) / np.sqrt(seeds)
        r = mse.mean() / mse_pw.mean()
        cv = np.cov(np.vstack([mse, mse_pw]), ddof=1) / seeds
        var_r = (cv[0, 0] - 2 * r * cv[0, 1] + r * r * cv[1, 1]) / mse_pw.mean() ** 2
        out["cases"][tag] = {
            "a": a, "b": float(A[1, 1]), "stable": stable, "A": A.tolist(), "R_calibrated": Rc.tolist(),
            "exact": exact, "seed0_library_vs_exact": seed0,
            "mc_mse": float(mse.mean()), "mc_mse_se": float(se_m),
            "z_mse": float((mse.mean() - exact["mse"]) / se_m),
            "mc_nees": float(nees.mean()), "mc_nees_se": float(se_n),
            "z_nees": float((nees.mean() - exact["nees"]) / se_n),
            "mc_penalty_pct": float(100 * (r - 1)),
            "mc_penalty_se_pct": float(100 * np.sqrt(max(var_r, 0.0)))}
    out["runtime_s"] = time.time() - t0
    out["note"] = ("RTS, BF, MBF and VAR need no stable model and reproduce the exact smoother of "
                   "an unstable-memory member; 2F (Linear_PKS_MF) and DWY build stationary "
                   "quantities and fail or err, as the letter's Remark on choosing a smoother "
                   "states. The library's ParamLinear guard against unstable A is bypassed as "
                   "described in the docstring (no library file is modified).")
    return out


# ----------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------
STYLE = {  # Okabe-Ito, explicit on every call
    "pairwise": dict(color="#0072B2", marker="o", ls="-", label="pairwise (true model)"),
    "classical_projected": dict(color="#D55E00", marker="^", ls="-.",
                                label=r"classical $\mathbf{A}^{xy}{=}0$, KL fit"),
    "textbook_projected": dict(color="#009E73", marker="D", ls=":",
                               label=r"textbook $y_n{=}Hx_n{+}v_n$, KL fit"),
    "ablated": dict(color="#CC79A7", marker="s", ls="--",
                    label=r"ablated (true, $\mathbf{A}^{xy}{\to}0$)"),
}
ORDER = ["pairwise", "classical_projected", "textbook_projected", "ablated"]


def make_figure(res: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib as mpl
    import matplotlib.font_manager as fm
    avail = {f.name for f in fm.fontManager.ttflist}
    font = next((f for f in ("Arial", "Helvetica", "Liberation Sans") if f in avail),
                "DejaVu Sans")
    # set AFTER any prg import (prg's plot_settings would otherwise override)
    mpl.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "sans-serif", "font.sans-serif": [font],
        "mathtext.fontset": "stixsans",
        "font.size": 8, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "axes.titlesize": 9,
        "lines.linewidth": 1.2, "lines.markersize": 4,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "savefig.dpi": 300, "savefig.facecolor": "white", "figure.facecolor": "white",
        "figure.dpi": 100,      # fixed: prg's plot_settings sets 150, which moves tight_layout
        "axes.prop_cycle": mpl.cycler(color=["#000000"]),
        "legend.frameon": False,
    })
    import matplotlib.pyplot as plt

    rho = np.asarray(res["rho"])
    base = np.asarray(res["mse"]["pairwise"])
    pen = {k: 100.0 * (np.asarray(res["mse"][k]) / base - 1.0) for k in ORDER}
    nees = {k: np.asarray(res["nees"][k]) for k in ORDER}
    PEN_TOP, NEES_LIM = 20.0, (0.90, 1.40)      # ablated runs off both scales (annotated)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 4.3), sharex=True)
    handles = []
    for name in ORDER:
        st = STYLE[name]
        h, = ax1.plot(rho, pen[name], color=st["color"], marker=st["marker"],
                      ls=st["ls"], mfc=st["color"] if name == "pairwise" else "white",
                      mec=st["color"], label=st["label"], zorder=3 if name == "pairwise" else 2)
        handles.append(h)
    ax1.set_ylim(-1.5, PEN_TOP)
    ax1.set_ylabel("MSE penalty vs. pairwise (%)")
    ax1.grid(True, color="#DDDDDD", lw=0.5)
    ax1.set_title("(a)", loc="left", fontsize=9, color="#000000", pad=2)
    ab = STYLE["ablated"]["color"]
    ax1.text(0.99, 0.97, f"ablated off scale:\n+{pen['ablated'][-1]:.0f} % at " + r"$\rho=1$",
             transform=ax1.transAxes, ha="right", va="top", fontsize=8, color=ab)

    for name in ORDER:
        st = STYLE[name]
        ax2.plot(rho, nees[name], color=st["color"], marker=st["marker"],
                 ls=st["ls"], mfc=st["color"] if name == "pairwise" else "white",
                 mec=st["color"], zorder=3 if name == "pairwise" else 2)
    ax2.set_ylim(*NEES_LIM)
    ax2.text(0.99, 0.60, f"ablated off scale:\n{nees['ablated'][-1]:.2f} at " + r"$\rho=1$",
             transform=ax2.transAxes, ha="right", va="top", fontsize=8, color=ab)
    ax2.set_xlabel(r"back-action strength $\rho$  ($\mathbf{A}^{xy}=0.4\rho$)")
    ax2.set_ylabel("expected NEES")
    ax2.grid(True, color="#DDDDDD", lw=0.5)
    ax2.set_xlim(-0.02, 1.02)
    # calibrated level labelled directly (the pairwise curve lies exactly on NEES = 1)
    ax2.text(0.99, 0.03, "NEES = 1: calibrated (pairwise lies on it)", transform=ax2.transAxes,
             ha="right", va="bottom", fontsize=8, color="#555555")
    ax2.set_title("(b)", loc="left", fontsize=9, color="#000000", pad=2)
    fig.legend(handles=handles, loc="upper center", ncol=2, handlelength=2.4,
               columnspacing=0.8, handletextpad=0.4, borderaxespad=0.2)
    fig.tight_layout(h_pad=0.4, rect=(0, 0, 1, 0.915))
    FIG_OUT.parent.mkdir(exist_ok=True)
    fig.savefig(FIG_OUT)
    fig.savefig(FIG_OUT.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"figure: {FIG_OUT} (+ .png), font = {font}")


# ----------------------------------------------------------------------------
# Checks of the precision route, and text derived from the results
# ----------------------------------------------------------------------------
def model_gain_cov(Am, Rm, p, q, N):
    """Covariance route (stable models only), as in exact_metrics: K and the P_n blocks."""
    T = N + 1
    mxx, mxy, myy = split_xy(joint_cov(Am, Rm, np.eye(p + q), N), p, q, T)
    K = cho_solve(cho_factor(myy), mxy.T).T
    Pm = mxx.reshape(T, p, T, p)[np.arange(T), :, np.arange(T), :] - np.einsum(
        "nia,nja->nij", K.reshape(T, p, T * q), mxy.reshape(T, p, T * q))
    return K, sym3(Pm)


def precision_route_check(N: int = N_STEPS) -> dict:
    """For STABLE models both routes must agree to rounding: the finite-N gain and posterior
    covariances (precision vs covariance form) and the N->inf metrics (model side in precision
    vs spectral form), for the truth and the four comparators at rho = 0, 0.5, 1 and in every
    robustness row. Also the spectral radius of each KL projection."""
    cases = [(f"scalar_rho{r:g}", A_true(r), R_TRUE, 1, 1) for r in (0.0, 0.5, 1.0)]
    for tag, cfg in MODELS.items():
        for scale, lab in ((1.0, "nominal"), (4.0, "Ryy_x4")):
            At, Rt, pp, qq = model_from_cfg(cfg, scale)
            cases.append((f"{tag}_{lab}", At, Rt, pp, qq))
    out = {"max_abs_dK": 0.0, "max_abs_dP": 0.0, "max_abs_d_mse_inf": 0.0,
           "max_abs_d_nees_inf": 0.0, "max_spectral_radius_projections": 0.0, "cases": []}
    for tag, At, Rt, pp, qq in cases:
        comps, _ = comparators_main(At, Rt, pp, qq)
        models = {"pairwise": (At, Rt), **comps}
        rec = {"case": tag, "spectral_radius": {}}
        for name, (Am, Rm) in models.items():
            K1, P1 = model_gain_cov(Am, Rm, pp, qq, N)
            K2, P2 = model_gain_precision(Am, Rm, pp, qq, N)
            a1 = asym_metrics(At, Rt, Am, Rm, pp)
            a2 = asym_metrics_any(At, Rt, Am, Rm, pp)
            out["max_abs_dK"] = max(out["max_abs_dK"], float(np.max(np.abs(K1 - K2))))
            out["max_abs_dP"] = max(out["max_abs_dP"], float(np.max(np.abs(P1 - P2))))
            out["max_abs_d_mse_inf"] = max(out["max_abs_d_mse_inf"], abs(a1["mse"] - a2["mse"]))
            out["max_abs_d_nees_inf"] = max(out["max_abs_d_nees_inf"],
                                            abs(a1["nees"] - a2["nees"]))
            rec["spectral_radius"][name] = spectral_radius(Am)
            if name.endswith("_projected"):
                out["max_spectral_radius_projections"] = max(
                    out["max_spectral_radius_projections"], spectral_radius(Am))
        out["cases"].append(rec)
    return out


def _fmt_list(vals, fmt="{:.2f}"):
    return ", ".join(fmt.format(v) for v in vals)


def lib_txt(res: dict) -> str:
    mu = res["mc_validation_unstable_members"]["cases"]
    ok = [k[11:] for k in LIB_SMOOTHERS if all("error" not in c["seed0_library_vs_exact"][k] and
          c["seed0_library_vs_exact"][k]["max_abs_xhat_minus_Ky"] < 1e-10 for c in mu.values())]
    worst = max(c["seed0_library_vs_exact"][k]["max_abs_xhat_minus_Ky"] for c in mu.values()
                for k in LIB_SMOOTHERS if k[11:] in ok)
    bad = [{"MF": "2F (Linear_PKS_MF)"}.get(k[11:], k[11:]) for k in LIB_SMOOTHERS
           if k[11:] not in ok]
    return (" Library check (mc_validation_unstable_members, b = "
            + ", ".join(f"{c['b']:.2f}" for c in mu.values())
            + f"): the awesomepkf smoothers {', '.join(ok)} reproduce the exact smoother of these "
            f"unstable-memory members (max |xhat - K y| {worst:.0e}, seed 0, N=400) and a "
            f"{res['mc_validation_unstable_members']['seeds']}-seed MC agrees (|z| <= "
            f"{max(max(abs(c['z_mse']), abs(c['z_nees'])) for c in mu.values()):.1f}); "
            + (f"{', '.join(bad)} fail or err, as expected (they build stationary quantities). "
               if bad else "")
            + "The library's ParamLinear refuses an unstable A at construction (a guard, bypassed "
              "through its public setter for this check).")


def derive_text(res: dict) -> None:
    """Interpretation bullets and caption notes, with every number taken from this run."""
    d = res["design"]
    g, ga = res["rho_grid"], res["rho_grid"]["asymptotic"]
    if "capacity_best_in_class" not in res or "oracle_best_in_class" not in res:
        d["interpretation"] = ["capacity / oracle sections not computed in this run "
                               "(--no-capacity / --no-oracle): see a full run"]
        d["figure_caption_notes"] = ("Exact values for N=400 (401 states), prior N(0,I) on every "
                                     "model; (a) MSE penalty vs. pairwise; (b) expected NEES.")
        return
    cap, orc = res["capacity_best_in_class"], res["oracle_best_in_class"]
    per = {round(e["rho"], 6): e for e in cap["per_rho"]}
    e0, e05, e1 = per[0.0], per[0.5], per[1.0]
    f1 = e1["classC_family"]["best_calibrated"]
    pen400, peninf = g["penalty_pct"]["classical_projected"][-1], ga["penalty_pct"]["classical_projected"][-1]
    o1 = orc["classC_best"][-1]
    o1r = o1["diagnostics"]["R_rescaled_by_nees_inf"]
    r05 = e05["classC_family"]["kl_rate_ratio_range_over_sample"]
    r1 = e1["classC_family"]["kl_rate_ratio_range_over_sample"]
    ext = [per[round(r, 6)] for r in RHOS_EXT]
    rob = res["robustness_rho1"]["rows"]
    rows_txt = {f"{r['model']} {r['noise']}": r for r in rob}
    x4 = rows_txt["x1y1 Ryy_x4"]["capacity"]
    # textbook: share of the N->inf KL-fit penalty that the class itself cannot remove
    tb_cap = [e["textbook"]["best_value_pct"] for e in cap["per_rho"][:len(RHOS)]]
    tb_kl = ga["penalty_pct"]["textbook_projected"]
    tb_frac = [1.0 - c / k for c, k in zip(tb_cap, tb_kl)]
    d["textbook_fit_rule_share_inf"] = {
        "rho": g["rho"], "capacity_inf_pct": tb_cap, "kl_fit_inf_pct": tb_kl,
        "share_of_kl_fit_penalty_due_to_fitting_rule": tb_frac,
        "note": "1 - capacity/KL-fit penalty, both N->inf (same horizon)"}
    d["interpretation"] = [
        (f"classical_projected costs {pen400:.1f} % at rho=1 (N=400; {peninf:.1f} % as N->inf). "
         "This is the cost of excluding back-action WHEN the classical model is fitted by "
         "complete-data ML/KL (the same rule on the full GPMC class returns the truth and costs 0); "
         "it is not the cost of the class C={A^xy=0}: for this truth C contains, at every rho of the "
         "grid and of rho_extended, a one-parameter family of models whose N->inf smoother equals the "
         "pairwise one exactly and is calibrated after R -> kappa R "
         "(capacity_best_in_class.per_rho[*].classC_family). The best calibrated member found costs "
         f"{f1['N400']['penalty_pct']:.2f} % at N=400 and {f1['N1600']['penalty_pct']:.2f} % at "
         "N=1600 (rho=1): a boundary effect that decays like 1/N. The finite-N mean-only oracle "
         f"(stable members, NEES not controlled) reaches {o1['penalty_pct']:.2f} % (NEES "
         f"{o1['nees']:.2f}), {o1r['penalty_pct']:.2f} % once rescaled to NEES_inf = 1. So the class "
         f"itself loses nothing as N->inf and about {o1['penalty_pct']:.1f}-"
         f"{f1['N400']['penalty_pct']:.1f} % at N=400 (rho=1)."),
        ("Reachability: these members fit the transition much worse than the KL projection "
         f"(kl_rate {r05[0]:.1f}-{r05[1]:.1f}x the projection's at rho=0.5 and {r1[0]:.1f}-"
         f"{r1[1]:.1f}x at rho=1, over the sampled members) and, for rho > 0, none reproduces the true "
         "y law (the poles of their y process are {a, b}, not the eigenvalues of A_t), so neither a "
         "complete-data nor a y-only likelihood fit converges to them; reaching them needs a "
         "discriminative fit (smoothing error minimised with x observed in training)."),
        ("Stability (design.stability): the family's observation memory b = A^yy is stable only for "
         f"rho < {cap['classC_family_stable_for_rho_below']:.4f}; at rho = "
         f"{_fmt_list(RHOS_EXT, '{:g}')} it is b = {_fmt_list([e['classC_family']['b'] for e in ext], '{:.2f}')}, "
         f"and b = {x4['classC_family']['b']:.2f} for x1y1 with R^yy x4. Those members are exact "
         "(finite-N precision form; N->inf penalty 0). Restricted to STABLE models (spectral radius "
         "< 1), class C keeps an N->inf gap in these scalar cases ("
         f"{_fmt_list([e['C_stable']['best_value_pct'] for e in ext], '{:.3f}')} % at rho = "
         f"{_fmt_list(RHOS_EXT, '{:g}')}; {x4['C_stable']['best_value_pct']:.2f} % for x1y1 R^yy x4)"
         + "".join(f"; {k} {v['capacity']['C_stable']['best_value_pct']:.3f} % (vs "
                   f"{max(v['capacity']['C_free_Ayy']['best_value_pct'], 0.0):.3f} % with A^yy free)"
                   for k, v in rows_txt.items() if v["p"] > 1 and
                   v["capacity"]["C_stable"]["best_value_pct"]
                   - v["capacity"]["C_free_Ayy"]["best_value_pct"] > 5e-3)
         + "; local searches, upper bounds; where the cap binds (cap_active) the stable infimum is "
           "approached only at the stability boundary. Capacity statements about C refer to "
           "C_free_Ayy (the class as specified); C_stable is a reference."
         + (lib_txt(res) if "mc_validation_unstable_members" in res else "")),
        ("The textbook class (F stable, A^yy = 0; the stability cap "
         + ("binds in some cases" if any(e["textbook"]["cap_active"] for e in cap["per_rho"][:len(RHOS)])
            or any(v["capacity"]["textbook"]["cap_active"] for v in rob) else "never binds")
         + ") has a genuine N->inf gap: "
         f"{_fmt_list(tb_cap)} % for rho = {_fmt_list(g['rho'], '{:g}')}; it is 0 at rho = 0.5, where "
         "the truth's smoother numerator has no z^+1 term, so the exact-match family has b = 0 and "
         "lies in the textbook class. The share of its N->inf KL-fit penalty due to the fitting rule "
         f"(1 - capacity/KL fit) is {_fmt_list([100 * f for f in tb_frac], '{:.0f}')} % over the same "
         "rho (design.textbook_fit_rule_share_inf)."),
        ("Robustness rows (rho = 1): with A^yy unrestricted, class C reproduces the pairwise N->inf "
         "smoother in every row (best found: "
         + "; ".join(f"{k} {max(v['capacity']['C_free_Ayy']['best_value_pct'], 0.0):.3f} %"
                     for k, v in rows_txt.items())
         + "), against KL-fit penalties (N->inf) of "
         + "; ".join(f"{k} {v['asymptotic']['classical_projected']['penalty_pct']:.1f} %"
                     for k, v in rows_txt.items())
         + ". Class C = textbook + observation memory for the scalar truth only (y-conditional "
           "coefficient on x, A^yx - R^yx (R^xx)^-1 A^xx, is 0 there); it is nonzero for the 2-D "
           "models (robustness_rho1.rows[*].truth_y_conditional_coef_on_x)."),
        ("oracle_best_in_class (finite-N, mean-only, stable members) is a capacity reference, not "
         "an achievable fit: it needs the true x, at large rho it implies a y law far from the "
         f"truth (y variance x{o1['diagnostics']['y_var_ratio']:.1f} at rho=1), and its NEES is "
         "uncontrolled."),
    ]
    tbc0 = e0["textbook"]["calibrated_member"]["N400"]["penalty_pct"]
    d["figure_caption_notes"] = (
        "Exact values for N=400 (401 states), prior N(0,I) on every model. 'KL fit' = population "
        "complete-data ML (x observed in training) within the class. (a) MSE penalty relative to "
        "the pairwise smoother with the true parameters; (b) expected NEES, 1 = calibrated (the "
        "pairwise curve lies on it). The classical class C={A^xy=0} contains models whose N->inf "
        "smoother equals the pairwise one (best calibrated member found: "
        f"{f1['N400']['penalty_pct']:.2f} % at rho=1, N=400, decaying as 1/N), so the "
        "'classical, KL fit' curve is the cost of fitting C by complete-data KL with A^xy=0 imposed, "
        "not the cost of the class. At rho=0 the truth is in C but not in the textbook class "
        f"(observation memory A^yy=0.4): the best textbook member found costs {tbc0:.1f} % "
        f"(calibrated) against {g['penalty_pct']['textbook_projected'][0]:.1f} % for its KL fit. "
        "The ablated model (true parameters, A^xy set to 0) leaves both axes "
        f"(+{g['penalty_pct']['ablated'][-1]:.0f} %, NEES {g['nees']['ablated'][-1]:.2f} at rho=1).")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items() if k not in ("P_blocks", "E_blocks")}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def main() -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-mc", action="store_true")
    ap.add_argument("--no-oracle", action="store_true")
    ap.add_argument("--no-capacity", action="store_true")
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1),
                    help="processes for the multi-start capacity searches (results do not "
                         "depend on it)")
    ap.add_argument("--replot", action="store_true",
                    help="re-render the figure from the JSON only (no computation)")
    args = ap.parse_args()
    if args.replot:
        with open(JSON_OUT) as fh:
            saved = json.load(fh)
        make_figure({"rho": saved["rho_grid"]["rho"], "mse": saved["rho_grid"]["mse"],
                     "nees": saved["rho_grid"]["nees"]})
        return saved
    t_start = time.time()
    p, q, N = P_DIM, Q_DIM, N_STEPS
    res: dict = {
        "design": {
            "truth": "A(rho)=[[0.6,0.4*rho],[0.3,0.4]], R=[[0.1,0.05],[0.05,0.1]] fixed, "
                     "B=I, Z_0~N(0,I), N=400 (401 states), p=q=1",
            "rho": RHOS.tolist(),
            "models": {
                "pairwise": "true model",
                "classical_projected": "KL projection onto C={A^xy=0} (R incl. R^xy, A^yy free); "
                                       "iterated FGLS (SUR), checked vs closed form + multistart BFGS",
                "textbook_projected": "KL projection onto A=[[F,0],[HF,0]], R=[[Q,QH^T],[HQ,HQH^T+V]]; "
                                      "closed form, checked by multistart BFGS",
                "ablated": "true model with A^xy=0, other blocks kept (incl. R^xy)",
                "oldclass_projected": "REFERENCE: KL projection onto {A^xy=0,R^xy=0} (letter's old class)",
            },
            "prior_all_models": "N(0, I)",
            "metrics": "MSE = mean_n E_true||x_n - xhat_n||^2; NEES = mean_n tr(P_n^-1 E_true[e_n e_n^T])/p",
            "kl_rate": "E_stat KL(N(A_t z,R_t)||N(A_m z,R_m)) in nats per step",
            "horizon": ("All main values (rho_grid, rho_extended, robustness_rho1, decomposition, "
                        "old design) are exact for N=400 (401 states) with prior N(0,I) on EVERY "
                        "model, truth included; quote N and the prior with them. The 'asymptotic' "
                        "fields give the N->inf (stationary interior Wiener smoother, frequency "
                        "domain) values; they differ from N=400 by up to ~1.2 points of penalty."),
            "fit_rule": ("'projected' = KL projection of the true stationary transition, "
                         "E_stat[-log N(Z'; A_c Z, R_c)], i.e. the population limit of maximum "
                         "likelihood with x OBSERVED in training (complete-data, supervised fit) "
                         "within each class. It is not the limit of EM or ML on y alone (a y-only "
                         "fit does not identify x). The pairwise smoother likewise uses the "
                         "projection onto the full GPMC class, which is the truth."),
            "stability": (
                "The smoother of a model uses only its transition densities (y is observed, so "
                "the model is never simulated): x_hat = -Jxx^-1 Jxy y with the joint precision J, "
                "and N->inf H_m(w) = -Jxx(w)^-1 Jxy(w). Both are well defined for ANY A^yy as soon "
                "as the state dynamics A^xx is stable; no stationary law of the model is needed. "
                "Capacity is therefore reported for two regions: 'free_Ayy' (A^xx (resp. F) stable, "
                "A^yy unrestricted: the class C exactly as specified, A^xy=0 only) and 'stable' "
                "(spectral radius of A < 1, the usual stationary-model class). The KL projections "
                "are stable (checks.*spectral_radius*)."),
            "interpretation": ["filled at the end of the run from the results (derive_text)"],
            "figure_caption_notes": "filled at the end of the run from the results (derive_text)",
            "naming": ("models are referred to by name (pairwise, classical_projected, "
                       "textbook_projected, ablated, oldclass_projected); in the figure, (a) and "
                       "(b) label the PANELS (MSE penalty, NEES), not models"),
        },
        "env": {"python": sys.version.split()[0], "numpy": np.__version__,
                "scipy": scipy.__version__, "platform": platform.platform()},
    }
    names = ["pairwise", "classical_projected", "textbook_projected", "ablated",
             "oldclass_projected"]

    # ---- 1. rho grid ---------------------------------------------------------
    t0 = time.time()
    grid = {"rho": RHOS.tolist(), "mse": {k: [] for k in names},
            "nees": {k: [] for k in names}, "penalty_pct": {k: [] for k in names},
            "kl_rate": {k: [] for k in names}, "checks": [],
            "asymptotic": {"note": "N->inf (stationary interior smoother), frequency domain; "
                                   "mid_record_vs_asym_maxabs = max |E_n - E_inf|, |P_n - P_inf| "
                                   "at n = N/2 of the exact N=400 record (must be ~1e-16)",
                           "mse": {k: [] for k in names}, "nees": {k: [] for k in names},
                           "penalty_pct": {k: [] for k in names},
                           "mid_record_vs_asym_maxabs": {k: [] for k in names}}}
    for rho in RHOS:
        At = A_true(rho)
        comps, diag = comparators_main(At, R_TRUE, p, q)
        ev = evaluate(At, R_TRUE, p, q, N, comps, keep_blocks=True)
        asy = evaluate_asym(At, R_TRUE, p, comps, ev, N)
        for k in names:
            grid["mse"][k].append(ev[k]["mse"])
            grid["nees"][k].append(ev[k]["nees"])
            grid["penalty_pct"][k].append(ev[k]["penalty_pct"])
            grid["kl_rate"][k].append(ev[k]["kl_rate"])
            for key in ("mse", "nees", "penalty_pct", "mid_record_vs_asym_maxabs"):
                grid["asymptotic"][key][k].append(asy[k][key])
        fg = project_classC_fgls(At, R_TRUE, p, q)
        tb_A, tb_R, _ = project_textbook_closed(At, R_TRUE, p, q)
        diag["classC_multistart"] = check_classC_multistart(At, R_TRUE, p, q,
                                                            fg["objective_logdetC"])
        diag["classC_hessian_eigs"] = hessian_eigs_classC(At, R_TRUE, p, q, fg["A"])
        diag["textbook_multistart"] = check_textbook_multistart(At, R_TRUE, p, q, tb_A, tb_R)
        diag["pairwise_nees_minus_1"] = ev["pairwise"]["nees"] - 1.0
        diag["pairwise_mse_minus_mean_trP"] = ev["pairwise"]["mse"] - ev["pairwise"]["mean_tr_P_model"]
        diag["cond_Sig_yy"] = {k: ev[k]["cond_Sig_yy_model"] for k in names}
        diag["rho"] = float(rho)
        grid["checks"].append(diag)
    grid["runtime_s"] = time.time() - t0
    res["rho_grid"] = grid

    # ---- 2. extended rho (beyond 1, below the stability limit rho=2) ----------
    ext = {"rho": RHOS_EXT, "spectral_radius": [], "stationary_var_x": [],
           "mse": {k: [] for k in names}, "nees": {k: [] for k in names},
           "penalty_pct": {k: [] for k in names},
           "asymptotic": {"mse": {k: [] for k in names}, "nees": {k: [] for k in names},
                          "penalty_pct": {k: [] for k in names},
                          "mid_record_vs_asym_maxabs": {k: [] for k in names}}}
    for rho in RHOS_EXT:
        At = A_true(rho)
        comps, _ = comparators_main(At, R_TRUE, p, q)
        ev = evaluate(At, R_TRUE, p, q, N, comps, keep_blocks=True)
        asy = evaluate_asym(At, R_TRUE, p, comps, ev, N)
        ext["spectral_radius"].append(float(np.max(np.abs(np.linalg.eigvals(At)))))
        ext["stationary_var_x"].append(float(stationary(At, R_TRUE)[0][0, 0]))
        for k in names:
            ext["mse"][k].append(ev[k]["mse"])
            ext["nees"][k].append(ev[k]["nees"])
            ext["penalty_pct"][k].append(ev[k]["penalty_pct"])
            for key in ("mse", "nees", "penalty_pct", "mid_record_vs_asym_maxabs"):
                ext["asymptotic"][key][k].append(asy[k][key])
    ext["note"] = ("record starts from N(0,I), far below the stationary variance as rho->2; "
                   "the KL projection uses stationary moments, so the record is partly transient "
                   "(a large mid_record_vs_asym_maxabs shows that n=N/2 is not yet stationary)")
    res["rho_extended"] = ext

    # ---- 3. oracle MSE-best members (scalar grid, reference only) --------------
    if not args.no_oracle:
        t0 = time.time()
        orc = {"rho": RHOS.tolist(), "classC_best": [], "textbook_best": []}
        for i_rho, rho in enumerate(RHOS):
            At = A_true(rho)
            comps, diag = comparators_main(At, R_TRUE, p, q)
            starts = {k: comps[k] for k in ("classical_projected", "ablated",
                                            "oldclass_projected", "textbook_projected")}
            orc["classC_best"].append(oracle_classC_scalar(At, R_TRUE, N, starts))
            tb = {k: float(np.array(v)[0, 0]) for k, v in diag["textbook_params"].items()}
            tstarts = {"textbook_projected": (tb["F"], tb["H"], tb["Q"], tb["V"]),
                       "F=Axx,H=Rxy/Rxx": (AXX, 0.5, 0.10, 0.075)}
            orc["textbook_best"].append(oracle_textbook_scalar(At, R_TRUE, N, tstarts))
            print(f"  oracle rho={rho:.3f}: C-best "
                  f"{100 * (orc['classC_best'][-1]['mse'] / res['rho_grid']['mse']['pairwise'][i_rho] - 1):.2f}%",
                  flush=True)
        base = res["rho_grid"]["mse"]["pairwise"]
        for key in ("classC_best", "textbook_best"):
            for i, d in enumerate(orc[key]):
                d["penalty_pct"] = 100 * (d["mse"] / base[i] - 1)
        # reference diagnostics of the oracle members: N->inf metrics, implied y law,
        # and the same member with R rescaled so that NEES_inf = 1 (exact, N = 400)
        for i_rho, rho in enumerate(RHOS):
            At = A_true(rho)
            comps, _ = comparators_main(At, R_TRUE, p, q)
            tr400 = ExactTruth(At, R_TRUE, p, q, N)
            a0 = asym_metrics(At, R_TRUE, At, R_TRUE, p)["mse"]
            ytrue = y_autocov(At, R_TRUE, p)
            for key in ("classC_best", "textbook_best"):
                d = orc[key][i_rho]
                if key == "classC_best":
                    Am, Rm = np.array(d["A"]), np.array(d["R"])
                else:
                    Am, Rm = textbook_AR(*(np.array([[d[k]]]) for k in ("F", "H", "Q", "V")))
                am = asym_metrics(At, R_TRUE, Am, Rm, p)
                yac = y_autocov(Am, Rm, p)
                d["diagnostics"] = {
                    "penalty_inf_pct": 100.0 * (am["mse"] / a0 - 1.0), "nees_inf": am["nees"],
                    "y_autocov_lags0to3": [v[0][0] for v in yac],
                    "y_autocov_lags0to3_truth": [v[0][0] for v in ytrue],
                    "y_autocov_lags0to3_classical_projected": [
                        v[0][0] for v in y_autocov(*comps["classical_projected"], p)],
                    "y_var_ratio": yac[0][0][0] / ytrue[0][0][0],
                    "R_rescaled_by_nees_inf": {"kappa": am["nees"],
                                               **tr400.eval(Am, am["nees"] * Rm)}}
        orc["runtime_s"] = time.time() - t0
        orc["note"] = ("REFERENCE (capacity, not an achievable fit). Exact N=400 MSE minimized "
                       "over the class by BFGS from several starts (prior N(0,I) fixed, "
                       "eigenvalues of A capped at 0.9999 via tanh); non-convex, so these are "
                       "UPPER bounds on the best-in-class finite-N MSE (local search). Optimizes "
                       "the mean only: NEES not controlled. It needs the true x, and at large "
                       "rho its implied y law is far from the truth (see diagnostics: y "
                       "autocovariances, y variance ratio, and kl_rate), so no likelihood-based "
                       "fit reaches it. 'diagnostics' adds "
                       "its N->inf metrics and the same member with R rescaled so that "
                       "NEES_inf = 1. Searched over STABLE members only (|eigenvalues| < 0.9999: "
                       "a and A^yy both capped), so for the class C as specified (A^yy free) it is "
                       "still an upper bound. The class-C infimum itself tends to 0 as N->inf: see "
                       "capacity_best_in_class.")
        res["oracle_best_in_class"] = orc

    # ---- 3b. best-in-class capacity, N -> inf (+ calibrated members at finite N) ----
    pool = None if args.no_capacity else ProcessPoolExecutor(max_workers=args.workers)
    if not args.no_capacity:
        t0 = time.time()
        cap = {"rho": RHOS.tolist() + list(RHOS_EXT), "per_rho": [],
               "note": (
                   "REFERENCE. What each class can do whatever the fitting rule (N->inf). "
                   "classC_family: closed-form one-parameter family of C members whose N->inf "
                   "smoother transfer function equals the pairwise one (p=q=1); it exists at every "
                   "rho here; its observation memory b=A^yy is stable (|b|<1) only for rho < "
                   "classC_family_stable_for_rho_below, b < -1 beyond (smoother in precision form). "
                   "textbook (F stable), C_stable / old_stable (spectral radius of A < 1) and "
                   "C_free_Ayy / old_free_Ayy (A^xx stable, A^yy unrestricted; C_free_Ayy is the "
                   "class C as specified): N->inf best member by trust-region least squares on the "
                   "frequency-domain MSE (model side in precision form), multi-start in parallel "
                   "(given starts, perturbations, random members; C warm-started from the textbook "
                   "and old optima, which lie in C). Local search, so UPPER bounds on each class "
                   f"infimum ('best_value_pct'). The constrained radius has a soft cap {SR_CAP_OPT} "
                   f"(exterior penalty, hard wall {SR_HARD}); optima on the cap are continued to "
                   f"{SR_CAP_CONT} ('continuation'); 'cap_active' = the continuation lowers the "
                   "value (> 1e-4 pp), i.e. the infimum over the region is approached only at the "
                   "boundary (an optimum can also sit on the cap of a flat valley, cap_active "
                   "false). C_free_Ayy.best_value_pct includes the C_stable search (a subset). "
                   "Calibrated members: R rescaled so that NEES_inf = 1, then "
                   "evaluated EXACTLY (precision form) at N=400 and N=1600 with prior N(0,I); these "
                   "finite-N values are for one member, not finite-N optima.")}
        for rho in list(RHOS) + list(RHOS_EXT):
            At = A_true(rho)
            comps, _ = comparators_main(At, R_TRUE, p, q)
            truths_N = {N: ExactTruth(At, R_TRUE, p, q, N), 4 * N: ExactTruth(At, R_TRUE, p, q, 4 * N)}
            entry = {"rho": float(rho), "classC_family": classC_match_family(At, R_TRUE, truths_N)}
            entry.update(capacity_all(At, R_TRUE, p, q, comps, truths_N, pool, with_old=True))
            entry["classical_projected_N1600"] = truths_N[4 * N].eval(*comps["classical_projected"])
            entry["textbook_projected_N1600"] = truths_N[4 * N].eval(*comps["textbook_projected"])
            cap["per_rho"].append(entry)
            fam = entry["classC_family"]
            print(f"  capacity rho={rho:.3f}: C_free {max(entry['C_free_Ayy']['best_value_pct'], 0):.4f}% "
                  f"C_stable {max(entry['C_stable']['best_value_pct'], 0):.4f}% "
                  f"textbook {entry['textbook']['best_value_pct']:.4f}% "
                  f"old stable/free {entry['old_stable']['best_value_pct']:.3f}/"
                  f"{entry['old_free_Ayy']['best_value_pct']:.3f}% | family "
                  + (f"b={fam['b']:.3f} best calibrated N=400 "
                     f"{fam['best_calibrated']['N400']['penalty_pct']:.3f}%"
                     if fam["exists"] else "none") + f" [{time.time() - t0:.0f}s]", flush=True)
        # largest rho for which the family's observation memory is stable (|b(rho)| < 1)
        def b_of(r):
            (_, t0_, t1_), _ = scalar_laurent(A_true(r), R_TRUE)
            return abs(t1_ / t0_) - 1.0
        cap["classC_family_stable_for_rho_below"] = float(brentq(b_of, 1.0, 1.9, xtol=1e-12))
        cap["classC_family_b_vs_rho"] = {
            "rho": [float(r) for r in np.linspace(0.0, 1.9, 20)],
            "b": [float(-scalar_laurent(A_true(r), R_TRUE)[0][2]
                        / scalar_laurent(A_true(r), R_TRUE)[0][1]) for r in np.linspace(0.0, 1.9, 20)]}
        cap["runtime_s"] = time.time() - t0
        res["capacity_best_in_class"] = cap

    # ---- 4. old design + rho=1 decomposition (reference only) -----------------
    old = {"rho": RHOS.tolist(),
           "note": "old design: rho scales A^xy=0.4rho AND R^xy=0.05rho",
           "penalty_pct": {k: [] for k in names + ["ablated_old"]},
           "nees": {k: [] for k in names + ["ablated_old"]}}
    for rho in RHOS:
        At = A_true(rho)
        Rt = np.array([[0.10, 0.05 * rho], [0.05 * rho, 0.10]])
        comps, _ = comparators_main(At, Rt, p, q)
        comps["ablated_old"] = (A_true(0.0), np.diag([0.10, 0.10]))
        ev = evaluate(At, Rt, p, q, N, comps)
        for k in names + ["ablated_old"]:
            old["penalty_pct"][k].append(ev[k]["penalty_pct"])
            old["nees"][k].append(ev[k]["nees"])
    res["old_design_reference"] = old

    dec = {}
    truths = {"backaction_and_corrnoise": (A_true(1.0), R_TRUE),
              "backaction_only_Rxy0": (A_true(1.0), np.diag([0.10, 0.10])),
              "corrnoise_only_Axy0": (A_true(0.0), R_TRUE)}
    for tag, (At, Rt) in truths.items():
        comps, _ = comparators_main(At, Rt, p, q)
        comps["textbook_memory_projected"] = project_textbook_memory(At, Rt, p, q)
        ev = evaluate(At, Rt, p, q, N, comps)
        dec[tag] = {k: {"penalty_pct": v["penalty_pct"], "nees": v["nees"],
                        "kl_rate": v["kl_rate"]} for k, v in ev.items()}
    res["decomposition_rho1_reference"] = dec

    # ---- 5. robustness table (rho = 1) ----------------------------------------
    t0 = time.time()
    rob = {"note": "A^xy at full value, R^xy (noise cross block S) unchanged; "
                   "'x4' multiplies R^yy by 4 only (R^xy unchanged, so the noise "
                   "correlation coefficient halves)", "rows": []}
    for tag, cfg in MODELS.items():
        for scale, lab in ((1.0, "nominal"), (4.0, "Ryy_x4")):
            At, Rt, pp, qq = model_from_cfg(cfg, scale)
            comps, diag = comparators_main(At, Rt, pp, qq)
            fg = project_classC_fgls(At, Rt, pp, qq)
            tb_A, tb_R, _ = project_textbook_closed(At, Rt, pp, qq)
            ev = evaluate(At, Rt, pp, qq, N, comps, keep_blocks=True)
            asy = evaluate_asym(At, Rt, pp, comps, ev, N)
            Cyx = Rt[pp:, :pp] @ np.linalg.inv(Rt[:pp, :pp])
            row = {"model": tag, "p": pp, "q": qq, "noise": lab,
                   "spectral_radius": float(np.max(np.abs(np.linalg.eigvals(At)))),
                   "metrics": {k: {"mse": v["mse"], "nees": v["nees"],
                                   "penalty_pct": v["penalty_pct"], "kl_rate": v["kl_rate"]}
                               for k, v in ev.items()},
                   "asymptotic": asy,
                   "truth_y_conditional_coef_on_x": (At[pp:, :pp] - Cyx @ At[:pp, :pp]).tolist(),
                   "checks": {
                       "classC_fgls_iters": diag["classC_fgls_iters"],
                       "classC_fgls_grad_free_max": diag["classC_fgls_grad_free_max"],
                       "classC_fgls_vs_closed_maxabs_A": diag["classC_fgls_vs_closed_maxabs_A"],
                       "classC_fgls_vs_closed_maxabs_R": diag["classC_fgls_vs_closed_maxabs_R"],
                       "classC_multistart": check_classC_multistart(
                           At, Rt, pp, qq, fg["objective_logdetC"], n_starts=10),
                       "classC_hessian_min_eig": min(hessian_eigs_classC(At, Rt, pp, qq, fg["A"])),
                       "textbook_multistart": check_textbook_multistart(
                           At, Rt, pp, qq, tb_A, tb_R, n_starts=10),
                       "pairwise_nees_minus_1": ev["pairwise"]["nees"] - 1.0}}
            if not args.no_capacity:
                truths_N = {N: ExactTruth(At, Rt, pp, qq, N),
                            4 * N: ExactTruth(At, Rt, pp, qq, 4 * N)}
                row["capacity"] = capacity_all(At, Rt, pp, qq, comps, truths_N, pool, with_old=False)
                row["capacity"]["classical_projected_N1600"] = truths_N[4 * N].eval(
                    *comps["classical_projected"])
                row["capacity"]["textbook_projected_N1600"] = truths_N[4 * N].eval(
                    *comps["textbook_projected"])
                if pp == 1 and qq == 1:
                    row["capacity"]["classC_family"] = classC_match_family(At, Rt, truths_N)
                del truths_N
                rc = row["capacity"]
                print(f"  robustness {tag} {lab}: C_free {max(rc['C_free_Ayy']['best_value_pct'], 0):.4f}% "
                      f"C_stable {rc['C_stable']['best_value_pct']:.4f}% "
                      f"textbook {rc['textbook']['best_value_pct']:.4f}% "
                      f"[{time.time() - t0:.0f}s]", flush=True)
            rob["rows"].append(row)
    rob["runtime_s"] = time.time() - t0
    if pool is not None:
        pool.shutdown()
    rob["note_y_conditional"] = ("truth_y_conditional_coef_on_x = A^yx - R^yx (R^xx)^-1 A^xx: "
                                 "class C keeps the truth's y'|(x',x,y) exactly and replaces only "
                                 "the x dynamics by an AR(1) fit of x alone; when this coefficient "
                                 "is 0 (scalar truth) C-projection = textbook+memory projection; "
                                 "it is nonzero for x2y2 and x2y1")
    res["robustness_rho1"] = rob

    # ---- 6. MC validation at rho = 1 ------------------------------------------
    if not args.no_mc:
        At = A_true(1.0)
        comps, _ = comparators_main(At, R_TRUE, p, q)
        models = {"pairwise": (At, R_TRUE)}
        models.update({k: comps[k] for k in ("classical_projected", "textbook_projected",
                                             "ablated")})
        ev = evaluate(At, R_TRUE, p, q, N, {k: v for k, v in models.items() if k != "pairwise"},
                      keep_blocks=True)
        res["mc_validation_rho1"] = mc_validate(At, R_TRUE, p, q, N, models, ev, args.seeds)
        res["mc_validation_unstable_members"] = mc_validate_unstable(args.seeds, N)

    res["precision_route_check"] = precision_route_check()
    derive_text(res)
    res["runtime_total_s"] = time.time() - t_start
    with open(JSON_OUT, "w") as fh:
        json.dump(_clean(res), fh, indent=1)
    print_summary(res)
    make_figure({"rho": RHOS.tolist(), "mse": res["rho_grid"]["mse"],
                 "nees": res["rho_grid"]["nees"]})
    print(f"results: {JSON_OUT}")
    print(f"total runtime: {res['runtime_total_s']:.1f} s")
    return res


def print_summary(res: dict) -> None:
    g = res["rho_grid"]
    short = {"pairwise": "pair", "classical_projected": "C-proj",
             "textbook_projected": "textbk", "ablated": "ablat", "oldclass_projected": "old*"}
    ks = list(short)
    print("\n=== Exact results, scalar GPMC, N=400, prior N(0,I) (old* = reference only) ===")
    print("rho  | MSE: " + " ".join(f"{short[k]:>8}" for k in ks)
          + " | penalty %: " + " ".join(f"{short[k]:>7}" for k in ks[1:])
          + " | NEES: " + " ".join(f"{short[k]:>6}" for k in ks))
    for i, rho in enumerate(g["rho"]):
        print(f"{rho:4.3f}| "
              + " ".join(f"{g['mse'][k][i]:8.5f}" for k in ks) + " |            "
              + " ".join(f"{g['penalty_pct'][k][i]:7.2f}" for k in ks[1:]) + " |       "
              + " ".join(f"{g['nees'][k][i]:6.3f}" for k in ks))
    ga = g["asymptotic"]
    print("\nSame grid, N -> inf (frequency domain): rho | MSE pair | penalty %: "
          + " ".join(f"{short[k]:>7}" for k in ks[1:]) + " | NEES: "
          + " ".join(f"{short[k]:>6}" for k in ks[1:]) + " | max mid-record check")
    for i, rho in enumerate(g["rho"]):
        print(f"  {rho:5.3f} | {ga['mse']['pairwise'][i]:.5f} |            "
              + " ".join(f"{ga['penalty_pct'][k][i]:7.2f}" for k in ks[1:]) + " |       "
              + " ".join(f"{ga['nees'][k][i]:6.3f}" for k in ks[1:])
              + f" | {max(ga['mid_record_vs_asym_maxabs'][k][i] for k in ks):.1e}")
    if "oracle_best_in_class" in res:
        o = res["oracle_best_in_class"]
        print("\nOracle finite-N MSE-best member (reference; mean only, needs true x): rho | "
              "C-best pen% NEES [N->inf pen% NEES] y-var ratio | R rescaled: pen% NEES | "
              "textbook-best pen% NEES [N->inf pen%]")
        for i, rho in enumerate(o["rho"]):
            a, b = o["classC_best"][i], o["textbook_best"][i]
            da, db = a.get("diagnostics", {}), b.get("diagnostics", {})
            extra_a = (f" [{da['penalty_inf_pct']:6.3f} {da['nees_inf']:5.3f}] "
                       f"{da['y_var_ratio']:6.2f} | {da['R_rescaled_by_nees_inf']['penalty_pct']:6.2f} "
                       f"{da['R_rescaled_by_nees_inf']['nees']:5.3f}") if da else ""
            extra_b = f" [{db['penalty_inf_pct']:6.3f}]" if db else ""
            print(f"  {rho:5.3f} | {a['penalty_pct']:7.2f} {a['nees']:6.3f}{extra_a} | "
                  f"{b['penalty_pct']:7.2f} {b['nees']:6.3f}{extra_b}")
    if "capacity_best_in_class" in res:
        c = res["capacity_best_in_class"]
        print("\nBest-in-class CAPACITY (reference). N->inf best penalty % (local search, upper "
              "bounds; * = the stability cap binds, value after continuation):\n"
              "  C_free = {A^xy=0}, A^xx stable, A^yy free | C_stable = spectral radius < 1 | "
              "textbook (F stable) | old stable / free;\n"
              "  C exact-match family (closed form): b = A^yy, max|H_m-H_t|, best calibrated member "
              "(R->kappa R, NEES_inf=1): a, exact penalty%/NEES at N=400 and N=1600;\n"
              "  KL fits at N=1600 for comparison")

        def star(v):
            return f"{max(v['best_value_pct'], 0.0):7.4f}{'*' if v['cap_active'] else ' '}"
        for e_ in c["per_rho"]:
            fam = e_["classC_family"]
            if fam["exists"]:
                bc = fam["best_calibrated"]
                ftxt = (f"b={fam['b']:6.3f} dH {fam['max_abs_H_m_minus_H_t_over_family']:.0e} "
                        f"a*={bc['a']:.3f} N400 {bc['N400']['penalty_pct']:5.2f}%/"
                        f"{bc['N400']['nees']:.3f} N1600 {bc['N1600']['penalty_pct']:5.2f}%/"
                        f"{bc['N1600']['nees']:.3f} (sample N400 "
                        f"{fam['range_over_sample_penalty_pct']['N400'][0]:.2f}-"
                        f"{fam['range_over_sample_penalty_pct']['N400'][1]:.2f}%)")
            else:
                ftxt = "no family"
            print(f"  rho {e_['rho']:5.3f}: C_free {star(e_['C_free_Ayy'])} C_stable "
                  f"{star(e_['C_stable'])} tb {star(e_['textbook'])} old {star(e_['old_stable'])}/"
                  f"{star(e_['old_free_Ayy'])} | {ftxt} | KL N1600 C "
                  f"{e_['classical_projected_N1600']['penalty_pct']:5.2f}% tb "
                  f"{e_['textbook_projected_N1600']['penalty_pct']:5.2f}%")
        print(f"  family observation memory stable (|b| < 1) for rho < "
              f"{c['classC_family_stable_for_rho_below']:.4f}")
    e = res["rho_extended"]
    print("\nExtended rho (JSON): rho | sr | penalty% C-proj textbk ablat old* | N->inf: same")
    for i, rho in enumerate(e["rho"]):
        print(f"  {rho:4.2f} | {e['spectral_radius'][i]:.4f} | "
              + " ".join(f"{e['penalty_pct'][k][i]:8.2f}" for k in ks[1:]) + " | "
              + " ".join(f"{e['asymptotic']['penalty_pct'][k][i]:8.2f}" for k in ks[1:]))
    r = res["robustness_rho1"]
    print("\n=== Robustness, rho=1 (exact, N=400, prior N(0,I)). Ryy_x4: R^yy x4, R^xy unchanged ===")
    print(f"{'model':>5} {'noise':>8} | dMSE%: {'C-proj':>7} {'textbk':>7} {'ablat':>7} "
          f"{'old*':>7} | NEES: {'pair':>6} {'C-proj':>6} {'textbk':>6} {'ablat':>6} {'old*':>6}")
    for row in r["rows"]:
        m = row["metrics"]
        print(f"{row['model']:>5} {row['noise']:>8} |       "
              + " ".join(f"{m[k]['penalty_pct']:7.2f}" for k in ks[1:]) + " |       "
              + " ".join(f"{m[k]['nees']:6.3f}" for k in ks))
    print("  N->inf: model noise | dMSE%: C-proj textbk ablat old* | NEES C-proj textbk ablat "
          "old* | mid-record check | capacity N->inf C_free, C_stable, textbook (* on cap) | "
          "best calibrated C member found N400 / N1600 pen% | scalar: exact family member")
    for row in r["rows"]:
        a = row["asymptotic"]
        txt = (f"  {row['model']:>5} {row['noise']:>8} | "
               + " ".join(f"{a[k]['penalty_pct']:7.2f}" for k in ks[1:]) + " | "
               + " ".join(f"{a[k]['nees']:6.3f}" for k in ks[1:])
               + f" | {max(a[k]['mid_record_vs_asym_maxabs'] for k in ks):.1e}")
        if "capacity" in row:
            rc = row["capacity"]
            cf, cs, ct = rc["C_free_Ayy"], rc["C_stable"], rc["textbook"]
            cm = cf["calibrated_member_best_of_C"]

            def st(v):
                return f"{max(v['best_value_pct'], 0.0):.4f}" + ("*" if v["cap_active"] else "")
            txt += (f" | {st(cf)} {st(cs)} {st(ct)} | "
                    f"{cm['N400']['penalty_pct']:.2f} / {cm['N1600']['penalty_pct']:.2f}")
            if "classC_family" in rc and rc["classC_family"]["exists"]:
                bc = rc["classC_family"]["best_calibrated"]
                txt += (f" | family b={rc['classC_family']['b']:.2f}: "
                        f"{bc['N400']['penalty_pct']:.2f} / {bc['N1600']['penalty_pct']:.2f}")
        print(txt)
    d = res["decomposition_rho1_reference"]
    print("\nDecomposition at rho=1 (reference): truth -> penalty% [old class | C | textbook | "
          "textbook+memory | ablated]")
    for tag, v in d.items():
        print(f"  {tag:>26}: " + " | ".join(
            f"{v[k]['penalty_pct']:6.2f}" for k in ("oldclass_projected", "classical_projected",
                                                    "textbook_projected",
                                                    "textbook_memory_projected", "ablated")))
    od = res["old_design_reference"]
    print("\nOld design (rho scales A^xy and R^xy), penalty% at rho=1: "
          f"old class {od['penalty_pct']['oldclass_projected'][-1]:.2f}, "
          f"old ablated {od['penalty_pct']['ablated_old'][-1]:.2f}, "
          f"class C {od['penalty_pct']['classical_projected'][-1]:.2f}, "
          f"textbook {od['penalty_pct']['textbook_projected'][-1]:.2f}")
    if "precision_route_check" in res:
        pc = res["precision_route_check"]
        print(f"\nPrecision vs covariance route (stable models): max|dK| {pc['max_abs_dK']:.1e}, "
              f"max|dP| {pc['max_abs_dP']:.1e}; N->inf max|dMSE| {pc['max_abs_d_mse_inf']:.1e}, "
              f"max|dNEES| {pc['max_abs_d_nees_inf']:.1e}; max spectral radius of the KL "
              f"projections {pc['max_spectral_radius_projections']:.3f}")
    if "mc_validation_rho1" in res:
        mc = res["mc_validation_rho1"]
        print(f"\n=== MC validation at rho=1 ({mc['seeds']} seeds, N={mc['N']}, "
              f"{mc['smoother']}, {mc['runtime_s']:.0f} s) ===")
        print(f"{'model':>20} | {'MC MSE':>9} {'s.e.':>8} {'exact':>9} {'z':>6} | "
              f"{'MC NEES':>8} {'s.e.':>7} {'exact':>7} {'z':>6} | max|P_lib-P_exact|")
        for k, v in mc["models"].items():
            print(f"{k:>20} | {v['mc_mse']:9.5f} {v['mc_mse_se']:8.5f} {v['exact_mse']:9.5f} "
                  f"{v['z_mse']:6.2f} | {v['mc_nees']:8.4f} {v['mc_nees_se']:7.4f} "
                  f"{v['exact_nees']:7.4f} {v['z_nees']:6.2f} | "
                  f"{v['max_abs_gap_library_vs_exact_posterior_cov']:.1e}")
        if "mc_validation_unstable_members" in res:
            mu = res["mc_validation_unstable_members"]
            print(f"  unstable-memory class-C members (library, {mu['seeds']} seeds, "
                  f"{mu['runtime_s']:.0f} s):")
            for tag, c in mu["cases"].items():
                s0 = "; ".join(f"{k[11:]} " + (f"{v['max_abs_xhat_minus_Ky']:.0e}" if "error" not in v
                                               else v["error"])
                               for k, v in c["seed0_library_vs_exact"].items())
                print(f"    {tag} (b={c['b']:.3f}): VAR MC MSE {c['mc_mse']:.5f} exact "
                      f"{c['exact']['mse']:.5f} z={c['z_mse']:+.2f}; NEES {c['mc_nees']:.4f} exact "
                      f"{c['exact']['nees']:.4f} z={c['z_nees']:+.2f}; paired penalty "
                      f"{c['mc_penalty_pct']:.2f}+/-{c['mc_penalty_se_pct']:.2f} % (exact "
                      f"{c['exact']['penalty_pct']:.2f} %) | seed 0 max|xhat-Ky|: {s0}")
        for k, v in mc["models"].items():
            if "mc_penalty_pct" in v:
                print(f"  paired MC penalty {k}: {v['mc_penalty_pct']:.2f} +/- "
                      f"{v['mc_penalty_se_pct']:.2f} % (exact {v['exact_penalty_pct']:.2f} %)")


if __name__ == "__main__":
    main()
