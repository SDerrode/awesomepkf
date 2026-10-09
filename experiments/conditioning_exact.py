#!/usr/bin/env python
"""Accuracy of the six awesomepkf pairwise smoothers against an EXACT reference.

New Sec. III experiment of the SPL letter "Six Smoothers for Gaussian Pairwise Markov
Chains, and How to Choose One".

Model (GPMC): Z_n = (X_n, Y_n) in R^{p+q}, Z_{n+1} = A Z_n + W_{n+1}, W ~ N(0, R)
(B = I, so R = mQ), Y_n = y_n observed, prior Z_0 ~ N(0, Pz0). Target p(x_n | y_{0:N}).

Two sets of models are run through the same four stress regimes (D, S, C, M), at
(p, q) = (2, 1) and (2, 2), N = 400:
  * the BASE model (one fixed (A0, R0) per (p, q)), 10 seeds per sweep value, with the full
    diagnostics (condition numbers, cancellation factors, float-vs-exact intermediates);
  * two RANDOM FAMILIES of models, 20 models per (generator, q), 3 seeds per model, with a
    compact record per case:
      G1 (random_model_g1): Gaussian A rescaled to rho ~ U[0.6, 0.9], back-action block
         rescaled to ||A^xy||_2 = 0.3, redrawn until rho(A) < 0.95 and the relative
         Henrici departure > 0.3; random correlation matrices Cxx, Cyy (normalised
         Wishart) and a random direction K (||K||_2 = 1); unit noise variances. (Its
         first 10 models per q are the "random C" models of the previous version.)
      G2 (random_model_g2): Schur form A = U T U^T, U Haar-orthogonal, T upper triangular
         with real eigenvalues ~ U[-0.85, 0.85] and N(0, s^2) strictly-upper entries,
         s ~ U[0.3, 1]; redrawn until ||A^xy||_2 in [0.2, 0.4] and rho(A) >= 0.5; noise
         correlation matrices with a random spectrum (eigenvalues ~ U[0.2, 1.8], unit
         diagonal), K = U_x diag(1, s_2) U_y^T (s_2 ~ U[0, 1]), and random noise standard
         deviations ~ logU[0.5, 2].
For every case, ONE float64 record (x, y)_{0:N} per seed is simulated and the SAME y record
is smoothed by
  * the six LIBRARY smoothers (awesomepkf, prg.classes.linear_pks): RTS, BF, MBF,
    2F (= Linear_PKS_MF), DWY, VAR. It requires awesomepkf >= 2.16.0 (the version
    actually imported is recorded in meta.library), in which the shared 2F/DWY backward
    filter re-symmetrises Sigma_n, Q^b_n, P^b_{n-1|n} and every conditioned covariance at
    each step, the InvertibleMatrix determinant test is invariant to scaling and to
    dimension, and in-place covariance regularisations are listed in
    PKF.covariance_regularisations;
  * an exact reference: the lifted system J x = eta in the latent trajectory, solved in
    mpmath at 60 significant digits from the very same float64 model matrices
    (block-Thomas sweep for the mean, Takahashi recursion for the diagonal and lag-one
    blocks of J^{-1}).

Consistency check (not a smoother of the study; kept out of the tables and figures): 2F and DWY
are also run with the symmetrised backward filter the previous version of this script used as
a control ("2F*", "DWY*": the 2.15.1 backward filter with P^b_{n-1|n} and P^b_n symmetrised
once per step, Sigma_n, Q^b_n and P^y_n left as computed), swapped in by an in-process
monkeypatch. Per case the script records the control's errors and the direct difference
between the library output and the control output (consistency_control); a line-by-line
replicate of the library's 2.16.0 backward filter is checked bitwise against the library on
a few cases (meta.replicate_selfcheck).

The library regularises in place any covariance that fails its CovarianceMatrix check
(min eigenvalue <= 0 or cond >= 1e12) inside PKF._check_covariance. This script wraps that
method (in this process only) to record each smoothed covariance BEFORE the check and to
log every in-place regularisation (forward filter and smoothing passes alike); the count is
cross-checked against the library's own PKF.covariance_regularisations list. Errors are
reported both for the raw algorithm output ("raw") and for what the library hands back
("returned").

Library aborts: every exception is recorded with the library check that raised it. The
script wraps (in-process) PKF._check_invertible (to record which InvertibleMatrix test
failed: normalised determinant, cond >= 1e12, rank, residual) and the module-level
cho_factor used by prg.classes.linear_pks and prg.classes.pkf (to record the float64 matrix
whose Cholesky factorisation failed). Each abort gets a cause, a "tolerance artefact" flag
(True when the abort comes from a fixed-threshold test on a float64 matrix that is
nevertheless Cholesky-factorisable) and, when the failing matrix has an exact counterpart,
its exact smallest eigenvalue and condition number and the relative error of the float
matrix.

Metrics (per smoother, per seed):
  mean_err  = max_n ||xhat_n - xref_n||_2 / max_n ||xref_n||_2
  cov_err   = max_n ||P_n - Pref_n||_2 / ||Pref_n||_2          (raw and returned)
  lag1_err  = max_n ||C_n - Cref_n||_2 / sqrt(||Pref_n|| ||Pref_{n+1}||), C_n = Cov(x_{n+1}, x_n | y)
              (RTS from Gk_smooth, VAR from Mk_smooth; the others do not return it)
  *_mid     = the same maxima restricted to the interior window n in [100, 300]
  indefinite: min eigenvalue of the raw smoothed covariance < 0 (first n, number of n)
All differences are formed against the double-double split (hi, lo) of the 60-digit
reference, (est - hi) - lo, so roundoff-level errors are resolved. Each error above 1e-13
is localised: "persistent" if its interior-window value is >= 1/10 of its maximum or >= 1e-8
(absolute), else "start-up" (argmax n < 100) or "end" (argmax n > 300); errors <= 1e-13 are
"small".
Conditioning (seed-independent: the covariance recursions do not see the data), max and
median over n of the condition number of what each smoother inverts:
  RTS  P_{n+1|n} (p+q)            BF/MBF  S_n (q)
  2F   P^xx_{n|n}, P^b_n, P^y_n (p) and Sigma_n (p+q, inside the backward model)
  DWY  Sigma_n (p+q), P^b_{n-1|n} (p+q)
  VAR  R (p+q), P_{0|0} (p), the Thomas pivots Delta_n (p), and dense cond(J) (SVD; base
       model only)
computed on the EXACT matrices (all intermediates recomputed at 60 digits, then rounded to
float64: "cond_exact") and, for the base model, on the float64 matrices the library forms
("cond_float"), with the relative error of each float intermediate against its exact value
("intermediate_err"). A cancellation factor kappa = ||subtracted term||_2 / ||result||_2
(max AND median over n, exact intermediates; base model) is given for each covariance update
that is a difference (RTS, BF, MBF, DWY; 2F's information fusion; VAR's pivots).

Control U (not a regime): the base model with (R, Pz0) scaled by 2^-k and the record by
2^-k/2, k = 0, 1, ..., 60. The problem is unchanged up to scale; for even k the float64
scalings are exact powers of two, so any abort or deviation exposes an absolute threshold
in the library (odd k also round the record, deviations at roundoff level).

VAR pivots (base model): besides the library's operation order, the pivot recursion is
replayed with np.linalg.inv for Delta^{-1}, and the ratio ||Delta_float - Delta_exact|| /
lambda_min(Delta_exact) is reported: when it reaches ~1, abort vs. completion is decided by
rounding.

Regimes (each sweep moves ONE parameter of the model's own (A, R) with c = 0.5 as the
default noise correlation; R = D_sd C(c) D_sd, D_sd = I for the base model and G1):
  D  diffuse prior (classical start-up): smoother prior Pz0 = s I, s = 1 .. 1e10; R fixed;
     data drawn with Z_0 from the stationary law.
  S  state-noise starvation: R(eps) = D_eps R D_eps, D_eps = diag(sqrt(eps) I_p, I_q),
     eps = 1 .. 1e-10 (R^xx -> eps R^xx, R^xy -> sqrt(eps) R^xy, R^yy fixed: the
     correlation matrix is kept, R stays PD, cond R ~ 1/eps).
  C  strongly correlated noise: R = D_sd C(c) D_sd, c = 1 - delta, delta = 0.5 .. 1e-10
     (base) or 1e-1 .. 1e-10 (random families); cond R up to ~1e10-1e12, R PD.
  M  slow mixing: A(rho) = (rho / rho(A_model)) A_model (non-normal), rho = 0.9 .. 0.9999.
In S, C and M the smoother prior is the stationary law of the current model (Pz0 =
Sigma_inf, solution of Sigma = A Sigma A^T + R), which is also the law the data start from,
so these regimes carry no prior/data mismatch transient.

Cross-check of the reference: (i) an independent 60-digit pairwise Kalman filter + RTS in
covariance form over the full N = 400 record; (ii) dense 60-digit Gaussian conditioning of
the joint law of Z_{0:12} on y_{0:12}, against the lifted solver on the same truncated
record; (iii) the lifted solver at 80 digits. Base model: one case per regime and (p,q) at
the most extreme sweep value, seed 0. Random families: per (regime, generator, q), the model
with the largest covariance error of any smoother at the most extreme sweep value, seed 0.

Outputs
  experiments/conditioning_exact_results.json   all numbers, summary notes, captions
  figures/conditioning_exact.pdf (+ .png)        single-column figure: random families
  figures/conditioning_letter.pdf (+ .png)       compact letter figure (--letter-fig)
  a printed summary (stdout)
  optional (--base-fig-out PATH): the base-model figure (scratch use)

Run (from the letter directory):
  PYTHONPATH=/path/to/awesomePKF python -B experiments/conditioning_exact.py
Options: --seeds K (base, default 10), --rf-models M (per generator and q, default 20),
         --rf-seeds S (default 3), --workers W (default min(10, cpu)), --quick (smoke test),
         --no-figure, --letter-fig (also draw the compact letter figure), --from-json
         (reprint / redraw / rebuild notes from the saved JSON), --base-fig-out PATH,
         --fig-q 2 (q of the optional base figure).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

sys.dont_write_bytecode = True

import mpmath as mp  # noqa: E402
import numpy as np  # noqa: E402
from scipy.linalg import LinAlgError, cho_factor, cho_solve, solve_discrete_lyapunov  # noqa: E402

try:  # the library must be importable (PYTHONPATH=/path/to/awesomePKF)
    import prg  # noqa: F401
except ImportError:  # pragma: no cover - convenience fallback
    _root = os.environ.get("AWESOMEPKF_ROOT")
    if _root:
        sys.path.insert(0, _root)

import prg.classes.linear_pks as _lpk  # noqa: E402
import prg.classes.pkf as _pkfmod  # noqa: E402
from prg.classes.linear_pks import (  # noqa: E402
    Linear_PKS_BF,
    Linear_PKS_DWY,
    Linear_PKS_MBF,
    Linear_PKS_MF,
    Linear_PKS_RTS,
    Linear_PKS_VAR,
    _cond_xy,
)
from prg.classes.param_linear import ParamLinear  # noqa: E402
from prg.classes.matrix_diagnostics import InvertibleMatrix  # noqa: E402
from prg.classes.matrix_diagnostics.status import Status  # noqa: E402
from prg.classes.pkf import PKF  # noqa: E402
from prg.models.linear._amq import LinearAmQ  # noqa: E402
from prg.utils.exceptions import InvertibilityError  # noqa: E402

HERE = Path(__file__).resolve().parent
LETTER = HERE.parent
OUT_JSON = HERE / "conditioning_exact_results.json"
OUT_FIG = LETTER / "figures" / "conditioning_exact"
OUT_LETTER_FIG = LETTER / "figures" / "conditioning_letter"

N = 400
P = 2
QS = (1, 2)
DPS = 60
N_DENSE = 12
MID = (100, 300)          # interior window for the "_mid" errors (start-up excluded)
LOC_THR, LOC_RATIO = 1e-13, 0.1   # localisation of an error (see localise())
LOC_ABS = 1e-8            # interior error >= this (~sqrt(unit roundoff)) is always "persistent"
SCALE_EXPS = list(range(0, 61))   # control U: (R, Pz0) scaled by alpha = 2^-k, every k
GENS = ("G1", "G2")       # random-model generators (random_model_g1 / random_model_g2)
RF_MODELS = 20            # random models per (generator, q)
RF_SEEDS = 3              # seeds per random-family case (covariances are seed-independent)
RF_VALUES = {"D": [1.0, 1e2, 1e4, 1e6, 1e8, 1e10],
             "S": [1.0, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10],
             "C": [1e-1, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10],
             "M": [0.1, 0.03, 0.01, 0.003, 0.001, 0.0001]}
SMOOTHERS = {"RTS": Linear_PKS_RTS, "BF": Linear_PKS_BF, "MBF": Linear_PKS_MBF,
             "2F": Linear_PKS_MF, "DWY": Linear_PKS_DWY, "VAR": Linear_PKS_VAR}
NAMES = list(SMOOTHERS)
# consistency control (not part of the study): library 2F / DWY classes run with the
# symmetrised backward filter of the previous version of this script (see the docstring)
VARIANTS = {"2F*": "2F", "DWY*": "DWY"}
ALL_NAMES = NAMES + list(VARIANTS)          # what run_case runs
CONS_ROUNDOFF = 1e-12     # stock vs control output difference regarded as round-off level
CONS_RATIO = 10.0         # stock vs control errors (vs the reference) "differ": ratio > this

logging.getLogger("prg").setLevel(logging.CRITICAL)   # regularisations are captured below

# ----------------------------------------------------------------------------------
# Capture of the smoothed covariances BEFORE the library's in-place regularisation,
# and of the library check behind every abort (in-process wrappers; library untouched)
# ----------------------------------------------------------------------------------
_ORIG_CHECK = PKF._check_covariance
_ORIG_CHECK_INV = PKF._check_invertible
_LAST_FAIL: dict = {}      # filled by the wrappers below, read by run_library()


def _mat_stats(M):
    """Small summary of a float64 matrix (for abort diagnostics)."""
    M = np.atleast_2d(np.asarray(M, dtype=float))
    out = {"shape": list(M.shape)}
    if not np.all(np.isfinite(M)):
        out["finite"] = False
        return out
    S = 0.5 * (M + M.T)
    w = np.linalg.eigvalsh(S)
    with np.errstate(all="ignore"):
        det = float(abs(np.linalg.det(M)))
    out.update({"finite": True, "min_eig_sym": float(w[0]), "max_eig_sym": float(w[-1]),
                "norm2": float(np.linalg.norm(M, 2)),
                "rel_asymmetry": float(np.abs(M - M.T).max() / max(np.abs(M).max(), 1e-300)),
                "cond": float(np.linalg.cond(M)), "abs_det": det,
                "matrix": M.tolist()})
    return out


def _recording_check(self, mat, k, name=""):
    cap = getattr(self, "_capture", None)
    if cap is None:
        return _ORIG_CHECK(self, mat, k, name)
    before = np.array(mat, dtype=float, copy=True)
    if name == "PXXkp1_smooth":
        cap["raw"][int(k)] = before
    try:
        _ORIG_CHECK(self, mat, k, name)
    except Exception as e:  # NaN/Inf, or the library could not even regularise
        cap["events"].append({"name": name, "k": int(k), "kind": "raise",
                              "error": f"{type(e).__name__}: {str(e).splitlines()[0]}"})
        _LAST_FAIL.clear()
        _LAST_FAIL.update({"kind": "covariance_check", "matrix_name": name, "k": int(k),
                           "check": ("NaN/Inf guard" if not np.all(np.isfinite(before))
                                     else "CovarianceMatrix check failed and Tikhonov "
                                          "regularisation could not repair it"),
                           "stats": _mat_stats(before)})
        raise
    if not np.array_equal(before, mat):
        w = np.linalg.eigvalsh(0.5 * (before + before.T))
        w2 = np.linalg.eigvalsh(0.5 * (mat + mat.T))
        cap["events"].append({
            "name": name, "k": int(k), "kind": "regularised",
            "min_eig_before": float(w[0]), "max_eig_before": float(w[-1]),
            "min_eig_after": float(w2[0]),
            "eps_added": float(np.mean(np.diag(mat - 0.5 * (before + before.T))))})
    return None


def _recording_check_invertible(self, mat, k, name=""):
    try:
        return _ORIG_CHECK_INV(self, mat, k, name)
    except InvertibilityError:
        rep = InvertibleMatrix(np.array(mat, dtype=float, copy=True)).check()
        failed = [{"check": c.name, "value": c.value, "threshold": c.threshold}
                  for c in rep.checks if c.status == Status.FAIL]
        try:
            cho_factor(np.array(mat, dtype=float))
            chol_ok = True
        except (LinAlgError, ValueError):
            chol_ok = False
        _LAST_FAIL.clear()
        _LAST_FAIL.update({"kind": "invertibility_check", "matrix_name": name, "k": int(k),
                           "failed_checks": failed, "cholesky_of_same_matrix_ok": chol_ok,
                           "stats": _mat_stats(mat)})
        raise


def _wrap_cho_factor(orig, where):
    def cho_factor_recording(a, *args, **kwargs):
        try:
            return orig(a, *args, **kwargs)
        except (LinAlgError, ValueError) as e:
            _LAST_FAIL.clear()
            _LAST_FAIL.update({"kind": "cholesky", "where": where,
                               "error": f"{type(e).__name__}: {e}", "stats": _mat_stats(a)})
            raise
    return cho_factor_recording


PKF._check_covariance = _recording_check   # this process (and its workers) only
PKF._check_invertible = _recording_check_invertible
_lpk.cho_factor = _wrap_cho_factor(cho_factor, "prg.classes.linear_pks")
_pkfmod.cho_factor = _wrap_cho_factor(cho_factor, "prg.classes.pkf")

# ----------------------------------------------------------------------------------
# Replicates of the library's shared backward filter: "v2160" (= the library used, checked
# bitwise) and "control" (the 2F*/DWY* control of the previous version of this script)
# ----------------------------------------------------------------------------------
_STOCK_BACKWARD_FILTER = _lpk._dwy_backward_filter
_STOCK_COND_XY = _lpk._cond_xy


def _cond_xy_v2151(mean_z, Pzz, y, dx):
    """prg.classes.linear_pks._cond_xy as released in awesomepkf 2.15.1 (the conditioned
    covariance is NOT symmetrised). Used only by the 2F*/DWY* consistency control."""
    mx, my = mean_z[:dx], mean_z[dx:]
    Pxx = Pzz[:dx, :dx]
    Pyy, Pyx = Pzz[dx:, dx:], Pzz[dx:, :dx]
    K = np.linalg.solve(Pyy, Pyx).T
    return mx + K @ (y - my), Pxx - K @ Pyx


def backward_filter_replicate(s, N_records, mode="v2160", partial=False):
    """Line-by-line replicate of prg.classes.linear_pks._dwy_backward_filter.

    mode "v2160": the library function of awesomepkf >= 2.16.0: Sigma_n, Q^b_n and P^b_{n-1|n} re-symmetrised at each step, conditioned
      covariances symmetrised by _cond_xy. Checked bitwise against the library at run time
      (meta.replicate_selfcheck).
    mode "control": the 2F*/DWY* control of the previous version of this script, i.e. the
      2.15.1 function (no symmetrisation, 2.15.1 _cond_xy) plus one symmetrisation of
      P^b_{n-1|n} before and of P^b_n after each measurement update (terminal P^b_N
      included); Sigma_n and Q^b_n as computed.
    partial=True (script diagnostics only): stop at the first numpy failure, leave the
    remaining entries None and report the failing n in "failed_at"."""
    lib = mode == "v2160"
    cond = _STOCK_COND_XY if lib else _cond_xy_v2151
    dx, dy, dz = s.dim_x, s.dim_y, s.dim_xy
    A, AT, Qp = s._A, s._AT, s._BmQBT
    NN = N_records - 1
    Sig0 = np.asarray(s.param.Pz0, dtype=float).reshape(dz, dz)
    mz0 = np.asarray(s.param.mz0, dtype=float).reshape(dz, 1)
    Sig: list = [Sig0]
    mz: list = [mz0]
    for _ in range(NN):
        S_next = A @ Sig[-1] @ AT + Qp
        Sig.append(0.5 * (S_next + S_next.T) if lib else S_next)
        mz.append(A @ mz[-1])
    Ab: list = [None] * NN
    Qb: list = [None] * NN
    Mb: list = [None] * NN
    cb: list = [None] * NN
    for n in range(NN):
        Abn = Sig[n] @ AT @ np.linalg.inv(Sig[n + 1])
        Ab[n] = Abn
        Qbn = Sig[n] - Abn @ Sig[n + 1] @ Abn.T
        Qb[n] = 0.5 * (Qbn + Qbn.T) if lib else Qbn
        Mb[n] = Abn[:, :dx]
        cb[n] = mz[n] - Abn @ mz[n + 1]
    ys = [np.asarray(s.history[n]["ykp1"], dtype=float).reshape(dy, 1)
          for n in range(N_records)]
    Pb: list = [None] * (NN + 1)
    xb: list = [None] * (NN + 1)
    Ppred: list = [None] * (NN + 1)
    zpred: list = [None] * (NN + 1)
    xb[NN], Pb[NN] = cond(mz[NN], Sig[NN], ys[NN], dx)
    if not lib:
        Pb[NN] = 0.5 * (Pb[NN] + Pb[NN].T)
    failed_at = None
    for n in range(NN, 0, -1):
        try:
            Pzz = Mb[n - 1] @ Pb[n] @ Mb[n - 1].T + Qb[n - 1]
            Pzz = 0.5 * (Pzz + Pzz.T)
            zp = Ab[n - 1] @ np.vstack([xb[n], ys[n]]) + cb[n - 1]
            Ppred[n - 1] = Pzz
            zpred[n - 1] = zp
            xb[n - 1], Pb[n - 1] = cond(zp, Pzz, ys[n - 1], dx)
            if not lib:
                Pb[n - 1] = 0.5 * (Pb[n - 1] + Pb[n - 1].T)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            if not partial:
                raise
            failed_at = n - 1
            break
    return {"dx": dx, "dy": dy, "dz": dz, "NN": NN, "Sig": Sig, "mz": mz, "Mb": Mb,
            "ys": ys, "xb": xb, "Pb": Pb, "Ppred": Ppred, "zpred": zpred, "failed_at": failed_at}


def _control_backward_filter(s, N_records):
    return backward_filter_replicate(s, N_records, mode="control")


def _v2160_backward_filter(s, N_records):
    return backward_filter_replicate(s, N_records, mode="v2160")


def library_info():
    """awesomepkf version actually imported: package version string, git HEAD and whether
    the source tree has uncommitted changes (with hashes of the diff)."""
    root = Path(sys.modules["prg"].__file__).resolve().parent.parent
    info = {"path": str(root), "prg.__version__": getattr(sys.modules["prg"], "__version__", None)}

    def git(*a):
        return subprocess.run(["git", "-C", str(root), *a], capture_output=True, text=True,
                              check=True).stdout
    try:
        info["git_head"] = git("rev-parse", "HEAD").strip()
        st = [ln for ln in git("status", "--porcelain").splitlines() if ln.strip()]
        diff = git("diff")
        info["modified_files"] = [ln[3:] for ln in st if ln[:2].strip() != "??"]
        info["untracked_files"] = [ln[3:] for ln in st if ln[:2] == "??"]
        info["dirty"] = bool(info["modified_files"])
        info["git_diff_sha256"] = hashlib.sha256(diff.encode()).hexdigest()
        info["git_diff_prg_sha256"] = hashlib.sha256(git("diff", "--", "prg").encode()).hexdigest()
        info["git_diff_numstat"] = git("diff", "--numstat").strip().splitlines()
    except (OSError, subprocess.CalledProcessError) as e:
        info["git_error"] = f"{type(e).__name__}: {e}"
    return info

# ----------------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------------
A0 = {
    1: np.array([[0.50, 0.60, 0.30],
                 [-0.05, 0.40, 0.25],
                 [0.30, 0.20, 0.20]]),
    2: np.array([[0.50, 0.60, 0.30, -0.20],
                 [-0.05, 0.40, 0.25, 0.30],
                 [0.30, 0.20, 0.20, 0.10],
                 [-0.10, 0.30, 0.00, 0.20]]),
}
CXX = np.array([[1.0, 0.3], [0.3, 1.0]])
CYY = {1: np.array([[1.0]]), 2: np.array([[1.0, 0.2], [0.2, 1.0]])}


def _rot(t):
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


KDIR = {1: np.array([[1.0], [1.0]]) / np.sqrt(2.0),
        2: _rot(np.pi / 6) @ np.diag([1.0, 0.6]) @ _rot(-np.pi / 9).T}
C_BASE = 0.5

REGIMES = {
    "D": {"param": "s", "values": [1.0, 1e2, 1e4, 1e6, 1e8, 1e10],
          "title": "diffuse prior (start-up)", "xlabel": r"prior scale $s$",
          "desc": "smoother prior Pz0 = s*I_{p+q}; A = A0, R = R0 fixed; data start from the "
                  "stationary law of (A0, R0)"},
    "S": {"param": "eps", "values": [1.0, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10],
          "title": "state-noise starvation", "xlabel": r"state-noise scale $\varepsilon$",
          "desc": "R = D_eps C(0.5) D_eps, D_eps = diag(sqrt(eps) I_p, I_q); A = A0; "
                  "Pz0 = stationary covariance"},
    "C": {"param": "1-c", "values": [0.5, 1e-1, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10],
          "title": "correlated noise", "xlabel": r"$1-c$ (noise correlation $c$)",
          "desc": "R = C(c), c = 1 - value; A = A0; Pz0 = stationary covariance"},
    "M": {"param": "1-rho", "values": [0.1, 0.03, 0.01, 0.003, 0.001, 0.0001],
          "title": "slow mixing", "xlabel": r"$1-\rho(\mathbf{A})$",
          "desc": "A = (rho/rho(A0)) A0 with rho = 1 - value; R = R0; Pz0 = stationary "
                  "covariance"},
}


def _sym_sqrt(S):
    w, V = np.linalg.eigh(S)
    return (V * np.sqrt(w)) @ V.T


def corr_matrix(q, c):
    """Correlation matrix with cross block c * Cxx^{1/2} K Cyy^{1/2}, ||K||_2 = 1.

    Its Schur complement is Cxx^{1/2} (I - c^2 K K^T) Cxx^{1/2}: PD iff c < 1."""
    cxy = c * _sym_sqrt(CXX) @ KDIR[q] @ _sym_sqrt(CYY[q])
    C = np.block([[CXX, cxy], [cxy.T, CYY[q]]])
    return 0.5 * (C + C.T)


def noise_matrix(q, eps=1.0, c=C_BASE):
    C = corr_matrix(q, c)
    d = np.r_[np.full(P, np.sqrt(eps)), np.ones(q)]
    R = d[:, None] * C * d[None, :]
    return 0.5 * (R + R.T)


def spectral_radius(A):
    return float(np.max(np.abs(np.linalg.eigvals(A))))


def henrici(A):
    """Relative Henrici departure from normality sqrt(||A||_F^2 - sum|lambda|^2)/||A||_F."""
    F = np.linalg.norm(A, "fro")
    return float(np.sqrt(max(0.0, F ** 2 - np.sum(np.abs(np.linalg.eigvals(A)) ** 2))) / F)


def stationary_cov(A, R):
    S = solve_discrete_lyapunov(A, R)
    return 0.5 * (S + S.T)


def _random_corr(rng, k):
    """Random correlation matrix (normalised Wishart with k + 2 degrees of freedom)."""
    G = rng.standard_normal((k, k + 2))
    W = G @ G.T
    d = 1.0 / np.sqrt(np.diag(W))
    C = d[:, None] * W * d[None, :]
    return 0.5 * (C + C.T)


def _haar_orthogonal(rng, k):
    Qm, Rm = np.linalg.qr(rng.standard_normal((k, k)))
    return Qm * np.sign(np.diag(Rm))[None, :]


def _random_corr_spectrum(rng, k):
    """Random correlation matrix with a random spectrum: V diag(l) V^T, l ~ U[0.2, 1.8],
    V Haar, then normalised to a unit diagonal."""
    if k == 1:
        return np.ones((1, 1))
    V = _haar_orthogonal(rng, k)
    W = (V * rng.uniform(0.2, 1.8, k)) @ V.T
    d = 1.0 / np.sqrt(np.diag(W))
    C = d[:, None] * W * d[None, :]
    return 0.5 * (C + C.T)


def random_model_g1(q, j):
    """Generator G1, model #j (the "random C" generator of the previous version).

    A: Gaussian entries, rescaled to a spectral radius drawn in [0.6, 0.9], then its
    back-action block A^xy rescaled to ||A^xy||_2 = 0.3; redrawn until rho(A) < 0.95 and
    the relative Henrici departure from normality is > 0.3. Noise correlation structure:
    random correlation matrices Cxx (p x p), Cyy (q x q) (normalised Wishart) and a random
    direction K with ||K||_2 = 1, so that c is the largest canonical correlation between
    the X- and Y-noise channels (C(c) is PD iff c < 1). Unit noise variances."""
    rng = np.random.default_rng(50_000 + 100 * q + j)
    d = P + q
    while True:
        A = rng.standard_normal((d, d))
        A *= rng.uniform(0.6, 0.9) / spectral_radius(A)
        A[:P, P:] *= 0.3 / np.linalg.norm(A[:P, P:], 2)
        if spectral_radius(A) < 0.95 and henrici(A) > 0.3:
            break
    Cxx, Cyy = _random_corr(rng, P), _random_corr(rng, q)
    K = rng.standard_normal((P, q))
    K /= np.linalg.norm(K, 2)
    return {"A": A, "Cxx": Cxx, "Cyy": Cyy, "K": K, "sd": np.ones(d)}


def random_model_g2(q, j):
    """Generator G2, model #j (independent of G1: Schur form, random noise scales).

    A = U T U^T with U Haar-orthogonal and T upper triangular: real eigenvalues
    lambda_i ~ U[-0.85, 0.85] on the diagonal, strictly-upper entries N(0, s^2) with
    s ~ U[0.3, 1] (non-normality); redrawn until ||A^xy||_2 is in [0.2, 0.4] and
    rho(A) >= 0.5 (no rescaling, so the eigenvalues stay in [-0.85, 0.85]). Noise:
    correlation matrices Cxx, Cyy with a random spectrum (eigenvalues ~ U[0.2, 1.8],
    unit diagonal), K = U_x diag(1, s_2) U_y^T (Haar U_x, U_y; s_2 ~ U[0, 1] when q = 2;
    ||K||_2 = 1, so c is again the largest canonical correlation), and noise standard
    deviations ~ logU[0.5, 2] on every channel: R = D_sd C(c) D_sd."""
    rng = np.random.default_rng(70_000 + 100 * q + j)
    d = P + q
    while True:
        lam = rng.uniform(-0.85, 0.85, d)
        T = np.diag(lam) + np.triu(rng.standard_normal((d, d)), 1) * rng.uniform(0.3, 1.0)
        U = _haar_orthogonal(rng, d)
        A = U @ T @ U.T
        nxy = np.linalg.norm(A[:P, P:], 2)
        if 0.2 <= nxy <= 0.4 and np.max(np.abs(lam)) >= 0.5:
            break
    Cxx, Cyy = _random_corr_spectrum(rng, P), _random_corr_spectrum(rng, q)
    Ux, Uy = _haar_orthogonal(rng, P), _haar_orthogonal(rng, q)
    sv = np.zeros((P, q))
    sv[0, 0] = 1.0
    if q > 1:
        sv[1, 1] = rng.uniform(0.0, 1.0)
    K = Ux @ sv @ Uy.T
    sd = np.exp(rng.uniform(np.log(0.5), np.log(2.0), d))
    return {"A": A, "Cxx": Cxx, "Cyy": Cyy, "K": K, "sd": sd}


GENERATORS = {"G1": random_model_g1, "G2": random_model_g2}


def model_spec(q, gen=None, model_id=None):
    """(A, Cxx, Cyy, K, sd) of the base model (gen None) or of random model #model_id."""
    if gen is None:
        return {"A": A0[q].copy(), "Cxx": CXX, "Cyy": CYY[q], "K": KDIR[q],
                "sd": np.ones(P + q)}
    return GENERATORS[gen](q, model_id)


def corr_matrix_general(Cxx, Cyy, K, c):
    cxy = c * _sym_sqrt(Cxx) @ K @ _sym_sqrt(Cyy)
    C = np.block([[Cxx, cxy], [cxy.T, Cyy]])
    return 0.5 * (C + C.T)


def build_model(regime, q, value, gen=None, model_id=None):
    """Model of one sweep point. Base model (gen None): identical (bitwise) to the previous
    version's construction; random models: the same regime transformations applied to the
    model's own (A, Cxx, Cyy, K, sd), with c = 0.5 outside regime C."""
    sp = model_spec(q, gen, model_id)
    A = sp["A"]
    c, eps = C_BASE, 1.0
    if regime == "S":
        eps = value
    elif regime == "C":
        c = 1.0 - value
    elif regime == "M":
        A = A * ((1.0 - value) / spectral_radius(A))
    Cm = corr_matrix_general(sp["Cxx"], sp["Cyy"], sp["K"], c)
    d = np.r_[np.full(P, np.sqrt(eps)), np.ones(q)] * sp["sd"]
    R = d[:, None] * Cm * d[None, :]
    R = 0.5 * (R + R.T)
    Sig_inf = stationary_cov(A, R)
    Pz0 = value * np.eye(P + q) if regime == "D" else Sig_inf.copy()
    return {"A": A, "R": R, "Pz0": Pz0, "Pz0_data": Sig_inf}


def make_param(model, q):
    m = LinearAmQ(P, q, A=model["A"], mQ=model["R"], mz0=np.zeros((P + q, 1)),
                  Pz0=model["Pz0"], pairwiseModel=True)
    pp = m.get_params().copy()
    pp.pop("dim_x")
    pp.pop("dim_y")
    return ParamLinear(0, P, q, **pp)


def simulate(model, q, seed, n_steps=N):
    """Float64 record (x, y)_{0:N}; common random numbers across a sweep (seed, q)."""
    rng = np.random.default_rng(1000 * q + seed)
    d = P + q
    xi = rng.standard_normal((n_steps + 1, d))
    L0 = np.linalg.cholesky(model["Pz0_data"])
    LR = np.linalg.cholesky(model["R"])
    Z = np.empty((n_steps + 1, d))
    Z[0] = L0 @ xi[0]
    for n in range(n_steps):
        Z[n + 1] = model["A"] @ Z[n] + LR @ xi[n + 1]
    return Z[:, :P].copy(), Z[:, P:].copy()


# ----------------------------------------------------------------------------------
# Exact reference (mpmath, object arrays)
# ----------------------------------------------------------------------------------
def to_mp(M):
    M = np.atleast_2d(np.asarray(M, dtype=float))
    out = np.empty(M.shape, dtype=object)
    for idx, v in np.ndenumerate(M):
        out[idx] = mp.mpf(float(v))
    return out


def mp_eye(n):
    out = np.empty((n, n), dtype=object)
    for i in range(n):
        for j in range(n):
            out[i, j] = mp.mpf(1) if i == j else mp.mpf(0)
    return out


def mp_zeros(shape):
    out = np.empty(shape, dtype=object)
    out[...] = mp.mpf(0)
    return out


def mp_inv(M):
    """Gauss-Jordan inverse with partial pivoting (object array of mpf)."""
    n = M.shape[0]
    a = [list(M[i]) + [mp.mpf(1) if i == j else mp.mpf(0) for j in range(n)]
         for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(a[r][col]))
        if a[piv][col] == 0:
            raise ZeroDivisionError("singular matrix in mp_inv")
        a[col], a[piv] = a[piv], a[col]
        ip = 1 / a[col][col]
        a[col] = [v * ip for v in a[col]]
        for r in range(n):
            if r != col:
                f = a[r][col]
                if f != 0:
                    a[r] = [vr - f * vc for vr, vc in zip(a[r], a[col])]
    out = np.empty((n, n), dtype=object)
    for i in range(n):
        out[i, :] = a[i][n:]
    return out


def mpf_arr(M):
    return np.asarray(M, dtype=object).astype(float)


def mpf_dd(M):
    """Double-double split of an mpf array: (hi, lo) float64 with hi + lo = M to ~1e-32.

    Errors against the reference are formed as (est - hi) - lo, which is accurate to
    ~u * |error| + u^2 |M| instead of u * |M| (rounding the reference first would leave
    a ~1e-16 floor of metric noise on every roundoff-level error)."""
    M = np.asarray(M, dtype=object)
    hi = M.astype(float)
    lo = np.empty(M.shape, dtype=float)
    for idx, v in np.ndenumerate(M):
        lo[idx] = float(v - mp.mpf(hi[idx]))
    return hi, lo


class LiftedReference:
    """Exact lifted solver: J x = eta, diag + lag-one blocks of J^{-1} (Takahashi).

    J_00 = P00^{-1} + M^T R^{-1} M, J_nn = E^T R^{-1} E + M^T R^{-1} M (0<n<N),
    J_NN = E^T R^{-1} E, J_{n,n-1} = -E^T R^{-1} M; eta from c_n = [0; y_n] - A [0; y_{n-1}].
    The covariance part is data-free and computed once per model."""

    def __init__(self, A, R, Pz0, q, n_steps, dps=DPS):
        self.dps, self.q, self.N = dps, q, n_steps
        with mp.workdps(dps):
            p = P
            self.A = to_mp(A)
            Rm = to_mp(R)
            P0 = to_mp(Pz0)
            self.Rinv = mp_inv(Rm)
            self.M = self.A[:, :p]
            self.Axy, self.Ayy = self.A[:p, p:], self.A[p:, p:]
            EtRinvE = self.Rinv[:p, :p]
            MtRinvM = self.M.T.dot(self.Rinv).dot(self.M)
            self.Loff = -(self.Rinv[:p, :].dot(self.M))              # J_{n,n-1}
            self.MtRinv = self.M.T.dot(self.Rinv)
            self.EtRinv = self.Rinv[:p, :]
            self.P0yy_inv = mp_inv(P0[p:, p:])
            self.P0xy = P0[:p, p:]
            P00 = P0[:p, :p] - self.P0xy.dot(self.P0yy_inv).dot(P0[p:, :p])
            self.P00inv = mp_inv(P00)
            Nn = n_steps
            D = []
            self.D = D
            for n in range(Nn + 1):
                Dn = mp_zeros((p, p))
                if n == 0:
                    Dn = Dn + self.P00inv
                if n >= 1:
                    Dn = Dn + EtRinvE
                if n <= Nn - 1:
                    Dn = Dn + MtRinvM
                D.append(Dn)
            self.Dinv = [None] * (Nn + 1)
            self.T = [None] * (Nn + 1)            # T_n = Loff Delta_{n-1}^{-1}
            self.Delta = [None] * (Nn + 1)
            Delta = D[0]
            self.Delta[0] = Delta
            self.Dinv[0] = mp_inv(Delta)
            for n in range(1, Nn + 1):
                self.T[n] = self.Loff.dot(self.Dinv[n - 1])
                Delta = D[n] - self.T[n].dot(self.Loff.T)
                self.Delta[n] = Delta
                self.Dinv[n] = mp_inv(Delta)
            Pn = [None] * (Nn + 1)
            Cn = [None] * (Nn + 1)                 # Cov(x_{n+1}, x_n | y)
            Pn[Nn] = self.Dinv[Nn]
            for n in range(Nn - 1, -1, -1):
                W = self.Dinv[n].dot(self.Loff.T)
                Pn[n] = self.Dinv[n] + W.dot(Pn[n + 1]).dot(W.T)
                Cn[n] = -(Pn[n + 1].dot(W.T))
            self.Pmp = Pn
            self.Cmp = Cn
            dP = [mpf_dd(x) for x in Pn]
            dC = [mpf_dd(x) for x in Cn[:-1]]
            self.P, self.P_lo = np.array([h for h, _ in dP]), np.array([lo for _, lo in dP])
            self.C, self.C_lo = np.array([h for h, _ in dC]), np.array([lo for _, lo in dC])

    def mean(self, y, dps=None):
        dps = dps or self.dps
        with mp.workdps(dps):
            p, Nn = P, self.N
            ym = [to_mp(np.asarray(y[n], dtype=float).reshape(-1, 1)) for n in range(Nn + 1)]
            mu0 = self.P0xy.dot(self.P0yy_inv).dot(ym[0])
            c = [None] + [np.vstack([-(self.Axy.dot(ym[n - 1])),
                                     ym[n] - self.Ayy.dot(ym[n - 1])])
                          for n in range(1, Nn + 1)]
            eta = []
            for n in range(Nn + 1):
                e = mp_zeros((p, 1))
                if n == 0:
                    e = e + self.P00inv.dot(mu0)
                if n >= 1:
                    e = e - self.EtRinv.dot(c[n])
                if n <= Nn - 1:
                    e = e + self.MtRinv.dot(c[n + 1])
                eta.append(e)
            ct = [None] * (Nn + 1)
            ct[0] = eta[0]
            for n in range(1, Nn + 1):
                ct[n] = eta[n] - self.T[n].dot(ct[n - 1])
            x = [None] * (Nn + 1)
            x[Nn] = self.Dinv[Nn].dot(ct[Nn])
            for n in range(Nn - 1, -1, -1):
                x[n] = self.Dinv[n].dot(ct[n] - self.Loff.T.dot(x[n + 1]))
            return x

    def mean_dd(self, y):
        """Smoothed means as a double-double pair (hi, lo), each of shape (N+1, p)."""
        x = self.mean(y)
        with mp.workdps(self.dps):
            parts = [mpf_dd(np.asarray(v, dtype=object).ravel()) for v in x]
        return np.array([h for h, _ in parts]), np.array([lo for _, lo in parts])


def mp_kf_rts(A, R, Pz0, y, q, dps=DPS):
    """Independent route: pairwise Kalman filter + RTS in covariance form (mpmath)."""
    with mp.workdps(dps):
        p, Nn = P, len(y) - 1
        Am, Rm, P0 = to_mp(A), to_mp(R), to_mp(Pz0)
        M, Ay = Am[:, :p], Am[:, p:]
        ym = [to_mp(np.asarray(y[n], dtype=float).reshape(-1, 1)) for n in range(Nn + 1)]
        iyy = mp_inv(P0[p:, p:])
        xf = [None] * (Nn + 1)
        Pf = [None] * (Nn + 1)
        zp = [None] * (Nn + 1)
        Pp = [None] * (Nn + 1)
        xf[0] = P0[:p, p:].dot(iyy).dot(ym[0])
        Pf[0] = P0[:p, :p] - P0[:p, p:].dot(iyy).dot(P0[p:, :p])
        for n in range(1, Nn + 1):
            zp[n] = M.dot(xf[n - 1]) + Ay.dot(ym[n - 1])
            Pp[n] = M.dot(Pf[n - 1]).dot(M.T) + Rm
            K = Pp[n][:p, p:].dot(mp_inv(Pp[n][p:, p:]))
            xf[n] = zp[n][:p] + K.dot(ym[n] - zp[n][p:])
            Pf[n] = Pp[n][:p, :p] - K.dot(Pp[n][p:, :p])
        xs = [None] * (Nn + 1)
        Ps = [None] * (Nn + 1)
        Cs = [None] * Nn
        xs[Nn], Ps[Nn] = xf[Nn], Pf[Nn]
        for n in range(Nn - 1, -1, -1):
            G = Pf[n].dot(M.T).dot(mp_inv(Pp[n + 1]))
            zs = np.vstack([xs[n + 1], ym[n + 1]])
            xs[n] = xf[n] + G.dot(zs - zp[n + 1])
            Pzz = mp_zeros((p + q, p + q))
            Pzz[:p, :p] = Ps[n + 1]
            Ps[n] = Pf[n] + G.dot(Pzz - Pp[n + 1]).dot(G.T)
            Cs[n] = Ps[n + 1].dot(G[:, :p].T)
        return xs, Ps, Cs


def mp_dense_conditioning(A, R, Pz0, y, q, dps=DPS):
    """Independent route: dense Gaussian conditioning of x_{0:N} on y_{0:N} (short N)."""
    with mp.workdps(dps):
        p, d, Nn = P, P + q, len(y) - 1
        Am, Rm = to_mp(A), to_mp(R)
        Sig = [to_mp(Pz0)]
        for _ in range(Nn):
            Sig.append(Am.dot(Sig[-1]).dot(Am.T) + Rm)
        Apow = [mp_eye(d)]
        for _ in range(Nn):
            Apow.append(Am.dot(Apow[-1]))
        big = mp_zeros(((Nn + 1) * d, (Nn + 1) * d))
        for i in range(Nn + 1):
            for j in range(i + 1):
                blk = Apow[i - j].dot(Sig[j])          # Cov(Z_i, Z_j), i >= j
                big[i * d:(i + 1) * d, j * d:(j + 1) * d] = blk
                big[j * d:(j + 1) * d, i * d:(i + 1) * d] = blk.T
        ix = [n * d + k for n in range(Nn + 1) for k in range(p)]
        iy = [n * d + p + k for n in range(Nn + 1) for k in range(q)]
        Cxx = big[np.ix_(ix, ix)]
        Cxy = big[np.ix_(ix, iy)]
        Cyy = big[np.ix_(iy, iy)]
        yv = to_mp(np.asarray(y, dtype=float).reshape(-1, 1))
        G = Cxy.dot(mp_inv(Cyy))
        xm = G.dot(yv)
        Pm = Cxx - G.dot(Cxy.T)
        xs = [xm[n * p:(n + 1) * p] for n in range(Nn + 1)]
        Ps = [Pm[n * p:(n + 1) * p, n * p:(n + 1) * p] for n in range(Nn + 1)]
        Cs = [Pm[(n + 1) * p:(n + 2) * p, n * p:(n + 1) * p] for n in range(Nn)]
        return xs, Ps, Cs


def mp_rel_diff(xa, xb, Pa, Pb, Ca=None, Cb=None):
    """Max relative differences between two mp solutions (computed in mp, then float)."""
    num = max(mp.sqrt(sum(v ** 2 for v in (a - b).ravel())) for a, b in zip(xa, xb))
    den = max(mp.sqrt(sum(v ** 2 for v in a.ravel())) for a in xa)
    mean_d = float(num / den)
    cov_d = 0.0
    for a, b in zip(Pa, Pb):
        diff = mp.matrix((a - b).tolist())
        ref = mp.matrix(b.tolist())
        cov_d = max(cov_d, float(mp.mnorm(diff, "f") / mp.mnorm(ref, "f")))
    lag_d = None
    if Ca is not None:
        lag_d = 0.0
        for a, b in zip(Ca, Cb):
            diff = mp.matrix((a - b).tolist())
            ref = mp.matrix(b.tolist())
            lag_d = max(lag_d, float(mp.mnorm(diff, "f") / mp.mnorm(ref, "f")))
    return {"mean": mean_d, "cov": cov_d, "lag1": lag_d}


# ----------------------------------------------------------------------------------
# Library runs and diagnostics
# ----------------------------------------------------------------------------------
def run_library(cls, param, X, Y, q, control=False, bfilter=None):
    """One library run. control=True: 2F*/DWY* consistency control (the control backward
    filter and the 2.15.1 _cond_xy swapped into prg.classes.linear_pks for the duration of
    this call, this process only). bfilter: another backward filter to swap in (self-check).
    After the run, the number of in-place regularisations seen by the wrapper is compared
    with the library's own record (PKF.covariance_regularisations)."""
    s = cls(param, sKey=0)
    s._capture = {"raw": {}, "events": []}
    gen = ((k, X[k].reshape(P, 1).copy(), Y[k].reshape(q, 1).copy()) for k in range(N + 1))
    err = None
    _LAST_FAIL.clear()
    if control:
        _lpk._dwy_backward_filter = _control_backward_filter
        _lpk._cond_xy = _cond_xy_v2151
    elif bfilter is not None:
        _lpk._dwy_backward_filter = bfilter
    try:
        s.process_N_data_smoother(N=N, data_generator=gen)
    except Exception as e:  # noqa: BLE001 - every library failure is a result
        where = None
        for fr in traceback.extract_tb(e.__traceback__):
            if f"{os.sep}prg{os.sep}" in fr.filename:
                where = f"{Path(fr.filename).name}:{fr.name}:{fr.lineno}"
        err = {"type": type(e).__name__, "msg": str(e).split("\n")[0], "where": where,
               "step": getattr(e, "step", None),
               "matrix": getattr(e, "matrix_name", None),
               "cause_type": type(e.__cause__).__name__ if e.__cause__ is not None else None,
               "check": dict(_LAST_FAIL) if _LAST_FAIL else None}
    finally:
        _lpk._dwy_backward_filter = _STOCK_BACKWARD_FILTER
        _lpk._cond_xy = _STOCK_COND_XY
    n_wrap = sum(e["kind"] == "regularised" for e in s._capture["events"])
    lib_list = getattr(s, "covariance_regularisations", None)
    s._capture["reg_check"] = {"wrapper": n_wrap,
                               "library": None if lib_list is None else len(lib_list)}
    return s, err


def classify_abort(err):
    """Library check behind an abort, and whether it is a tolerance artefact.

    tolerance artefact = the abort comes from a fixed-threshold test of InvertibleMatrix
    (normalised determinant |det| / prod ||row||, cond >= 1e12, rank, inversion residual) on
    a float64 matrix whose Cholesky factorisation nevertheless succeeds."""
    chk = err.get("check") or {}
    kind = chk.get("kind")
    if kind == "invertibility_check":
        names = [f["check"] for f in chk.get("failed_checks", [])]
        txt = "InvertibleMatrix check failed: " + ", ".join(
            f"{f['check']} (value {f['value']:.3g}, threshold {f['threshold']:.3g})"
            if f.get("value") is not None and f.get("threshold") is not None else f["check"]
            for f in chk.get("failed_checks", []))
        art = bool(chk.get("cholesky_of_same_matrix_ok")) and len(names) > 0
        return {"check": txt, "failed_checks": names, "tolerance_artefact": art,
                "why": ("fixed-threshold test on a Cholesky-factorisable float64 matrix"
                        if art else "threshold test on a matrix that is not Cholesky-"
                        "factorisable in float64 either")}
    if kind == "cholesky":
        st = chk.get("stats", {})
        return {"check": f"Cholesky factorisation (scipy cho_factor, {chk.get('where')}) failed: "
                         f"float64 matrix not numerically PD (min eig of its symmetric part "
                         f"{st.get('min_eig_sym', float('nan')):.3g}, rel. asymmetry "
                         f"{st.get('rel_asymmetry', float('nan')):.2g})",
                "failed_checks": ["cholesky"], "tolerance_artefact": False,
                "why": "no tolerance involved: the float64 matrix is not positive definite"}
    if kind == "covariance_check":
        return {"check": chk.get("check"), "failed_checks": ["covariance_check"],
                "tolerance_artefact": False, "why": "NaN/Inf or unrepairable covariance"}
    if err.get("type") in ("LinAlgError", "FloatingPointError", "ValueError"):
        return {"check": f"unguarded numpy {err.get('type')} ('{err.get('msg')}') raised in "
                         f"{err.get('where')}; no library check involved",
                "failed_checks": ["unguarded_numpy"], "tolerance_artefact": False,
                "why": "numpy linear-algebra failure outside the library's checks"}
    return {"check": f"unclassified ({err.get('type')}: {err.get('msg')} in {err.get('where')})",
            "failed_checks": [], "tolerance_artefact": None, "why": None}


def _stats(v):
    v = np.asarray(v, dtype=float)
    if v.size == 0 or not np.any(np.isfinite(v)):
        return {"max": None, "argmax": None, "median": None}
    return {"max": float(np.nanmax(v)), "argmax": int(np.nanargmax(v)),
            "median": float(np.nanmedian(v))}


def _norm2(M):
    return float(np.linalg.norm(M, 2))


def exact_intermediates(A, R, Pz0, q, n_steps, dps=DPS):
    """Data-free intermediate covariances of the six smoothers, in mpmath (60 digits).

    Forward pairwise filter (P^xx_{n|n}, P_{n|n-1}), prior recursion Sigma_n, the
    time-reversed model (A^b_n, Q^b_n, M^b_n), the backward filter (P^b_n, P^b_{n-1|n})
    and P^y_n = Var(X_n | Y_n) under N(0, Sigma_n): the same recursions as the library
    (Appendix of the letter), evaluated exactly. Returned rounded to float64."""
    with mp.workdps(dps):
        p, Nn = P, n_steps
        Am, Rm, P0 = to_mp(A), to_mp(R), to_mp(Pz0)
        M = Am[:, :p]

        def ccov(Pzz):
            return Pzz[:p, :p] - Pzz[:p, p:].dot(mp_inv(Pzz[p:, p:])).dot(Pzz[p:, :p])

        Pf = [ccov(P0)]
        Ppred = [None]
        for _ in range(1, Nn + 1):
            Pp = M.dot(Pf[-1]).dot(M.T) + Rm
            Ppred.append(Pp)
            Pf.append(ccov(Pp))
        Sig = [P0]
        for _ in range(Nn):
            Sig.append(Am.dot(Sig[-1]).dot(Am.T) + Rm)
        Mb, Qb = [], []
        for n in range(Nn):
            Abn = Sig[n].dot(Am.T).dot(mp_inv(Sig[n + 1]))
            Qb.append(Sig[n] - Abn.dot(Sig[n + 1]).dot(Abn.T))
            Mb.append(Abn[:, :p])
        Pb = [None] * (Nn + 1)
        Pbp = [None] * Nn
        Pb[Nn] = ccov(Sig[Nn])
        for n in range(Nn, 0, -1):
            Pbp[n - 1] = Mb[n - 1].dot(Pb[n]).dot(Mb[n - 1].T) + Qb[n - 1]
            Pb[n - 1] = ccov(Pbp[n - 1])
        Py = [ccov(S) for S in Sig]
        f = lambda L: np.array([mpf_arr(x) for x in L])  # noqa: E731
        return {"Pf": f(Pf), "Ppred": f(Ppred[1:]), "Sig": f(Sig), "Mb": f(Mb), "Qb": f(Qb),
                "Pb": f(Pb), "Pbp": f(Pbp), "Py": f(Py)}


def exact_counterpart(name, step, ex, ref, model, delta_fail=None):
    """Exact (60-digit, rounded) value of the matrix a library check failed on."""
    n = step
    M = model["A"][:, :P]
    try:
        if name == "Pb_backward":
            return ex["Pb"][n]
        if name == "Py_prior":
            return ex["Py"][n]
        if name in ("PXXkp1_update", "PXXkp1_update_Joseph"):
            return ex["Pf"][n]
        if name == "info_smooth":
            return np.linalg.inv(ref.P[n])
        if name == "PZZpred_backward":
            return ex["Pbp"][n - 1]
        if name == "PZZkp1_predict":
            return M @ ex["Pf"][n] @ M.T + model["R"]
        if name == "Skp1":
            return ex["Ppred"][n - 1][P:, P:]
        if name == "Pkp1_predict":
            return ex["Ppred"][n - 1]
        if name == "Sigma22":
            return model["Pz0"][P:, P:]
        if name == "Delta" and delta_fail is not None:
            return mpf_arr(ref.Delta[delta_fail])
        if name == "PXXkp1_smooth":
            return ref.P[n]
    except (IndexError, TypeError, np.linalg.LinAlgError):
        return None
    return None


def diagnose_abort(err, ex, ref, model, delta_fail=None):
    """Attach the check classification and the exact counterpart of the failing matrix."""
    cls = classify_abort(err)
    out = {"cause": cls}
    name, step = err.get("matrix"), err.get("step")
    if name == "Delta":
        out["failing_pivot_index_replicate"] = delta_fail
    Mex = exact_counterpart(name, step, ex, ref, model, delta_fail) if step is not None else None
    if Mex is not None:
        Mex = np.atleast_2d(Mex)
        w = np.linalg.eigvalsh(0.5 * (Mex + Mex.T))
        out["exact_min_eig"] = float(w[0])
        out["exact_cond"] = float(np.linalg.cond(Mex))
        st = (err.get("check") or {}).get("stats") or {}
        if st.get("matrix") is not None and np.shape(st["matrix"]) == Mex.shape:
            Mf = np.array(st["matrix"], dtype=float)
            out["float_rel_err"] = float(np.linalg.norm(Mf - Mex, 2) / np.linalg.norm(Mex, 2))
        fe = out.get("float_rel_err")
        if out["exact_min_eig"] <= 0 or out["exact_cond"] >= 1e15:
            out["diagnosis_code"] = "exact_singular"
            out["diagnosis"] = "exact matrix numerically singular at float64 precision"
        elif fe is not None and fe >= 1:
            out["diagnosis_code"] = "float_off"
            out["diagnosis"] = ("float64 matrix off its exact value by a relative error >= 1 "
                                "while the exact matrix is PD with cond " f"{out['exact_cond']:.2g}")
        elif fe is not None and fe * out["exact_cond"] >= 0.1:
            out["diagnosis_code"] = "rounding_vs_eigengap"
            out["diagnosis"] = ("rounding error of the float64 matrix comparable to its exact "
                                "smallest relative eigenvalue (rel. err x cond >= 0.1)")
        elif cls.get("tolerance_artefact"):
            out["diagnosis_code"] = "threshold_on_wellposed"
            out["diagnosis"] = "threshold test failed on a well-defined matrix"
        else:
            out["diagnosis_code"] = "other"
            out["diagnosis"] = "other"
    else:
        out["diagnosis_code"] = "no_exact_counterpart"
    return out


def _conds(mats):
    return _stats([np.linalg.cond(np.atleast_2d(x)) for x in mats])


def _relerr(a_list, b_list):
    """max_n ||a_n - b_n||_2 / ||b_n||_2 and its argmax (b exact). Entries a_n = None (not
    computed) are skipped; a non-finite a_n counts as an infinite error, stored as 1e308."""
    r = []
    for a, b in zip(a_list, b_list):
        if a is None:
            r.append(np.nan)
            continue
        a = np.atleast_2d(a)
        r.append(_norm2(a - b) / _norm2(b) if np.all(np.isfinite(a)) else np.inf)
    r = np.minimum(np.array(r, dtype=float), 1e308)
    return {"max": float(np.nanmax(r)), "argmax": int(np.nanargmax(r)),
            "median": float(np.nanmedian(r))}


def conditioning(inst, model, q, ref, ex, dense_J=True):
    """Seed-independent diagnostics.

    cond_exact : condition numbers of the EXACT matrices each smoother inverts (60-digit
                 values rounded to float64; reliable while cond << 1e16)
    cond_float : the same for the float64 matrices the library actually forms
    kappa      : cancellation factors (exact intermediates)
    interm_err : relative error of the library's float64 intermediates vs exact
    """
    A, R = model["A"], model["R"]
    M = A[:, :P]
    H = inst.history
    Nn = len(H) - 1
    Pref = ref.P
    nref = np.array([_norm2(x) for x in Pref])
    # ---------------- exact
    Pf_e, Pp_e = ex["Pf"], ex["Ppred"]
    Pzz_e = [M @ Pf_e[n] @ M.T + R for n in range(Nn)]               # P_{n+1|n}, n=0..N-1
    ce = {"P_pred[RTS]": _conds(Pzz_e),
          "S[BF,MBF]": _conds([x[P:, P:] for x in Pp_e]),
          "Sigma[2F,DWY]": _conds(ex["Sig"]),
          "Pb[2F]": _conds(ex["Pb"]),
          "Pb_pred[DWY]": _conds(ex["Pbp"]),
          "Pf[2F]": _conds(Pf_e),
          "Py[2F]": _conds(ex["Py"]),
          "Qb[2F,DWY]": _conds(ex["Qb"]),
          "Delta[VAR]": _conds([mpf_arr(x) for x in ref.Delta]),
          "R[VAR]": float(np.linalg.cond(R)),
          "P00[VAR]": float(np.linalg.cond(Pf_e[0]))}
    Dex = [mpf_arr(x) for x in ref.D]
    Lex = mpf_arr(ref.Loff)
    J = np.zeros(((Nn + 1) * P, (Nn + 1) * P))
    for n in range(Nn + 1):
        J[n * P:(n + 1) * P, n * P:(n + 1) * P] = Dex[n]
        if n >= 1:
            J[n * P:(n + 1) * P, (n - 1) * P:n * P] = Lex
            J[(n - 1) * P:n * P, n * P:(n + 1) * P] = Lex.T
    if dense_J:
        ce["J[VAR] (dense)"] = float(np.linalg.cond(J))
    kap = {
        "RTS": _stats([_norm2(Pf_e[n] @ M.T @ np.linalg.solve(Pzz_e[n], M @ Pf_e[n])) / nref[n]
                       for n in range(Nn)]),
        "BF": _stats([_norm2(Pp_e[n - 1][:P, :P] - Pref[n]) / nref[n] for n in range(1, Nn + 1)]
                     + [_norm2(Pf_e[0] - Pref[0]) / nref[0]]),
        "MBF": _stats([_norm2(Pf_e[n] - Pref[n]) / nref[n] for n in range(Nn)]),
        "2F": _stats([_norm2(np.linalg.inv(ex["Py"][n])) * nref[n] for n in range(Nn + 1)]),
        "DWY": _stats([_norm2(ex["Pb"][n] @ ex["Mb"][n - 1].T
                              @ np.linalg.solve(ex["Pbp"][n - 1], ex["Mb"][n - 1] @ ex["Pb"][n]))
                       / nref[n] for n in range(1, Nn + 1)]),
        "VAR": _stats([_norm2(Lex @ np.linalg.solve(mpf_arr(ref.Delta[n - 1]), Lex.T))
                       / _norm2(mpf_arr(ref.Delta[n])) for n in range(1, Nn + 1)]),
    }
    # ---------------- float64, as formed by the library
    Pf = [np.atleast_2d(H[n]["PXXkp1_update"]) for n in range(Nn + 1)]
    S = [np.atleast_2d(H[n]["Skp1"]) for n in range(1, Nn + 1)]
    Pzz = [M @ Pf[n] @ M.T + R for n in range(Nn)]
    cf = {"P_pred[RTS]": _conds(Pzz), "S[BF,MBF]": _conds(S), "Pf[2F]": _conds(Pf)}
    ie = {"Pf (forward filter)": _relerr(Pf, Pf_e),
          "P_pred (forward filter)": _relerr(Pzz, Pzz_e)}
    try:
        bf = _STOCK_BACKWARD_FILTER(inst, Nn + 1)
        Sig, Pb, Pbp, mz, ys = bf["Sig"], bf["Pb"], bf["Ppred"], bf["mz"], bf["ys"]
        cf["Sigma[2F,DWY]"] = _conds(Sig)
        cf["Pb[2F]"] = _conds(Pb)
        cf["Pb_pred[DWY]"] = _conds(Pbp[:Nn])
        Py = [np.atleast_2d(_cond_xy(mz[n], Sig[n], ys[n], P)[1]) for n in range(Nn + 1)]
        cf["Py[2F]"] = _conds(Py)
        ie["Sigma (prior recursion)"] = _relerr(Sig, ex["Sig"])
        ie["Pb (backward filter)"] = _relerr(Pb, ex["Pb"])
        ie["Pb_pred (backward filter)"] = _relerr(Pbp[:Nn], ex["Pbp"])
        ie["Py"] = _relerr(Py, ex["Py"])
        # the backward filter runs n = N -> 0: first n (from the end) where it is off by > 1e-2
        rb = [_norm2(np.atleast_2d(Pb[n]) - ex["Pb"][n]) / _norm2(ex["Pb"][n]) for n in range(Nn + 1)]
        bad = [n for n in range(Nn, -1, -1) if not rb[n] < 1e-2]
        ie["Pb_first_n_off_by_1e-2_from_end"] = bad[0] if bad else None
        ie["Pb_min_eig_float"] = float(min(np.linalg.eigvalsh(np.atleast_2d(x))[0] for x in Pb))
        # spectral radius of the X block of the reversed transition, max over n
        ie["rho_Ab_xx_max"] = float(max(spectral_radius(m[:P]) for m in ex["Mb"]))
    except Exception as e:  # noqa: BLE001
        cf["backward_filter_error"] = f"{type(e).__name__}: {e}"
        # bitwise replicate of the library filter, stopped at its first numpy failure
        with np.errstate(all="ignore"):
            bp = backward_filter_replicate(inst, Nn + 1, mode="v2160", partial=True)
            ie["Pb (backward filter)"] = _relerr(bp["Pb"], ex["Pb"])
        ie["Pb_backward_filter_failed_at_n"] = bp["failed_at"]
        ie["rho_Ab_xx_max"] = float(max(spectral_radius(m[:P]) for m in ex["Mb"]))
    # VAR pivots: operation-for-operation replicate of prg.classes.linear_pks._lifted_pass
    # (same float64 operations in the same order), to locate the pivot whose Cholesky
    # factorisation fails (the library's error message always reports step = k0).
    E = np.zeros((P + q, P))
    E[:P, :] = np.eye(P)
    cR, low = cho_factor(R)
    Rinv = cho_solve((cR, low), np.eye(P + q))
    EtRE, MtRM, Loff = E.T @ Rinv @ E, M.T @ Rinv @ M, -(E.T @ Rinv @ M)
    P0inv = cho_solve(cho_factor(Pf[0]), np.eye(P))
    D = [np.zeros((P, P)) for _ in range(Nn + 1)]
    D[0] += P0inv
    for n in range(1, Nn + 1):
        D[n] += EtRE
        D[n - 1] += MtRM
    Delta = [0.5 * (D[0] + D[0].T)]
    fail_at = None
    try:
        chol = cho_factor(Delta[0])
        for n in range(1, Nn + 1):
            T = Loff @ cho_solve(chol, np.eye(P))
            Dn = D[n] - T @ Loff.T
            Delta.append(0.5 * (Dn + Dn.T))
            chol = cho_factor(Delta[-1])
    except (LinAlgError, ValueError):
        fail_at = len(Delta) - 1
    cf["Delta[VAR]"] = _conds(Delta)
    Dex_piv = [mpf_arr(x) for x in ref.Delta]
    ie["Delta (VAR pivots, float replicate)"] = _relerr(Delta, Dex_piv[:len(Delta)])
    ie["Rinv"] = _relerr([Rinv], [mpf_arr(ref.Rinv)])
    cf["Delta_first_chol_fail"] = fail_at

    # Rounding sensitivity of the pivots: ||Delta_float - Delta_exact||_2 / lambda_min(Delta_exact)
    # (>= 1 means the float pivot carries an error as large as its smallest eigenvalue, so
    # whether a Cholesky factorisation fails is decided by rounding), for the library's
    # operation order and for a variant that only replaces cho_solve(chol, I) by
    # np.linalg.inv(Delta) in T = Loff Delta^{-1}.
    def _pivot_sensitivity(Dl):
        lex = [float(np.linalg.eigvalsh(Dex_piv[n])[0]) for n in range(len(Dl))]
        lfl = [float(np.linalg.eigvalsh(Dl[n])[0]) for n in range(len(Dl))]
        r = [_norm2(Dl[n] - Dex_piv[n]) / lex[n] for n in range(len(Dl))]
        dl = [abs(lfl[n] - lex[n]) / lex[n] for n in range(len(Dl))]
        return {"max": float(np.max(r)), "argmax": int(np.argmax(r)),
                "lmin_rel_dev_max": float(np.max(dl)), "lmin_rel_dev_argmax": int(np.argmax(dl)),
                "min_eig_float_min": float(min(lfl)),
                "min_eig_exact_min": float(min(np.linalg.eigvalsh(x)[0] for x in Dex_piv))}
    ie["Delta_roundoff_ratio (library order)"] = _pivot_sensitivity(Delta)
    Dv = [0.5 * (D[0] + D[0].T)]
    fail_v = None
    try:
        cho_factor(Dv[0])
        for n in range(1, Nn + 1):
            T = Loff @ np.linalg.inv(Dv[-1])
            Dn = D[n] - T @ Loff.T
            Dv.append(0.5 * (Dn + Dn.T))
            cho_factor(Dv[-1])
    except (LinAlgError, ValueError):
        fail_v = len(Dv) - 1
    ie["Delta_roundoff_ratio (inv variant)"] = _pivot_sensitivity(Dv)
    cf["Delta_first_chol_fail_inv_variant"] = fail_v
    if dense_J:
        Jf = np.zeros(((Nn + 1) * P, (Nn + 1) * P))
        for n in range(Nn + 1):
            Jf[n * P:(n + 1) * P, n * P:(n + 1) * P] = D[n]
            if n >= 1:
                Jf[n * P:(n + 1) * P, (n - 1) * P:n * P] = Loff
                Jf[(n - 1) * P:n * P, n * P:(n + 1) * P] = Loff.T
        cf["J[VAR] (dense)"] = float(np.linalg.cond(Jf))
    return ce, cf, kap, ie


def smoother_metrics(inst, err, ref, xref, xref_lo, q):
    """Errors of one library run against the exact reference (one seed).

    Differences are formed against the double-double reference, (est - hi) - lo."""
    res = {"error": err, "events": inst._capture["events"] if inst is not None else [],
           "reg_check": inst._capture.get("reg_check") if inst is not None else None}
    if err is not None:
        return res
    H = inst.history
    Nn = len(H) - 1
    xs = np.array([np.asarray(H[n]["Xkp1_smooth"], dtype=float).ravel() for n in range(Nn + 1)])
    Pret = np.array([np.atleast_2d(H[n]["PXXkp1_smooth"]) for n in range(Nn + 1)])
    raw = inst._capture["raw"]
    Praw = np.array([raw.get(n, Pret[n]) for n in range(Nn + 1)])
    mid = slice(MID[0], MID[1] + 1)          # interior window, away from both ends
    nx = np.linalg.norm(xref, axis=1)
    e_mean = np.linalg.norm((xs - xref) - xref_lo, axis=1)
    res["mean_err"] = float(e_mean.max() / nx.max())
    res["mean_err_argmax"] = int(e_mean.argmax())
    res["mean_err_mid"] = float(e_mean[mid].max() / nx.max())
    den = np.linalg.norm(ref.P, ord=2, axis=(1, 2))
    r_raw = np.linalg.norm((Praw - ref.P) - ref.P_lo, ord=2, axis=(1, 2)) / den
    r_ret = np.linalg.norm((Pret - ref.P) - ref.P_lo, ord=2, axis=(1, 2)) / den
    res["cov_err_raw"] = float(r_raw.max())
    res["cov_err_raw_argmax"] = int(r_raw.argmax())
    res["cov_err_returned"] = float(r_ret.max())
    res["cov_err_raw_mid"] = float(r_raw[mid].max())
    mins = np.linalg.eigvalsh(0.5 * (Praw + np.transpose(Praw, (0, 2, 1))))[:, 0]
    neg = np.where(mins < 0)[0]
    res["indefinite_steps"] = [int(n) for n in neg]
    res["min_eig_raw"] = float(mins.min())
    res["min_eig_raw_argmin"] = int(mins.argmin())
    res["_arrays"] = (xs, Praw)        # for the stock-vs-control difference (not stored)
    # lag-one cross-covariance Cov(x_{n+1}, x_n | y)
    scale = np.sqrt(den[:-1] * den[1:])
    C = None
    if "Mk_smooth" in H[0] and H[0].get("Mk_smooth") is not None:
        C = np.array([np.atleast_2d(H[n]["Mk_smooth"]) for n in range(Nn)])
    elif "Gk_smooth" in H[0] and H[0].get("Gk_smooth") is not None:
        C = np.array([Praw[n + 1] @ np.atleast_2d(H[n]["Gk_smooth"])[:, :P].T for n in range(Nn)])
    if C is not None:
        r_lag = np.linalg.norm((C - ref.C) - ref.C_lo, ord=2, axis=(1, 2)) / scale
        res["lag1_err"] = float(r_lag.max())
        res["lag1_err_argmax"] = int(r_lag.argmax())
        res["lag1_err_mid"] = float(r_lag[mid].max())
    return res


def localise(max_err, mid_err, argmax_n, n_steps=N):
    """Where an error lives. 'small' if max <= LOC_THR; 'persistent' if the interior-window
    error is at least LOC_RATIO * max OR at least LOC_ABS in absolute terms (so that a large
    start-up spike cannot relabel a large interior error); otherwise 'start-up' / 'end' by the
    location of the maximum (argmax n < MID[0] or > MID[1])."""
    if max_err is None:
        return None
    if max_err <= LOC_THR:
        return "small"
    if mid_err is not None and (mid_err >= LOC_RATIO * max_err or mid_err >= LOC_ABS):
        return "persistent"
    if argmax_n is not None and argmax_n < MID[0]:
        return "start-up"
    if argmax_n is not None and argmax_n > MID[1]:
        return "end"
    return "persistent"


def _summarise_runs(runs, ex, ref, model, delta_fail):
    """Per-smoother record over the seeds of one case (full, base-model format)."""
    ok = [r for r in runs if r["error"] is None]
    d = {"n_ok": len(ok), "n_failed": len(runs) - len(ok)}
    fails = {}
    for r in runs:
        if r["error"] is not None:
            key = json.dumps({k: r["error"].get(k) for k in ("type", "matrix", "step")},
                             sort_keys=True)
            if key not in fails:
                e = dict(r["error"])
                e["n_seeds"] = 0
                e["diagnosis"] = diagnose_abort(r["error"], ex, ref, model, delta_fail)
                if e.get("check") and e["check"].get("stats"):
                    e["check"] = dict(e["check"])
                    e["check"]["stats"] = {k: v for k, v in e["check"]["stats"].items()
                                           if k != "matrix"}
                fails[key] = e
            fails[key]["n_seeds"] += 1
    d["failures"] = list(fails.values())
    if ok:
        for key in ("mean_err", "cov_err_raw", "cov_err_returned", "lag1_err",
                    "mean_err_mid", "cov_err_raw_mid", "lag1_err_mid"):
            vals = [r[key] for r in ok if key in r]
            if vals:
                d[key] = {"median": float(np.median(vals)), "max": float(np.max(vals)),
                          "per_seed": vals}
        d["mean_err_argmax_n"] = [r["mean_err_argmax"] for r in ok]
        d["cov_err_raw_argmax_n"] = sorted({r["cov_err_raw_argmax"] for r in ok})
        # localisation of each error: n of the worst seed, interior error (max over seeds)
        loc = {}
        for key, akey in (("mean_err", "mean_err_argmax"), ("cov_err_raw", "cov_err_raw_argmax"),
                          ("lag1_err", "lag1_err_argmax")):
            if key not in d:
                continue
            worst = max(ok, key=lambda r: r[key])
            loc[key] = {"class": localise(d[key]["max"], d[key + "_mid"]["max"], worst[akey]),
                        "argmax_n_worst_seed": worst[akey],
                        "max": d[key]["max"], "mid_max": d[key + "_mid"]["max"]}
        d["localisation"] = loc
        ind = [r for r in ok if r["indefinite_steps"]]
        d["indefinite"] = {
            "n_seeds": len(ind),
            "first_n": min((r["indefinite_steps"][0] for r in ind), default=None),
            "n_steps_max": max((len(r["indefinite_steps"]) for r in ind), default=0),
            "steps_seed0": ok[0]["indefinite_steps"][:20],
            "min_eig_raw": float(min(r["min_eig_raw"] for r in ok)),
            "min_eig_raw_argmin_n": ok[0]["min_eig_raw_argmin"],
        }
    ev = [e for r in runs for e in r["events"]]
    evs = [e for e in ev if e["name"] == "PXXkp1_smooth" and e["kind"] == "regularised"]
    evf = [e for e in ev if e["name"] != "PXXkp1_smooth" and e["kind"] == "regularised"]
    d["regularised_smoothed"] = {
        "n_events_all_seeds": len(evs),
        "n_seeds": len({i for i, r in enumerate(runs) for e in r["events"]
                        if e["name"] == "PXXkp1_smooth" and e["kind"] == "regularised"}),
        "steps_seed0": [e["k"] for e in runs[0]["events"]
                        if e["name"] == "PXXkp1_smooth" and e["kind"] == "regularised"][:20],
        "example": evs[0] if evs else None}
    d["forward_events"] = {
        "n_events_all_seeds": len(evf),
        "names": sorted({e["name"] for e in evf}),
        "steps_seed0": [e["k"] for e in runs[0]["events"]
                        if e["name"] != "PXXkp1_smooth" and e["kind"] == "regularised"][:20],
        "example": evf[0] if evf else None}
    # in-place regularisations: script wrapper vs the library's covariance_regularisations
    rcs = [r.get("reg_check") or {} for r in runs]
    d["regularisation_count_check"] = {
        "wrapper": int(sum(x.get("wrapper") or 0 for x in rcs)),
        "library": (None if any(x.get("library") is None for x in rcs)
                    else int(sum(x["library"] for x in rcs))),
        "agree": all(x.get("wrapper") == x.get("library") for x in rcs)}
    return d


def _stock_vs_control(stock_runs, ctrl_runs, xref_norm, ref):
    """Direct difference between the library 2F/DWY output and the 2F*/DWY* control output,
    per seed where both complete: max_n ||x_lib - x_ctrl||_2 / max_n ||x_ref,n||_2 and
    max_n ||P_lib - P_ctrl||_2 / ||P_ref,n||_2 (raw covariances); bitwise equality."""
    den = np.linalg.norm(ref.P, ord=2, axis=(1, 2))
    md, cd, bit = [], [], []
    for a, b, nx in zip(stock_runs, ctrl_runs, xref_norm):
        if a["error"] is not None or b["error"] is not None:
            continue
        (xa, Pa), (xb, Pb) = a["_arrays"], b["_arrays"]
        md.append(float(np.linalg.norm(xa - xb, axis=1).max() / nx))
        cd.append(float((np.linalg.norm(Pa - Pb, ord=2, axis=(1, 2)) / den).max()))
        bit.append(bool(np.array_equal(xa, xb) and np.array_equal(Pa, Pb)))
    return {"abort_stock": sum(r["error"] is not None for r in stock_runs),
            "abort_control": sum(r["error"] is not None for r in ctrl_runs),
            "n_both_ok": len(md), "n_bitwise": int(sum(bit)),
            "mean_diff_max": max(md) if md else None, "cov_diff_max": max(cd) if cd else None}


def run_case(task):
    """One sweep point: the six library smoothers and the 2F*/DWY* consistency control on
    n_seeds records, against the exact reference. task = (regime, q, index, value, n_seeds,
    gen, model_id, compact); gen None = base model (full record), else a random-family model
    (compact record, no dense cond(J))."""
    regime, q, iv, value, n_seeds, gen, model_id, compact = task
    mp.mp.dps = DPS
    t0 = time.time()
    model = build_model(regime, q, value, gen, model_id)
    param = make_param(model, q)
    ref = LiftedReference(param.A, param.mQ, param.Pz0, q, N)
    ex = exact_intermediates(param.A, param.mQ, param.Pz0, q, N)
    t_ref = time.time() - t0
    cond, kap = None, None
    cond_exact, interm = None, None
    per = {name: [] for name in ALL_NAMES}
    xref_norm = []
    for seed in range(n_seeds):
        X, Y = simulate(model, q, seed)
        xref, xref_lo = ref.mean_dd(Y)
        xref_norm.append(float(np.linalg.norm(xref, axis=1).max()))
        for name in ALL_NAMES:
            cls = SMOOTHERS[VARIANTS.get(name, name)]
            inst, err = run_library(cls, param, X, Y, q, control=name in VARIANTS)
            m = smoother_metrics(inst, err, ref, xref, xref_lo, q)
            if name not in VARIANTS.values() and name not in VARIANTS:
                m.pop("_arrays", None)
            per[name].append(m)
            if (cond is None and name == "RTS" and inst is not None
                    and len(inst.history) == N + 1):   # forward pass complete
                cond_exact, cond, kap, interm = conditioning(inst, model, q, ref, ex,
                                                             dense_J=not compact)
    if cond is None:   # forward pass failed for every seed: still report what we can
        cond, kap, cond_exact, interm = {"note": "forward pass failed; no history"}, {}, {}, {}
    delta_fail = cond.get("Delta_first_chol_fail")
    smo = {name: _summarise_runs(per[name], ex, ref, model, delta_fail) for name in NAMES}
    ctrl = {}
    for star, base in VARIANTS.items():
        ctrl[star] = _summarise_runs(per[star], ex, ref, model, delta_fail)
        ctrl[star]["stock_vs_control"] = _stock_vs_control(per[base], per[star], xref_norm, ref)
    eig_ref = np.linalg.eigvalsh(ref.P)
    out = {
        "regime": regime, "q": q, "index": iv, "value": value, "gen": gen, "model_id": model_id,
        "model": {"rho_A": spectral_radius(model["A"]), "cond_R": float(np.linalg.cond(model["R"])),
                  "cond_Pz0": float(np.linalg.cond(model["Pz0"])),
                  "henrici_rel": henrici(model["A"]),
                  "norm2_Axy": float(np.linalg.norm(model["A"][:P, P:], 2)),
                  "norm2_Ayx": float(np.linalg.norm(model["A"][P:, :P], 2))},
        "reference": {"min_eig_Pref": float(eig_ref[:, 0].min()),
                      "min_eig_Pref_argmin_n": int(eig_ref[:, 0].argmin()),
                      "max_cond_Pref": float((eig_ref[:, -1] / eig_ref[:, 0]).max()),
                      "max_norm_xref_per_seed": xref_norm},
        "cond_exact": cond_exact, "cond_float": cond, "kappa": kap,
        "intermediate_err": interm, "smoothers": smo, "consistency_control": ctrl,
        "runtime_s": {"reference_cov": t_ref, "total": time.time() - t0},
    }
    if compact:
        return compact_case(out)
    out["model"].update({"A": model["A"].tolist(), "R": model["R"].tolist(),
                         "Pz0_smoother": model["Pz0"].tolist(),
                         "Pz0_data": model["Pz0_data"].tolist()})
    return out


COND_KEYS = ["P_pred[RTS]", "S[BF,MBF]", "Sigma[2F,DWY]", "Pb[2F]", "Pb_pred[DWY]", "Pf[2F]",
             "Py[2F]", "Delta[VAR]"]


def _compact_smoother(d):
    """Compact per-smoother record (random families)."""
    c = {"n_ok": d["n_ok"], "n_failed": d["n_failed"]}
    if d["failures"]:
        c["failures"] = []
        for f in d["failures"]:
            dg = f.get("diagnosis", {})
            c["failures"].append({
                "type": f["type"], "matrix": f["matrix"], "step": f["step"],
                "n_seeds": f["n_seeds"], "check": dg.get("cause", {}).get("check"),
                "failed_checks": dg.get("cause", {}).get("failed_checks"),
                "tolerance_artefact": dg.get("cause", {}).get("tolerance_artefact"),
                "diagnosis_code": dg.get("diagnosis_code"), "where": f.get("where"),
                "exact_cond": dg.get("exact_cond"),
                "exact_min_eig": dg.get("exact_min_eig"),
                "float_rel_err": dg.get("float_rel_err"),
                "failing_pivot_index_replicate": dg.get("failing_pivot_index_replicate")})
    if "cov_err_raw" in d:
        loc = d["localisation"]
        c.update({"mean": d["mean_err"]["per_seed"], "mean_mid": d["mean_err_mid"]["per_seed"],
                  "cov": d["cov_err_raw"]["max"], "cov_mid": d["cov_err_raw_mid"]["max"],
                  "loc_cov": loc["cov_err_raw"]["class"], "loc_mean": loc["mean_err"]["class"],
                  "cov_argmax": loc["cov_err_raw"]["argmax_n_worst_seed"],
                  "indef": d["indefinite"]["n_seeds"]})
        if d["cov_err_returned"]["max"] != d["cov_err_raw"]["max"]:
            c["cov_ret"] = d["cov_err_returned"]["max"]
        if d["indefinite"]["n_seeds"]:
            c["min_eig"] = d["indefinite"]["min_eig_raw"]
    c["reg"] = d["regularised_smoothed"]["n_events_all_seeds"]
    c["fwd_reg"] = d["forward_events"]["n_events_all_seeds"]
    c["reg_lib_agree"] = d["regularisation_count_check"]["agree"]
    if "stock_vs_control" in d:
        c["stock_vs_control"] = d["stock_vs_control"]
    return c


def compact_case(out):
    """Compact record of a random-family case (the full per-seed record of the base model
    is kept in "cases"; here only what the aggregated tables and the figure need)."""
    smo = {name: _compact_smoother(d) for name, d in out["smoothers"].items()}
    ctrl = {name: _compact_smoother(d) for name, d in out["consistency_control"].items()}
    ce, ie, cf = out["cond_exact"] or {}, out["intermediate_err"] or {}, out["cond_float"] or {}
    cond = {k: [ce[k]["max"], ce[k]["median"]] for k in COND_KEYS if ce.get(k)}
    cond["R"] = ce.get("R[VAR]")
    bwd = {"Pb_relerr_float_stock": (ie.get("Pb (backward filter)") or {}).get("max"),
           "rho_Ab_xx_max": ie.get("rho_Ab_xx_max"),
           "backward_filter_error": cf.get("backward_filter_error"),
           "backward_filter_failed_at_n": ie.get("Pb_backward_filter_failed_at_n")}
    var = {"Delta_first_chol_fail": cf.get("Delta_first_chol_fail"),
           "Delta_roundoff_ratio": (ie.get("Delta_roundoff_ratio (library order)") or {}).get("max"),
           "Delta_lmin_rel_dev": (ie.get("Delta_roundoff_ratio (library order)") or {}).get(
               "lmin_rel_dev_max")}
    return {"regime": out["regime"], "q": out["q"], "index": out["index"], "value": out["value"],
            "gen": out["gen"], "model_id": out["model_id"], "model": out["model"],
            "bwd": bwd, "var_pivots": var, "cond_exact": cond,
            "smoothers": smo, "consistency_control": ctrl, "runtime_s": out["runtime_s"]["total"]}


def crosscheck_case(task):
    """Reference cross-check at one (regime, q, value), seed 0."""
    regime, q, value, gen, model_id = task
    mp.mp.dps = DPS
    t0 = time.time()
    model = build_model(regime, q, value, gen, model_id)
    param = make_param(model, q)
    A, R, Pz0 = param.A, param.mQ, param.Pz0
    X, Y = simulate(model, q, 0)
    ref = LiftedReference(A, R, Pz0, q, N)
    x60 = ref.mean(Y)
    # (i) KF + RTS covariance form, full record
    xk, Pk, Ck = mp_kf_rts(A, R, Pz0, Y, q)
    d_kf = mp_rel_diff(xk, x60, Pk, ref.Pmp, Ck, ref.Cmp[:-1])
    # (iii) lifted solver at 80 digits
    ref80 = LiftedReference(A, R, Pz0, q, N, dps=80)
    with mp.workdps(80):
        x80 = ref80.mean(Y)
        d_80 = mp_rel_diff(x60, x80, ref.Pmp, ref80.Pmp, ref.Cmp[:-1], ref80.Cmp[:-1])
    # (ii) dense conditioning on a short record (same model, first N_DENSE+1 samples)
    Ys = Y[:N_DENSE + 1]
    refs = LiftedReference(A, R, Pz0, q, N_DENSE)
    xd, Pd, Cd = mp_dense_conditioning(A, R, Pz0, Ys, q)
    d_dense = mp_rel_diff(refs.mean(Ys), xd, refs.Pmp, Pd, refs.Cmp[:-1], Cd)
    return {"regime": regime, "q": q, "value": value, "gen": gen, "model_id": model_id,
            "kf_rts_N400": d_kf, "lifted_80digits_N400": d_80,
            f"dense_conditioning_N{N_DENSE}": d_dense, "runtime_s": time.time() - t0}


def scale_probe(task):
    """Control U, one (q, k): base model with (R, Pz0) scaled by alpha = 2^-k and the
    record by 2^-k/2. In exact arithmetic the smoothed means scale by 2^-k/2 and the
    covariances by alpha. In float64 the model scaling is exact for every k, the record
    scaling only for even k (2^-k/2 is not a power of two for odd k), so for even k any
    abort or deviation is caused by an absolute (scale-dependent) threshold in the
    library. Returns the rescaled outputs; deviations are formed in main()."""
    q, k = task
    base_model = build_model("S", q, 1.0)
    X0, Y0 = simulate(base_model, q, 0)
    a, r = 2.0 ** (-k), 2.0 ** (-k / 2)
    model = {"A": base_model["A"], "R": base_model["R"] * a,
             "Pz0": base_model["Pz0"] * a, "Pz0_data": base_model["Pz0_data"] * a}
    row = {"k": k, "alpha": a, "record_scaling_exact": k % 2 == 0, "smoothers": {}}
    rep_s22 = InvertibleMatrix(model["Pz0"][P:, P:]).check()
    row["Sigma22_failed_checks"] = [c.name for c in rep_s22.checks if c.status == Status.FAIL]
    S22 = model["Pz0"][P:, P:]
    row["Sigma22_abs_det"] = float(abs(np.linalg.det(S22)))
    row["Sigma22_normalised_det"] = float(abs(np.linalg.det(S22))
                                          / np.prod(np.linalg.norm(S22, axis=1)))
    outs = {}
    try:
        param = make_param(model, q)
    except Exception as e:  # noqa: BLE001
        row["param_error"] = f"{type(e).__name__}: {e}"
        return {"q": q, "row": row, "outs": outs}
    for name, cls in SMOOTHERS.items():
        inst, err = run_library(cls, param, X0 * r, Y0 * r, q)
        if err is not None:
            row["smoothers"][name] = {"error": err}
            continue
        H = inst.history
        xs = np.array([np.asarray(h["Xkp1_smooth"], dtype=float).ravel() for h in H]) / r
        Ps = np.array([np.atleast_2d(h["PXXkp1_smooth"]) for h in H]) / a
        outs[name] = (xs, Ps)
        row["smoothers"][name] = {"error": None, "n_events": len(inst._capture["events"])}
    return {"q": q, "row": row, "outs": outs}


def assemble_scale_control(parts):
    """Deviations of every k from k = 0, threshold k* (first k with an abort)."""
    blocks = []
    for q in QS:
        mine = sorted([p for p in parts if p["q"] == q], key=lambda p: p["row"]["k"])
        base = next(p["outs"] for p in mine if p["row"]["k"] == 0)
        rows = []
        for p in mine:
            row = p["row"]
            for name, (xs, Ps) in p["outs"].items():
                if name in base:
                    bx, bP = base[name]
                    row["smoothers"][name]["max_rel_dev_mean"] = float(np.abs(xs - bx).max() / np.abs(bx).max())
                    row["smoothers"][name]["max_rel_dev_cov"] = float(np.abs(Ps - bP).max() / np.abs(bP).max())
            row["any_abort"] = ("param_error" in row) or any(
                (d.get("error") is not None) for d in row["smoothers"].values())
            rows.append(row)
        ab = [r for r in rows if r["any_abort"]]
        ok = [r for r in rows if not r["any_abort"]]

        def _maxdev(sel):
            v = [max(d.get("max_rel_dev_mean", 0.0), d.get("max_rel_dev_cov", 0.0))
                 for r in sel for d in r["smoothers"].values()]
            return float(max(v)) if v else None
        k_star = ab[0]["k"] if ab else None
        blocks.append({
            "q": q, "rows": rows, "k_star_first_abort": k_star,
            "abs_det_Sigma22_at_k_star": ab[0]["Sigma22_abs_det"] if ab else None,
            "abs_det_Sigma22_at_k_star_minus_1": next(
                (r["Sigma22_abs_det"] for r in rows if k_star is not None and r["k"] == k_star - 1), None),
            "all_k_after_k_star_abort": bool(ab) and all(r["any_abort"] for r in rows if r["k"] >= k_star),
            "max_dev_even_k_running": _maxdev([r for r in ok if r["k"] % 2 == 0]),
            "max_dev_odd_k_running": _maxdev([r for r in ok if r["k"] % 2 == 1]),
            "failure_at_k_star": (ab[0].get("param_error") or next(
                (d["error"] for d in ab[0]["smoothers"].values() if d.get("error")), None)) if ab else None,
            "failed_checks_at_k_star": ab[0]["Sigma22_failed_checks"] if ab else None})
    return blocks


# ----------------------------------------------------------------------------------
# Random families: aggregation
# ----------------------------------------------------------------------------------
LOC_CLASSES = ("small", "start-up", "end", "persistent")


def _q3(v):
    """median / 90th percentile (numpy linear interpolation) / max of a list, or None."""
    if not v:
        return None
    v = np.asarray(v, dtype=float)
    return {"median": float(np.median(v)), "p90": float(np.percentile(v, 90)),
            "max": float(v.max())}


def abort_key(f):
    chk = "/".join(f.get("failed_checks") or []) or "?"
    return f"{f['type']} {_where_or_matrix(f)} [{chk}] {f.get('diagnosis_code')}"


def _num(x):
    """JSON number back to float ('inf' strings and None handled)."""
    if x is None:
        return None
    return float(x)


def _cons_pairs(c):
    """(library smoother record, control record) for 2F and DWY of one case (compact or full)."""
    out = []
    for star, base in VARIANTS.items():
        out.append((base, c["smoothers"][base], c["consistency_control"][star]))
    return out


def _errs(d):
    """(cov, mean) error of a compact or full smoother record (None when it aborted)."""
    if "cov" in d:
        return d["cov"], max(d["mean"])
    if "cov_err_raw" in d:
        return d["cov_err_raw"]["max"], d["mean_err"]["max"]
    return None, None


def consistency_cell(at):
    """Library 2F/DWY vs the 2F*/DWY* control over the cases in `at`, per smoother:
    abort agreement, direct output differences (max over seeds) and error ratios."""
    out = {}
    for base in VARIANTS.values():
        rows = [(d, ct) for c in at for (b, d, ct) in _cons_pairs(c) if b == base]
        sv = [ct["stock_vs_control"] for _, ct in rows]
        both = [(d, ct, x) for (d, ct), x in zip(rows, sv) if d["n_failed"] == 0 and ct["n_failed"] == 0]
        cdiff = [x["cov_diff_max"] for _, _, x in both]
        mdiff = [x["mean_diff_max"] for _, _, x in both]
        ratio, lib_better, lib_worse, rel = [], 0, 0, []
        for d, ct, x in both:
            (c1, m1), (c2, m2) = _errs(d), _errs(ct)
            ratio.append(max(c1 / max(c2, 1e-300), c2 / max(c1, 1e-300),
                             m1 / max(m2, 1e-300), m2 / max(m1, 1e-300)))
            lib_better += max(c2 / max(c1, 1e-300), m2 / max(m1, 1e-300)) > CONS_RATIO
            lib_worse += max(c1 / max(c2, 1e-300), m1 / max(m2, 1e-300)) > CONS_RATIO
            if x["cov_diff_max"] > CONS_ROUNDOFF:
                rel.append(x["cov_diff_max"] / max(c1, c2, 1e-300))
        out[base] = {
            "n_cases": len(rows),
            "both_complete": len(both),
            "both_abort": sum(d["n_failed"] > 0 and ct["n_failed"] > 0 for d, ct in rows),
            "library_only_aborts": sum(d["n_failed"] > 0 and ct["n_failed"] == 0 for d, ct in rows),
            "control_only_aborts": sum(d["n_failed"] == 0 and ct["n_failed"] > 0 for d, ct in rows),
            "same_abort_seed_count": sum(x["abort_stock"] == x["abort_control"] for x in sv),
            "n_bitwise_all_seeds": sum(x["n_bitwise"] == x["n_both_ok"] and x["n_both_ok"] > 0
                                       for _, _, x in both),
            "cov_diff": _q3(cdiff), "mean_diff": _q3(mdiff),
            "n_cov_diff_gt_roundoff": sum(v > CONS_ROUNDOFF for v in cdiff),
            "n_mean_diff_gt_roundoff": sum(v > CONS_ROUNDOFF for v in mdiff),
            "n_error_ratio_gt": sum(r > CONS_RATIO for r in ratio),
            "n_library_more_accurate_gt": int(lib_better), "n_control_more_accurate_gt": int(lib_worse),
            # cov difference / max(library, control) cov error, where the difference > CONS_ROUNDOFF
            # (at most 2 by the triangle inequality; ~1 = the difference is as large as the errors)
            "cov_diff_over_max_error": _q3(rel)}
    return out


def consistency_outliers(cases):
    """Cases where the library 2F/DWY and the 2F*/DWY* control disagree: abort status, an
    output difference above CONS_ROUNDOFF, or errors (vs the reference) differing by more than
    CONS_RATIO."""
    out = []
    for c in cases:
        for base, d, ct in _cons_pairs(c):
            x = ct["stock_vs_control"]
            ab = (d["n_failed"] > 0) != (ct["n_failed"] > 0)
            (c1, m1), (c2, m2) = _errs(d), _errs(ct)
            diff = ((x.get("cov_diff_max") or 0) > CONS_ROUNDOFF
                    or (x.get("mean_diff_max") or 0) > CONS_ROUNDOFF)
            rat = (c1 is not None and c2 is not None and
                   max(c1 / max(c2, 1e-300), c2 / max(c1, 1e-300),
                       m1 / max(m2, 1e-300), m2 / max(m1, 1e-300)) > CONS_RATIO)
            if ab or diff or rat:
                out.append({"regime": c["regime"], "gen": c.get("gen"), "q": c["q"],
                            "model_id": c.get("model_id"), "value": c["value"], "smoother": base,
                            "abort_library": d["n_failed"], "abort_control": ct["n_failed"],
                            "cov_library": c1, "cov_control": c2, "mean_library": m1,
                            "mean_control": m2, "cov_diff_max": x.get("cov_diff_max"),
                            "mean_diff_max": x.get("mean_diff_max"),
                            "abort_status_differs": ab, "error_ratio_gt": bool(rat)})
    return out


def consistency_summary(results):
    """Per regime: random families pooled, and the base model."""
    out = {}
    for reg in REGIMES:
        rc = [c for c in results["random_cases"] if c["regime"] == reg]
        bc = [c for c in results["cases"] if c["regime"] == reg]
        out[reg] = {"random_families": consistency_cell(rc), "base_model": consistency_cell(bc)}
    allx = consistency_outliers(results["cases"]) + consistency_outliers(results["random_cases"])
    out["n_beyond_roundoff_or_disagreeing"] = len(allx)
    out["disagreements"] = [x for x in allx if x["abort_status_differs"] or x["error_ratio_gt"]]
    out["largest_differences"] = sorted([x for x in allx if not x["abort_status_differs"]],
                                        key=lambda x: -(x["cov_diff_max"] or 0))[:10]
    return out


def summarise_cell(at, name):
    """One (regime, generator, q, value, smoother) cell over the models in `at`.

    Errors: per model, max over its seeds that ran (covariances are seed-independent);
    statistics over the models on which the smoother ran (aborting models excluded)."""
    ds = [c["smoothers"][name] for c in at]
    ok = [d for d in ds if "cov" in d]
    cell = {"n_models": len(ds), "n_abort": sum(d["n_failed"] > 0 for d in ds),
            "n_ran": len(ok), "n_indefinite": sum(d.get("indef", 0) > 0 for d in ok),
            "n_regularised": sum(d["reg"] > 0 for d in ds),
            "n_forward_regularised": sum(d["fwd_reg"] > 0 for d in ds)}
    for key, f in (("cov", lambda d: d["cov"]), ("cov_mid", lambda d: d["cov_mid"]),
                   ("mean", lambda d: max(d["mean"])), ("mean_mid", lambda d: max(d["mean_mid"]))):
        cell[key] = _q3([f(d) for d in ok])
    for key in ("loc_cov", "loc_mean"):
        cell[key] = {k: sum(d[key] == k for d in ok) for k in LOC_CLASSES}
    causes, steps = {}, {}
    n_art = 0
    for d in ds:
        fl = d.get("failures", [])
        if fl and any(f.get("tolerance_artefact") for f in fl):
            n_art += 1
        for f in fl:
            k = abort_key(f)
            causes[k] = causes.get(k, 0) + 1
            steps.setdefault(k, []).append(f["step"])
    cell["abort_causes"] = causes
    cell["abort_steps"] = {k: [min(v), max(v)] for k, v in steps.items() if None not in v}
    cell["n_abort_tolerance_artefact"] = n_art
    return cell


def summarise_random(rc):
    """Nested summary[regime][gen|'all'][q|'all'][value index] -> per smoother cells (the six
    library smoothers) and the consistency of 2F/DWY with the 2F*/DWY* control."""
    out = {}
    for reg in REGIMES:
        out[reg] = {}
        for gen in list(GENS) + ["all"]:
            out[reg][gen] = {}
            for q in list(QS) + ["all"]:
                sel = [c for c in rc if c["regime"] == reg and (gen == "all" or c["gen"] == gen)
                       and (q == "all" or c["q"] == q)]
                if not sel:
                    continue
                blk = {}
                for iv, v in enumerate(RF_VALUES[reg]):
                    at = [c for c in sel if c["index"] == iv]
                    if not at:
                        continue
                    row = {"value": v, "n_models": len(at)}
                    for name in NAMES:
                        row[name] = summarise_cell(at, name)
                    row["consistency"] = consistency_cell(at)
                    blk[str(iv)] = row
                out[reg][gen][str(q)] = blk
    return out


# ----------------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------------
def _f(x, w=8):
    if x is None:
        return f"{'-':>{w}}"
    if isinstance(x, str):
        return f"{x:>{w}}"
    return f"{x:>{w}.1e}"


def _flags(d):
    return ("i" if d["indefinite"]["n_seeds"] else "") + \
        ("r" if d["regularised_smoothed"]["n_events_all_seeds"] else "") + \
        ("f" if d["forward_events"]["n_events_all_seeds"] else "")


LOC_TAG = {None: " ", "small": " ", "start-up": "s", "end": "e", "persistent": "P"}


def _loc(d, key):
    return LOC_TAG[d.get("localisation", {}).get(key, {}).get("class")]


def print_summary(results):
    cases = results["cases"]
    W = 20
    print(f"Localisation tags: s = error > {LOC_THR:g} confined to start-up (interior-window "
          f"[{MID[0]},{MID[1]}] error < {LOC_RATIO:g} x max and < {LOC_ABS:g}, argmax n < {MID[0]}); "
          f"e = same at the end (argmax n > {MID[1]}); P = persistent (interior error >= "
          f"{LOC_RATIO:g} x max or >= {LOC_ABS:g}).")
    for reg, spec in REGIMES.items():
        for q in QS:
            rows = sorted([c for c in cases if c["regime"] == reg and c["q"] == q],
                          key=lambda c: c["index"])
            if not rows:
                continue
            pn = spec["param"]
            print(f"\n=== Regime {reg} ({spec['title']}), (p,q)=({P},{q}); sweep {pn} ===")
            print("  [a] mean error max_n||dx_n||/max_n||x_ref,n||: median / max over seeds"
                  "  (F = some seeds aborted; s/e/P localisation)")
            print(f"  {pn:>7} " + "".join(f"{nm:>{W}}" for nm in NAMES))
            for c in rows:
                line = f"  {c['value']:>7.0e} "
                for nm in NAMES:
                    d = c["smoothers"][nm]
                    if "mean_err" in d:
                        s = f"{d['mean_err']['median']:.1e}/{d['mean_err']['max']:.1e}"
                        line += f"{s + _loc(d, 'mean_err') + ('F' if d['n_failed'] else ' '):>{W}}"
                    else:
                        line += f"{'ABORT':>{W}}"
                print(line)
            print(f"  [b] same, interior window n in [{MID[0]},{MID[1]}] (max over seeds)")
            print(f"  {pn:>7} " + "".join(f"{nm:>10}" for nm in NAMES))
            for c in rows:
                print(f"  {c['value']:>7.0e} " + "".join(
                    _f(c["smoothers"][nm].get("mean_err_mid", {}).get("max"), 10) for nm in NAMES))
            print("  [c] covariance error max_n||dP_n||_2/||P_ref,n||_2, raw output (returned output if"
                  " it differs); flags i=raw indefinite, r=library regularised a smoothed covariance,"
                  " f=forward-pass regularisation")
            print(f"  {pn:>7} " + "".join(f"{nm:>{W}}" for nm in NAMES))
            for c in rows:
                line = f"  {c['value']:>7.0e} "
                for nm in NAMES:
                    d = c["smoothers"][nm]
                    if "cov_err_raw" not in d:
                        line += f"{'ABORT':>{W}}"
                        continue
                    a, b = d["cov_err_raw"]["max"], d["cov_err_returned"]["max"]
                    s = f"{a:.1e}" + (f"({b:.0e})" if not np.isclose(a, b, rtol=1e-3, atol=0) else "")
                    line += f"{s + _loc(d, 'cov_err_raw') + _flags(d):>{W}}"
                print(line)
            print(f"  [d] covariance error, interior window n in [{MID[0]},{MID[1]}] (raw);"
                  "  lag-one Cov(x_n+1,x_n|y) error (RTS, VAR): all n / interior window")
            print(f"  {pn:>7} " + "".join(f"{nm:>10}" for nm in NAMES)
                  + f"{'lag1 RTS':>11}{'(mid)':>9}{'lag1 VAR':>11}{'(mid)':>9}")
            for c in rows:
                sr, sv = c["smoothers"]["RTS"], c["smoothers"]["VAR"]
                print(f"  {c['value']:>7.0e} " + "".join(
                    _f(c["smoothers"][nm].get("cov_err_raw_mid", {}).get("max"), 10) for nm in NAMES)
                    + _f(sr.get("lag1_err", {}).get("max"), 10) + _loc(sr, "lag1_err")
                    + _f(sr.get("lag1_err_mid", {}).get("max"), 9)
                    + _f(sv.get("lag1_err", {}).get("max"), 10) + _loc(sv, "lag1_err")
                    + _f(sv.get("lag1_err_mid", {}).get("max"), 9))
            print("  [e] indefinite raw smoothed covariances: seeds affected / first n / #steps (seed max)"
                  " / most negative eigenvalue")
            for c in rows:
                for nm in NAMES:
                    ind = c["smoothers"][nm].get("indefinite")
                    if ind and ind["n_seeds"]:
                        rg = c["smoothers"][nm]["regularised_smoothed"]
                        print(f"  {c['value']:>7.0e} {nm:>4}: {ind['n_seeds']}/{c['smoothers'][nm]['n_ok']} seeds,"
                              f" first n={ind['first_n']}, {ind['n_steps_max']} steps, min eig "
                              f"{ind['min_eig_raw']:.2e}; library regularised {rg['n_events_all_seeds']}"
                              f" smoothed covariances (seed 0 steps {rg['steps_seed0'][:6]})")
            keys = ["P_pred[RTS]", "S[BF,MBF]", "Sigma[2F,DWY]", "Pb[2F]", "Pb_pred[DWY]",
                    "Pf[2F]", "Py[2F]", "Delta[VAR]"]
            for tag, ck in (("EXACT matrices (60-digit, rounded)", "cond_exact"),
                            ("float64 matrices formed by the library", "cond_float")):
                print(f"  [f] condition numbers of the {tag}: max over n / median over n (argmax n)")
                print(f"  {pn:>7} " + "".join(f"{k:>24}" for k in keys)
                      + f"{'J[VAR]':>9}{'R':>9}{'P00':>9}")
                for c in rows:
                    cd = c[ck]
                    line = f"  {c['value']:>7.0e} "
                    for k in keys:
                        st = cd.get(k)
                        if not st or st.get("max") is None:
                            line += f"{'-':>24}"
                        else:
                            line += f"{st['max']:>9.1e}/{st['median']:.1e}({st['argmax']:>3d})"
                    line += _f(cd.get("J[VAR] (dense)"), 9) + _f(c["cond_exact"].get("R[VAR]"), 9) + \
                        _f(c["cond_exact"].get("P00[VAR]"), 9)
                    if cd.get("Delta_first_chol_fail") is not None:
                        line += f"  VAR pivot Cholesky fails at n={cd['Delta_first_chol_fail']}"
                    if cd.get("backward_filter_error"):
                        line += "  backward filter: " + cd["backward_filter_error"][:60]
                    print(line)
            ikeys = ["Pf (forward filter)", "P_pred (forward filter)", "Sigma (prior recursion)",
                     "Pb (backward filter)", "Pb_pred (backward filter)", "Py",
                     "Delta (VAR pivots, float replicate)", "Rinv"]
            print("  [h] relative error of the library's float64 intermediates vs exact, max over n (argmax n)")
            print(f"  {pn:>7} " + "".join(f"{k.split(' (')[0]:>17}" for k in ikeys)
                  + "  Pb first off (>1e-2) from end")
            for c in rows:
                ie = c["intermediate_err"]
                line = f"  {c['value']:>7.0e} "
                for k in ikeys:
                    st = ie.get(k)
                    line += f"{'-':>17}" if not st else f"{st['max']:>10.1e}({st['argmax']:>3d})"
                line += f"  {ie.get('Pb_first_n_off_by_1e-2_from_end')}"
                if ie.get("rho_Ab_xx_max") is not None:
                    line += f"  [max_n rho(A^b_xx) {ie['rho_Ab_xx_max']:.3f}]"
                print(line)
            print("  [h'] VAR pivots, library order | np.linalg.inv variant: max_n ||dDelta_n||/"
                  "lambda_min(Delta_n exact) ; max_n |dlambda_min|/lambda_min (n) ; first failing "
                  "pivot (Cholesky)")
            for c in rows:
                ie, cf = c["intermediate_err"], c["cond_float"]
                a = ie.get("Delta_roundoff_ratio (library order)")
                b = ie.get("Delta_roundoff_ratio (inv variant)")
                if not a:
                    continue
                print(f"  {c['value']:>7.0e}  {a['max']:.1e} ; {a['lmin_rel_dev_max']:.1e} "
                      f"(n={a['lmin_rel_dev_argmax']}) ; fail@{cf.get('Delta_first_chol_fail')} | "
                      f"{b['max']:.1e} ; {b['lmin_rel_dev_max']:.1e} (n={b['lmin_rel_dev_argmax']}) ; "
                      f"fail@{cf.get('Delta_first_chol_fail_inv_variant')}   min_n lambda_min: exact "
                      f"{a['min_eig_exact_min']:.3g}, float lib {a['min_eig_float_min']:.3g}, "
                      f"float inv {b['min_eig_float_min']:.3g}")
            print("  [g] cancellation factor kappa = ||subtracted term||/||result||: max over n / "
                  "median over n (argmax n)")
            print(f"  {pn:>7} " + "".join(f"{nm:>22}" for nm in NAMES))
            for c in rows:
                line = f"  {c['value']:>7.0e} "
                for nm in NAMES:
                    k = c["kappa"].get(nm) or {}
                    line += (f"{'-':>22}" if k.get("max") is None else
                             f"{k['max']:>10.1e}/{k['median']:.1e}({k['argmax']:>3d})")
                print(line)
            print("  [i] 2F/DWY vs the 2F*/DWY* consistency control: aborting seeds library/control,"
                  " max over seeds of the direct output difference (cov, mean), seeds bitwise equal")
            for c in rows:
                parts = []
                for star, base in VARIANTS.items():
                    x = c["consistency_control"][star]["stock_vs_control"]
                    parts.append(f"{base}: abort {x['abort_stock']}/{x['abort_control']}, diff cov "
                                 f"{_f(x['cov_diff_max'], 7)} mean {_f(x['mean_diff_max'], 7)}, "
                                 f"bitwise {x['n_bitwise']}/{x['n_both_ok']}")
                print(f"  {c['value']:>7.0e}  " + " | ".join(parts))
            for c in rows:
                for nm in NAMES:
                    d = c["smoothers"][nm]
                    if d["n_failed"]:
                        for f in d["failures"]:
                            dg = f.get("diagnosis", {})
                            cs = dg.get("cause", {})
                            ex_txt = (f"; exact cond {dg['exact_cond']:.2g}, exact min eig "
                                      f"{dg['exact_min_eig']:.2g}" if dg.get("exact_cond") is not None
                                      else "")
                            fe = (f", float rel. err {dg['float_rel_err']:.2g}"
                                  if dg.get("float_rel_err") is not None else "")
                            piv = (f", failing pivot n={dg['failing_pivot_index_replicate']}"
                                   if dg.get("failing_pivot_index_replicate") is not None else "")
                            print(f"  ABORT {pn}={c['value']:.0e} {nm}: {f['n_seeds']}/"
                                  f"{d['n_failed'] + d['n_ok']} seeds: {f['type']} at step {f['step']}"
                                  f" ({f['matrix']}){piv} | check: {cs.get('check')} | tolerance "
                                  f"artefact: {cs.get('tolerance_artefact')} | {dg.get('diagnosis_code')}"
                                  f"{ex_txt}{fe}")
                    if d["forward_events"]["n_events_all_seeds"]:
                        print(f"  FORWARD-PASS REGULARISATION {pn}={c['value']:.0e} {nm}: "
                              f"{d['forward_events']['names']} steps(seed0) "
                              f"{d['forward_events']['steps_seed0'][:8]}")
    print_localised(cases)
    print("\n=== Reference cross-check (seed 0; base: most extreme sweep value; random: worst "
          "model at the most extreme value) ===")
    for c in results["crosscheck"] + results.get("crosscheck_random", []):
        a, b, d = c["kf_rts_N400"], c["lifted_80digits_N400"], c[f"dense_conditioning_N{N_DENSE}"]
        tag = c["regime"] + (f" {c['gen']}#{c['model_id']}" if c.get("model_id") is not None else " base")
        print(f"  {tag} q={c['q']} value={c['value']:.0e}:  KF+RTS(60d) vs lifted(60d): "
              f"mean {a['mean']:.1e} cov {a['cov']:.1e} lag1 {a['lag1']:.1e} | lifted 60d vs 80d: "
              f"mean {b['mean']:.1e} cov {b['cov']:.1e} lag1 {b['lag1']:.1e} | dense N={N_DENSE}: "
              f"mean {d['mean']:.1e} cov {d['cov']:.1e} lag1 {d['lag1']:.1e}")
    print("\n=== Control U: (R, Pz0) scaled by 2^-k, record by 2^-k/2 (base model, seed 0), "
          f"k = {SCALE_EXPS[0]} .. {SCALE_EXPS[-1]} ===")
    for blk in results.get("scale_control", []):
        if blk["k_star_first_abort"] is None:
            print(f"  (p,q)=({P},{blk['q']}): no abort for any k ({len(blk['rows'])} values of k)")
        else:
            print(f"  (p,q)=({P},{blk['q']}): first abort at k* = {blk['k_star_first_abort']}; "
                  f"every k >= k* aborts: {blk['all_k_after_k_star_abort']}; failure "
                  f"{(blk['failure_at_k_star'] or {}).get('type')} "
                  f"{(blk['failure_at_k_star'] or {}).get('matrix')} at step "
                  f"{(blk['failure_at_k_star'] or {}).get('step')}; failed checks on Sigma22 "
                  f"{blk['failed_checks_at_k_star']}")
        print(f"    max relative deviation from k=0 over the runs that complete: even k "
              f"{blk['max_dev_even_k_running']:.1e}, odd k {blk['max_dev_odd_k_running']:.1e} "
              "(record scaling 2^-k/2 is exact only for even k)")


def print_localised(cases):
    """Every error above LOC_THR in the four regimes, with where it lives."""
    print(f"\n=== Errors > {LOC_THR:g}: localisation (argmax n of the worst seed; interior "
          f"window [{MID[0]},{MID[1]}] max) ===")
    for reg in REGIMES:
        for q in QS:
            rows = sorted([c for c in cases if c["regime"] == reg and c["q"] == q],
                          key=lambda c: c["index"])
            for nm in NAMES:
                for key, lab in (("mean_err", "mean"), ("cov_err_raw", "cov"), ("lag1_err", "lag1")):
                    items = []
                    for c in rows:
                        lo = c["smoothers"][nm].get("localisation", {}).get(key)
                        if lo and lo["class"] not in (None, "small"):
                            items.append(f"{c['value']:.0e}: {lo['max']:.1e}@n={lo['argmax_n_worst_seed']}"
                                         f" (mid {lo['mid_max']:.1e}) {lo['class']}")
                    if items:
                        print(f"  {reg} ({P},{q}) {nm:>3} {lab:>4}: " + "; ".join(items))


# ----------------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------------
STYLE = {  # Okabe-Ito colours, explicit on every call; marker FILL encodes localisation
    "RTS": {"color": "#000000", "marker": "o", "ls": "-", "ms": 3.6},
    "BF": {"color": "#E69F00", "marker": "s", "ls": "--", "ms": 3.4},
    "MBF": {"color": "#0072B2", "marker": "^", "ls": "-", "ms": 3.8},
    "2F": {"color": "#009E73", "marker": "D", "ls": "-", "ms": 3.4},
    "DWY": {"color": "#D55E00", "marker": "v", "ls": "--", "ms": 3.8},
    "VAR": {"color": "#CC79A7", "marker": "h", "ls": ":", "ms": 4.0},
}
FLOOR, YTOP = 1e-17, 1e3
FIG_XLABEL = {"D": r"(D) diffuse prior: prior scale $s$",
              "S": r"(S) state-noise starvation: scale $\varepsilon$",
              "C": r"(C) correlated noise: $1-c$",
              "M": r"(M) slow mixing: $1-\rho(\mathbf{A})$"}


def _pow10_label(v, _pos=None):
    """'1', '1e4', '1e−16': plain text at the tick font size (no superscripts)."""
    if v <= 0:
        return ""
    e = int(np.round(np.log10(v)))
    if not np.isclose(v, 10.0 ** e, rtol=1e-6):
        return ""
    return "1" if e == 0 else ("1e" + str(e)).replace("-", "−")


def make_figure_base(results, q_fig, path):
    import matplotlib as mpl
    mpl.rcdefaults()   # drop prg's plot_settings (backend is not reset)
    mpl.use("Agg")
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    avail = {f.name for f in fm.fontManager.ttflist}
    family = next((f for f in ("Arial", "Helvetica", "Liberation Sans") if f in avail), "DejaVu Sans")
    mpl.rcParams.update({   # set AFTER prg is imported (prg's plot_settings would override)
        "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif",
        "font.sans-serif": [family], "mathtext.fontset": "stixsans",
        "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 9, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "legend.fontsize": 8, "lines.linewidth": 1.0,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.minor.visible": False, "ytick.minor.visible": False,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "savefig.pad_inches": 0.02, "figure.dpi": 150, "axes.grid": False})
    cases = [c for c in results["cases"] if c["q"] == q_fig]
    regs = list(REGIMES)
    band = None
    rc = [c for c in (results.get("random_cases") or []) if c["regime"] == "C" and c["q"] == q_fig]
    if rc:
        mids = sorted({(c["gen"], c["model_id"]) for c in rc})
        nmod = len(mids)
        band = {}
        vals = sorted({c["value"] for c in rc}, reverse=True)
        for key, ck in (("mean_err", "mean"), ("cov_err_raw", "cov")):
            bx, blo, bhi = [], [], []
            for v in vals:
                at = [c["smoothers"]["VAR"] for c in rc if c["value"] == v]
                if any(d["n_failed"] for d in at):
                    break          # band only where VAR runs on every random model
                e = [max(d[ck]) if ck == "mean" else d[ck] for d in at]
                bx.append(v), blo.append(min(e)), bhi.append(max(e))
            band[key] = (np.array(bx), np.array(blo), np.array(bhi))
        band["note"] = f"shaded: VAR on the {nmod} random models with q={q_fig}"
    # explicit layout (inches): legend block, 4 rows of axes, the gap under row C carries
    # one more text line (the random-model note)
    top_in, bot_in, ax_h, xlab_in = 0.78, 0.34, 0.53, 0.196
    gaps = [0.44, 0.44, 0.60 if band else 0.44]
    W, H = 3.5, top_in + len(regs) * ax_h + sum(gaps) + bot_in
    fig = plt.figure(figsize=(W, H))
    left, right, wsp = 0.165, 0.985, 0.07
    axw = (right - left) / (2 + wsp)
    axes = np.empty((len(regs), 2), dtype=object)
    y_top = H - top_in
    for i in range(len(regs)):
        y0 = y_top - ax_h
        for j in range(2):
            axes[i, j] = fig.add_axes([left + j * axw * (1 + wsp), y0 / H, axw, ax_h / H])
        y_top = y0 - (gaps[i] if i < len(gaps) else 0.0)
    fmt = mpl.ticker.FuncFormatter(_pow10_label)
    for i, reg in enumerate(regs):
        rows = sorted([c for c in cases if c["regime"] == reg], key=lambda c: c["index"])
        xs = np.array([c["value"] for c in rows])
        for j, key in enumerate(("mean_err", "cov_err_raw")):
            ax = axes[i, j]
            for nm in NAMES:
                st = STYLE[nm]
                ys = np.array([c["smoothers"][nm][key]["max"] if key in c["smoothers"][nm]
                               else np.nan for c in rows], dtype=float)
                ys = np.clip(ys, FLOOR, YTOP)
                z = 3 if nm in ("DWY", "MBF") else 2
                ax.plot(xs, ys, color=st["color"], ls=st["ls"], lw=1.0, label=nm, zorder=z)
                # filled marker: error present in the interior window (or at roundoff);
                # open marker: error confined to start-up / end (see localise())
                loc = np.array([c["smoothers"][nm].get("localisation", {}).get(key, {}).get("class")
                                in ("start-up", "end") for c in rows])
                for sel, face in ((~loc, st["color"]), (loc, "white")):
                    if sel.any():
                        ax.plot(xs[sel], ys[sel], ls="none", marker=st["marker"], ms=st["ms"],
                                mfc=face, mec=st["color"], mew=0.8, zorder=z + 0.5)
                fail = np.array([c["smoothers"][nm]["n_failed"] > 0 for c in rows])
                for xf in xs[fail]:   # library abort: cross on the top edge, spread in x
                    c_f = next(c for c in rows if c["value"] == xf)
                    who = [m for m in NAMES if c_f["smoothers"][m]["n_failed"] > 0]
                    off = 10 ** (0.45 * (who.index(nm) - 0.5 * (len(who) - 1)))
                    if reg in ("S", "C", "M"):
                        off = 1 / off
                    if reg == "D":
                        off = off / 10 ** 0.9     # keep clear of the indefinite rings
                    ax.plot([xf * off], [YTOP], ls="none", marker="x", color=st["color"],
                            ms=5, mew=1.1, clip_on=False, zorder=6)
                if key == "cov_err_raw":
                    ind = np.array([c["smoothers"][nm].get("indefinite", {}).get("n_seeds", 0) > 0
                                    for c in rows])
                    if ind.any():   # raw smoothed covariance indefinite: small ring, offset
                        ax.plot(xs[ind] * 10 ** (0.35 * (NAMES.index(nm) - 1.5)), ys[ind],
                                ls="none", marker="o", ms=6.5, mfc="none",
                                mec=st["color"], mew=0.7, zorder=5)
            if reg == "C" and band is not None:
                # VAR on the random models at q = q_fig, over the 1-c values where all run
                bx, blo, bhi = band[key]
                ax.fill_between(bx, np.clip(blo, FLOOR, YTOP), np.clip(bhi, FLOOR, YTOP),
                                color=STYLE["VAR"]["color"], alpha=0.22, lw=0, zorder=1)
            ax.set_xscale("log")
            ax.set_yscale("log")
            if reg in ("S", "C", "M"):
                ax.invert_xaxis()      # stress increases to the right in every row
            ax.set_ylim(FLOOR / 3, YTOP)
            ax.set_yticks([1e-16, 1e-8, 1])
            ax.set_yticks([1e-12, 1e-4], minor=True)
            ax.yaxis.set_major_formatter(fmt)
            ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
            ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
            ax.set_xticks({"D": [1, 1e4, 1e8], "S": [1, 1e-4, 1e-8], "C": [1e-1, 1e-5, 1e-9],
                           "M": [1e-1, 1e-3]}[reg])
            ax.xaxis.set_major_formatter(fmt)
            ax.margins(x=0.07)
            ax.tick_params(axis="both", which="major", length=2.5, pad=1.5)
            ax.tick_params(axis="y", which="minor", length=1.5)
            ax.axhline(1.1e-16, color="#999999", lw=0.5, ls="-", zorder=0)
            if j == 1:
                ax.tick_params(labelleft=False)
            if i == 0:
                ax.set_title(["mean error", "covariance error"][j], pad=3)
        a0, a1 = axes[i, 0].get_position(), axes[i, 1].get_position()
        fig.text(0.5 * (a0.x0 + a1.x1), a0.y0 - xlab_in / H, FIG_XLABEL[reg], ha="center",
                 va="top", fontsize=9)
        if reg == "C" and band is not None:
            fig.text(0.5 * (a0.x0 + a1.x1), a0.y0 - (xlab_in + 0.17) / H, band["note"],
                     ha="center", va="top", fontsize=8, color="#333333")
    fig.text(0.008, 0.5 * (axes[0, 0].get_position().y1 + axes[-1, 0].get_position().y0),
             "relative error (max over $n$ and seeds)", rotation=90, ha="left", va="center",
             fontsize=9)
    h = [Line2D([], [], color=STYLE[nm]["color"], ls=STYLE[nm]["ls"], lw=1.0,
                marker=STYLE[nm]["marker"], ms=STYLE[nm]["ms"], mfc=STYLE[nm]["color"],
                mec=STYLE[nm]["color"], mew=0.8) for nm in NAMES]
    fig.legend(h, list(NAMES), loc="upper center", ncol=6, frameon=False,
               bbox_to_anchor=(0.54, 1 + 0.023 / H), handlelength=1.6, columnspacing=0.55,
               handletextpad=0.25)
    h2 = [Line2D([], [], ls="none", marker="o", ms=3.6, mfc="#555555", mec="#555555", mew=0.8),
          Line2D([], [], ls="none", marker="o", ms=3.6, mfc="white", mec="#555555", mew=0.8),
          Line2D([], [], ls="none", marker="x", ms=5, mew=1.1, color="#555555"),
          Line2D([], [], ls="none", marker="o", ms=6.5, mfc="none", mec="#555555", mew=0.7)]
    fig.legend(h2, ["≤1e−13 or also 100≤n≤300", "start-up only", "abort", "indefinite"],
               loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.54, 1 - 0.173 / H),
               handlelength=1.2, columnspacing=0.7, handletextpad=0.4)
    fig.savefig(str(path) + ".pdf")
    fig.savefig(str(path) + ".png", dpi=300)
    plt.close(fig)
    return family


def _fig_style():
    import matplotlib as mpl
    mpl.rcdefaults()   # drop prg's plot_settings (backend is not reset)
    mpl.use("Agg")
    import matplotlib.font_manager as fm
    avail = {f.name for f in fm.fontManager.ttflist}
    family = next((f for f in ("Arial", "Helvetica", "Liberation Sans") if f in avail), "DejaVu Sans")
    mpl.rcParams.update({   # set AFTER prg is imported (prg's plot_settings would override)
        "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif",
        "font.sans-serif": [family], "mathtext.fontset": "stixsans",
        "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 9, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "legend.fontsize": 8, "lines.linewidth": 1.0,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.minor.visible": False, "ytick.minor.visible": False,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "savefig.pad_inches": 0.02, "figure.dpi": 150, "axes.grid": False})
    return family


# main figure: x = log10 of the sweep parameter on a linear axis (short integer tick labels)
RF_XTICKS = {"D": [0, 5, 10], "S": [0, -5, -10], "C": [-2, -6, -10], "M": [-1, -2, -3, -4]}
RF_XTICKS_FAIL = {"D": [0, 10], "S": [0, -10], "C": [-2, -10], "M": [-1, -4]}   # narrow panel
RF_FIG_XLABEL = {"D": r"(D) diffuse prior: log10 of the prior scale $s$",
                 "S": r"(S) state-noise starvation: log10 $\varepsilon$",
                 "C": r"(C) correlated noise: log10$(1-c)$",
                 "M": r"(M) slow mixing: log10$(1-\rho(\mathbf{A}))$"}
ERR_KEYS = ("cov", "cov_mid", "mean")


def figure_data(results, gen="all", q="all"):
    """Per regime and smoother: x values, median / max over the random-family models of the
    raw covariance error (whole record and interior window) and of the mean error (whole
    record), abort and indefinite counts."""
    rs = results["random_summary"]
    out = {}
    for reg in REGIMES:
        blk = rs[reg][gen][str(q)]
        ivs = sorted(blk, key=int)
        d = {"x": np.array([blk[i]["value"] for i in ivs]),
             "n_models": [blk[i]["n_models"] for i in ivs]}
        for nm in NAMES:
            e = {}
            for key in ERR_KEYS:
                e[key + "_med"] = np.array([np.nan if blk[i][nm][key] is None else blk[i][nm][key]["median"]
                                            for i in ivs])
                e[key + "_max"] = np.array([np.nan if blk[i][nm][key] is None else blk[i][nm][key]["max"]
                                            for i in ivs])
            e["abort"] = np.array([blk[i][nm]["n_abort"] for i in ivs])
            e["indef"] = np.array([blk[i][nm]["n_indefinite"] for i in ivs])
            e["n_ran"] = np.array([blk[i][nm]["n_ran"] for i in ivs])
            d[nm] = e
        out[reg] = d
    return out


def make_figure(results, path):
    """Main figure: random families (G1 + G2, q = 1 and 2 pooled).

    One row per regime, four panels: raw covariance error over the whole record, the same in
    the interior window, mean error over the whole record -- line = median over the models
    on which the smoother ran (not drawn where fewer than half ran), open marker = max -- and
    the number of models on which the smoother aborts (x) or returns an indefinite raw
    covariance (o)."""
    import matplotlib as mpl
    family = _fig_style()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fd = figure_data(results)
    regs = list(REGIMES)
    allv = [v for reg in regs for nm in NAMES for k in ERR_KEYS
            for v in fd[reg][nm][k + "_max"] if np.isfinite(v)]
    ytop = 10 ** np.ceil(np.log10(max(allv) * 3))
    ylo = 1e-17
    top_in, bot_in, ax_h, gap = 0.80, 0.36, 0.56, 0.40
    W, H = 3.5, top_in + len(regs) * ax_h + (len(regs) - 1) * gap + bot_in
    fig = plt.figure(figsize=(W, H))
    left_in, wpan, wab = 0.50, 0.69, 0.46
    seps = (0.08, 0.09, 0.09)
    xs_in = [left_in]
    for w, sp in zip((wpan, wpan, wpan), seps):
        xs_in.append(xs_in[-1] + w + sp)
    ws_in = [wpan, wpan, wpan, wab]
    axes = np.empty((len(regs), 4), dtype=object)
    y_top = H - top_in
    for i in range(len(regs)):
        y0 = y_top - ax_h
        for j in range(4):
            axes[i, j] = fig.add_axes([xs_in[j] / W, y0 / H, ws_in[j] / W, ax_h / H])
        y_top = y0 - gap
    fmt = mpl.ticker.FuncFormatter(_pow10_label)
    for i, reg in enumerate(regs):
        d = fd[reg]
        x = np.log10(d["x"])          # linear axis in log10 of the sweep parameter
        nmod = max(d["n_models"])
        for j, key in enumerate(ERR_KEYS):
            ax = axes[i, j]
            for nm in NAMES:
                st, e = STYLE[nm], d[nm]
                z = 3 if nm in ("DWY", "MBF", "VAR") else 2
                med = np.clip(e[key + "_med"], ylo, ytop)
                # a median over fewer than half of the models is not drawn (aborts)
                med = np.where(e["n_ran"] >= 0.5 * np.array(d["n_models"]), med, np.nan)
                mx = np.clip(e[key + "_max"], ylo, ytop)
                ax.plot(x, med, color=st["color"], ls=st["ls"], lw=1.0, zorder=z)
                ax.plot(x, med, ls="none", marker=st["marker"], ms=st["ms"] * 0.8,
                        mfc=st["color"], mec=st["color"], mew=0.6, zorder=z + 0.2)
                ax.plot(x, mx, ls="none", marker=st["marker"], ms=st["ms"], mfc="none",
                        mec=st["color"], mew=0.7, zorder=z + 0.4)
            ax.set_yscale("log")
            ax.set_ylim(ylo, ytop)
            ax.set_yticks([1e-16, 1e-8, 1])
            ax.yaxis.set_major_formatter(fmt)
            ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
            ax.axhline(1.1e-16, color="#999999", lw=0.5, ls="-", zorder=0)
            if j > 0:
                ax.tick_params(labelleft=False)
        ax = axes[i, 3]
        span = x.max() - x.min()
        drawn = 0
        for k, nm in enumerate(NAMES):
            e = d[nm]
            st = STYLE[nm]
            dodge = 0.03 * span * (k - 2.5)   # small x-offset so equal counts stay visible
            if e["abort"].any():
                ax.plot(x + dodge, e["abort"], color=st["color"], ls="-", lw=0.9, marker="x",
                        ms=4, mew=0.8, zorder=3)
                drawn += 1
            if e["indef"].any():
                ax.plot(x + dodge, e["indef"], color=st["color"], ls="--", lw=0.8,
                        marker="o", ms=3.5, mfc="none", mec=st["color"], mew=0.7, zorder=3)
                drawn += 1
        if not drawn:
            ax.text(0.5, 0.5, "none", transform=ax.transAxes, ha="center", va="center",
                    fontsize=8, color="#555555")
        ax.set_yscale("function", functions=(lambda v: np.sqrt(np.maximum(v, 0)),
                                             lambda v: np.square(v)))
        ax.set_ylim(0, nmod * 1.06)
        ax.set_yticks([0, 5, 20, 80] if nmod == 80 else [0, nmod // 4, nmod])
        ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%d"))
        ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax.yaxis.tick_right()
        for j in range(4):
            ax = axes[i, j]
            ax.margins(x=0.10)
            if reg in ("S", "C", "M"):
                ax.invert_xaxis()      # stress increases to the right in every row
            ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
            ax.set_xticks((RF_XTICKS_FAIL if j == 3 else RF_XTICKS)[reg])
            ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(
                lambda v, _p: f"{v:.0f}".replace("-", "−")))
            ax.tick_params(axis="both", which="major", length=2.5, pad=1.5)
        a0, a3 = axes[i, 0].get_position(), axes[i, 3].get_position()
        fig.text(0.5 * (a0.x0 + a3.x1), a0.y0 - 0.20 / H, RF_FIG_XLABEL[reg], ha="center",
                 va="top", fontsize=9)
    # column headers: quantity (9 pt) and range of n (8 pt)
    p0, p1, p2, p3 = (axes[0, j].get_position() for j in range(4))
    yq, yn = p0.y1 + 0.17 / H, p0.y1 + 0.03 / H
    fig.text(0.5 * (p0.x0 + p1.x1), yq, "covariance", ha="center", va="bottom", fontsize=9)
    fig.text(0.5 * (p2.x0 + p2.x1), yq, "mean", ha="center", va="bottom", fontsize=9)
    fig.text(0.5 * (p3.x0 + p3.x1), yq, "failures", ha="center", va="bottom", fontsize=9)
    for pp, lab in ((p0, "all n"), (p1, f"{MID[0]}≤n≤{MID[1]}"), (p2, "all n"),
                    (p3, "(models)")):
        fig.text(0.5 * (pp.x0 + pp.x1), yn, lab, ha="center", va="bottom", fontsize=8)
    fig.text(0.008, 0.5 * (axes[0, 0].get_position().y1 + axes[-1, 0].get_position().y0),
             "relative error (max over $n$)", rotation=90, ha="left", va="center", fontsize=9)
    h = [Line2D([], [], color=STYLE[nm]["color"], ls=STYLE[nm]["ls"], lw=1.0,
                marker=STYLE[nm]["marker"], ms=STYLE[nm]["ms"] * 0.8, mfc=STYLE[nm]["color"],
                mec=STYLE[nm]["color"], mew=0.6) for nm in NAMES]
    fig.legend(h, list(NAMES), loc="upper center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, 1 + 0.02 / H), handlelength=1.6, columnspacing=0.55,
               handletextpad=0.25)
    h2 = [Line2D([], [], ls="-", color="#555555", marker="o", ms=3, mfc="#555555", mew=0.6),
          Line2D([], [], ls="none", marker="o", ms=3.6, mfc="none", mec="#555555", mew=0.7),
          Line2D([], [], ls="-", color="#555555", marker="x", ms=4, mew=0.9),
          Line2D([], [], ls="--", color="#555555", lw=0.8, marker="o", ms=3.5, mfc="none",
                 mec="#555555", mew=0.7)]
    fig.legend(h2, ["median", "max", "abort", "indefinite"],
               loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1 - 0.18 / H),
               handlelength=1.5, columnspacing=0.6, handletextpad=0.3)
    fig.savefig(str(path) + ".pdf")
    fig.savefig(str(path) + ".png", dpi=300)
    plt.close(fig)
    return family


# compact letter figure (covariance error only; aborts / indefinite in a strip per panel)
LETTER_XLABEL = {"D": r"$\log_{10}(s)$", "S": r"$\log_{10}(\varepsilon)$",
                 "C": r"$\log_{10}(1-c)$", "M": r"$\log_{10}(1-\rho)$"}
LETTER_PANEL = {"D": "(a) D: broad prior", "S": "(b) S: vanishing state noise",
                "C": "(c) C: correlated noise", "M": "(d) M: slow mixing"}
MEAN_COV_FACTOR = 100.0    # mean vs covariance error "differ": ratio > this (either way) ...
MEAN_COV_FLOOR = 1e-12     # ... with the larger of the two above this (else both at round-off)


def mean_vs_cov(results):
    """Per regime and smoother (random families pooled): the sweep values where the mean
    error (whole record) and the covariance error (whole record, raw) differ by more than
    MEAN_COV_FACTOR in median or in max, the larger one exceeding MEAN_COV_FLOOR."""
    fd = figure_data(results)
    out = []
    for reg in REGIMES:
        d = fd[reg]
        for nm in NAMES:
            e = d[nm]
            for stat in ("med", "max"):
                c, m = e["cov_" + stat], e["mean_" + stat]
                for i, v in enumerate(d["x"]):
                    if not (np.isfinite(c[i]) and np.isfinite(m[i])):
                        continue
                    big = max(c[i], m[i])
                    r = max(c[i], 1e-300) / max(m[i], 1e-300)
                    if big > MEAN_COV_FLOOR and (r > MEAN_COV_FACTOR or r < 1 / MEAN_COV_FACTOR):
                        out.append({"regime": reg, "smoother": nm, "stat": stat, "value": float(v),
                                    "cov": float(c[i]), "mean": float(m[i]),
                                    "larger": "cov" if r > 1 else "mean"})
    return out


PN_TXT = {"D": "s", "S": "eps", "C": "1-c", "M": "1-rho"}


def _vf(v):
    """Compact sweep value: 1, 1e10, 1e-4, 3e-3."""
    if v == 1:
        return "1"
    m, e = f"{v:.0e}".split("e")
    return f"{m}e{int(e)}" if m != "1" else f"1e{int(e)}"


def _vals_txt(reg, vs):
    """Sweep values as text; a tail of the sweep (in the stress direction) as a range."""
    order = RF_VALUES[reg]
    vs = sorted(set(vs), key=lambda v: order.index(v) if v in order else 0)
    idx = [order.index(v) for v in vs if v in order]
    pn = PN_TXT[reg]
    if len(idx) == len(vs) and len(vs) > 1 and idx == list(range(len(order) - len(vs), len(order))):
        return f"{pn} {'>=' if reg == 'D' else '<='} {_vf(vs[0])}"
    return f"{pn} = " + ", ".join(_vf(v) for v in vs)


def mean_vs_cov_groups(results):
    """mean_vs_cov() grouped by (regime, smoother, larger quantity), with the sweep values
    and the extreme pair (cov, mean) where the ratio is largest."""
    groups = {}
    for x in mean_vs_cov(results):
        k = (x["regime"], x["smoother"], x["larger"])
        groups.setdefault(k, []).append(x)
    out = []
    for (reg, nm, larger), xs in sorted(groups.items(), key=lambda kv: (list(REGIMES).index(kv[0][0]),
                                                                        NAMES.index(kv[0][1]))):
        w = max(xs, key=lambda x: max(x["cov"] / max(x["mean"], 1e-300), x["mean"] / max(x["cov"], 1e-300)))
        out.append({"regime": reg, "smoother": nm, "larger": larger,
                    "values": sorted({x["value"] for x in xs}), "stats": sorted({x["stat"] for x in xs}),
                    "worst": {"value": w["value"], "stat": w["stat"], "cov": w["cov"], "mean": w["mean"]}})
    return out


def mean_vs_cov_note(results):
    gs = mean_vs_cov_groups(results)
    txt = "; ".join(
        f"{g['regime']} {g['smoother']}: {g['larger']} larger at {_vals_txt(g['regime'], g['values'])} "
        f"({' and '.join(g['stats'])}; most extreme: {g['worst']['stat']} cov {_g(g['worst']['cov'])} vs "
        f"mean {_g(g['worst']['mean'])} at {_vf(g['worst']['value'])})" for g in gs)
    return (f"Mean vs covariance error (random families pooled, whole record; median and max over the "
            f"models that ran): they differ by more than {MEAN_COV_FACTOR:g}x, the larger one above "
            f"{MEAN_COV_FLOOR:g}, only in: " + (txt if txt else "no cell") + ".")


def _abort_indef_txt(results, key):
    """'2F: 3 at s=1e8, 26 at s=1e10; ...' for n_abort (key) or n_indefinite, pooled models."""
    rs = results["random_summary"]
    out = []
    for reg in REGIMES:
        blk = rs[reg]["all"]["all"]
        per = []
        for nm in NAMES:
            items = [(blk[iv]["value"], blk[iv][nm][key]) for iv in sorted(blk, key=int)
                     if blk[iv][nm][key]]
            if items:
                per.append(f"{nm} " + ", ".join(f"{n} at {PN_TXT[reg]}={_vf(v)}" for v, n in items))
        if per:
            out.append(f"({reg}) " + "; ".join(per))
    return out


def build_caption_letter(results):
    rf = results["meta"]["random_families"]
    m = rf["models_per_generator_and_q"]
    n_all = 2 * len(QS) * m
    ab = _abort_indef_txt(results, "n_abort")
    ind = _abort_indef_txt(results, "n_indefinite")
    gs = mean_vs_cov_groups(results)

    def by_dir(larger):
        out = []
        for reg in REGIMES:
            it = [g for g in gs if g["regime"] == reg and g["larger"] == larger]
            items = {}
            for g in it:   # smoothers sharing the same values and statistics are listed together
                tail = (f" at {_vals_txt(reg, g['values'])}"
                        + (" (max only)" if g["stats"] == ["max"] else
                           " (median only)" if g["stats"] == ["med"] else ""))
                items.setdefault(tail, []).append(g["smoother"])
            if items:
                out.append(f"({reg}) " + ", ".join(f"{'/'.join(v)}{k}" for k, v in items.items()))
        return "; ".join(out)
    small, large = by_dir("cov"), by_dir("mean")
    mean_txt = (f"Mean errors (not shown) are within a factor {MEAN_COV_FACTOR:g} of the covariance "
                f"errors (pooled median and max), or both below {MEAN_COV_FLOOR:g}, except where they "
                f"are more than {MEAN_COV_FACTOR:g}x smaller: " + (small or "nowhere")
                + (f"; and where they are more than {MEAN_COV_FACTOR:g}x larger: {large}" if large else
                   f"; they are nowhere more than {MEAN_COV_FACTOR:g}x larger") + ".")
    return (f"Covariance error max_n ||P_n-P_ref,n||_2/||P_ref,n||_2 of the six library smoothers "
            f"(raw output, before the library's in-place regularisation) against a 60-digit "
            f"reference, N=400, in four stress regimes: (D) diffuse prior P_0=sI; (S) state-noise "
            f"scale eps (R^xx -> eps R^xx); (C) X/Y noise correlation c -> 1; (M) spectral radius "
            f"rho(A) -> 1. Each point pools {n_all} random models ({m} from each of two generators at "
            f"each of (p,q)=(2,1) and (2,2); {rf['seeds']} records per model, max over records). Line: "
            f"median over the models on which the smoother ran (not drawn where fewer than half ran); "
            f"open marker: max over them; grey line: unit roundoff 1.1e-16; errors near 1e-15 are at "
            f"round-off and overlap. Grey strip above each panel: x the smoother aborted, filled dot "
            f"it returned an indefinite raw covariance, on at least one model at that value. Number of "
            f"models (of {n_all}) with an abort: " + ("; ".join(ab) if ab else "none")
            + "; with an indefinite raw covariance: " + ("; ".join(ind) if ind else "none") + ". "
            + mean_txt)


def make_letter_figure(results, path, layout="2x2wide"):
    """Letter figure, layout "2x2wide" (full page width, D, S / C, M), "1x4" (full page width,
    one row) or "2x2" (one column): covariance error (raw, whole record)
    vs log10 of the sweep parameter, six smoothers: median line + max open marker over the
    random-family models; a strip above each panel marks aborts (x) and indefinite raw
    covariances (o), on at least one model, per smoother."""
    import matplotlib as mpl
    family = _fig_style()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fd = figure_data(results)
    regs = list(REGIMES)
    allv = [v for reg in regs for nm in NAMES for v in fd[reg][nm]["cov_max"] if np.isfinite(v)]
    ytop_all = 10 ** np.ceil(np.log10(max(allv) * 3))
    ylo = 10 ** -16.5 if layout == "2x2wide" else 1e-17

    def _ytop(reg):
        if layout != "2x2wide":
            return ytop_all
        v = [u for nm in NAMES for u in fd[reg][nm]["cov_max"] if np.isfinite(u)]
        return 10 ** np.ceil(np.log10(max(v) * 10))
    if layout == "2x2wide":
        mpl.rcParams.update({"xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9})
    wide = layout in ("1x4", "2x2wide")
    ncols = 4 if layout == "1x4" else 2
    nrows = len(regs) // ncols
    if layout == "2x2wide":
        W = 7.16
        leg_h, strip_h, sgap, ax_h, xlab_h, bot = 0.44, 0.22, 0.02, 2.45, 0.36, 0.03
        left_in, right_in, gap_in = 0.66, 0.04, 0.42
    elif layout == "1x4":
        W = 7.16
        leg_h, strip_h, sgap, ax_h, xlab_h, bot = 0.22, 0.12, 0.02, 0.92, 0.30, 0.03
        left_in, right_in, gap_in = 0.56, 0.04, 0.16
    else:
        W = 3.5
        leg_h, strip_h, sgap, ax_h, xlab_h, bot = 0.36, 0.12, 0.02, 0.64, 0.30, 0.03
        left_in, right_in, gap_in = 0.56, 0.04, 0.10
    H = leg_h + nrows * (strip_h + sgap + ax_h + xlab_h) + bot
    pw = (W - left_in - right_in - (ncols - 1) * gap_in) / ncols
    fig = plt.figure(figsize=(W, H))
    axes, strips = {}, {}
    for i, reg in enumerate(regs):
        r, c = divmod(i, ncols)
        x0 = left_in + c * (pw + gap_in)
        ytop_row = H - leg_h - r * (strip_h + sgap + ax_h + xlab_h)
        strips[reg] = fig.add_axes([x0 / W, (ytop_row - strip_h) / H, pw / W, strip_h / H])
        axes[reg] = fig.add_axes([x0 / W, (ytop_row - strip_h - sgap - ax_h) / H, pw / W, ax_h / H],
                                 sharex=strips[reg])
    fmt = mpl.ticker.FuncFormatter(_pow10_label)
    for i, reg in enumerate(regs):
        d = fd[reg]
        x = np.log10(d["x"])
        ax, sx = axes[reg], strips[reg]
        span = x.max() - x.min()
        for k, nm in enumerate(NAMES):
            st, e = STYLE[nm], d[nm]
            z = 3 if nm in ("DWY", "MBF", "VAR") else 2
            ytop = _ytop(reg)
            med = np.clip(e["cov_med"], ylo, ytop)
            med = np.where(e["n_ran"] >= 0.5 * np.array(d["n_models"]), med, np.nan)
            mx = np.clip(e["cov_max"], ylo, ytop)
            ax.plot(x, med, color=st["color"], ls=st["ls"], lw=1.2, marker=st["marker"],
                    ms=st["ms"], mfc=st["color"], mec=st["color"], mew=0.6, zorder=z + 0.4)
            ax.plot(x, mx, color=st["color"], ls=st["ls"], lw=1.0, alpha=0.7, zorder=z)
            dodge = 0.032 * span * (k - 2.5)
            sel = e["abort"] > 0
            if sel.any():
                sx.plot(x[sel] + dodge, np.full(sel.sum(), 0.68), ls="none", marker="x",
                        ms=6.0 if layout == "2x2wide" else 3.4, mew=1.6 if layout == "2x2wide" else 1.0,
                        color=st["color"], clip_on=False)
            sel = e["indef"] > 0
            if sel.any():
                sx.plot(x[sel] + dodge, np.full(sel.sum(), 0.28), ls="none", marker="o",
                        ms=5.0 if layout == "2x2wide" else 2.8,
                        mfc=st["color"], mec=st["color"], mew=0.5, clip_on=False)
        ax.set_yscale("log")
        ytop = _ytop(reg)
        ax.set_ylim(ylo, ytop)
        if layout == "2x2wide":
            ax.set_yticks([10.0 ** k for k in range(-16, int(np.log10(ytop)) + 1, 4)])
        else:
            ax.set_yticks([1e-16, 1e-8, 1])
        ax.yaxis.set_major_formatter(mpl.ticker.LogFormatterMathtext() if layout == "2x2wide" else fmt)
        ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
        ax.axhline(1.1e-16, color="#999999", lw=0.5, ls="-", zorder=0)
        if i % ncols != 0 and layout != "2x2wide":
            ax.tick_params(labelleft=False)
        sx.set_ylim(0, 1)
        sx.set_yticks([])
        sx.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        for sp in sx.spines.values():
            sp.set_visible(False)
        sx.set_facecolor("#ececec")
        if i % ncols == 0:
            sx.text(-0.03, 0.5, "fails", transform=sx.transAxes, ha="right", va="center",
                    fontsize=8, color="#444444")
        ax.margins(x=0.08)
        if reg in ("S", "C", "M"):
            ax.invert_xaxis()      # stress increases to the right in every panel
        ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
        ax.set_xticks(RF_XTICKS[reg])
        ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _p: f"{v:.0f}".replace("-", "−")))
        ax.tick_params(axis="both", which="major", length=2.5, pad=1.5)
        ax.set_xlabel(LETTER_XLABEL[reg], fontsize=11 if layout == "2x2wide" else 9, labelpad=1.5)
        ax.text(0.015, 0.97, LETTER_PANEL[reg], transform=ax.transAxes, ha="left", va="top",
                fontsize=10 if layout == "2x2wide" else 8, zorder=10,
                bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.85))
    y_mid = 0.5 * (axes["D"].get_position().y1 + axes[regs[ncols * (nrows - 1)]].get_position().y0)
    fig.text(0.004, y_mid, "covariance error", rotation=90, ha="left", va="center",
             fontsize=10 if layout == "2x2wide" else 9)
    h = [Line2D([], [], color=STYLE[nm]["color"], ls=STYLE[nm]["ls"], lw=1.2,
                marker=STYLE[nm]["marker"], ms=STYLE[nm]["ms"], mfc=STYLE[nm]["color"],
                mec=STYLE[nm]["color"], mew=0.6) for nm in NAMES]
    h2 = [Line2D([], [], ls="-", color="#888888", lw=1.4),
          Line2D([], [], ls="-", color="#888888", lw=1.0, alpha=0.5),
          Line2D([], [], ls="none", marker="x", ms=6.0 if layout == "2x2wide" else 3.4,
                 mew=1.6 if layout == "2x2wide" else 1.0, color="#555555"),
          Line2D([], [], ls="none", marker="o", ms=5.0 if layout == "2x2wide" else 2.8,
                 mfc="#555555", mec="#555555", mew=0.5)]
    labels2 = (["median", "maximum", "abort (grey strip)", "indefinite (grey strip)"]
               if layout == "2x2wide" else ["median", "maximum", "abort", "indefinite"])
    if layout == "2x2wide":
        fig.legend(h, list(NAMES), loc="upper center", ncol=6, frameon=False,
                   bbox_to_anchor=(0.5, 1 + 0.03 / H), handlelength=1.7, columnspacing=1.2,
                   handletextpad=0.3, borderaxespad=0.0)
        fig.legend(h2, labels2, loc="upper center", ncol=4, frameon=False,
                   bbox_to_anchor=(0.5, 1 - 0.20 / H), handlelength=1.7, columnspacing=1.6,
                   handletextpad=0.3, borderaxespad=0.0)
    elif wide:
        fig.legend(h + h2, list(NAMES) + labels2, loc="upper center", ncol=10, frameon=False,
                   bbox_to_anchor=(0.5, 1 + 0.03 / H), handlelength=1.5, columnspacing=0.55,
                   handletextpad=0.2, borderaxespad=0.0)
    else:
        fig.legend(h, list(NAMES), loc="upper center", ncol=6, frameon=False,
                   bbox_to_anchor=(0.5, 1 + 0.03 / H), handlelength=1.7, columnspacing=0.6,
                   handletextpad=0.25, borderaxespad=0.0)
        fig.legend(h2, labels2, loc="upper center", ncol=4,
                   frameon=False, bbox_to_anchor=(0.5, 1 - 0.155 / H), handlelength=1.5,
                   columnspacing=0.8, handletextpad=0.3, borderaxespad=0.0)
    fig.savefig(str(path) + ".pdf")
    fig.savefig(str(path) + ".png", dpi=300)
    plt.close(fig)
    return family, (W, H)


def _e(x):
    return "-" if x is None else f"{x:.0e}"


def print_random_families(results):
    """Aggregated tables of the random families (from results['random_summary'])."""
    rs = results["random_summary"]
    meta = results["meta"]["random_families"]
    print(f"\n=== Random families: {meta['models_per_generator_and_q']} models per (generator, q), "
          f"{meta['seeds']} seeds per model; errors = max over the model's seeds; stats over the "
          "models on which the smoother ran ===")
    W = 23
    for reg, spec in REGIMES.items():
        pn = spec["param"]
        for gen in list(GENS) + ["all"]:
            for q in list(QS) + ["all"]:
                blk = rs[reg][gen].get(str(q))
                if not blk or (gen == "all") != (q == "all"):
                    continue    # print G1/G2 x q=1/2 and the pooled all/all only
                nm0 = next(iter(blk.values()))["n_models"]
                print(f"\n--- Regime {reg} ({spec['title']}), generator {gen}, q={q}: {nm0} models ---")
                for key, lab in (("cov", "covariance error (raw), whole record"),
                                 ("cov_mid", f"covariance error (raw), interior [{MID[0]},{MID[1]}]"),
                                 ("mean", "mean error, whole record"),
                                 ("mean_mid", f"mean error, interior [{MID[0]},{MID[1]}]")):
                    print(f"  [{key}] {lab}: median/p90/max over models")
                    print(f"  {pn:>7} " + "".join(f"{nm:>{W}}" for nm in NAMES))
                    for iv in sorted(blk, key=int):
                        row = blk[iv]
                        line = f"  {row['value']:>7.0e} "
                        for nm in NAMES:
                            st = row[nm][key]
                            line += f"{'ALL ABORT' if st is None else _e(st['median']) + '/' + _e(st['p90']) + '/' + _e(st['max']):>{W}}"
                        print(line)
                print("  [counts] aborts a, indefinite raw covariance i, library-regularised r, forward-"
                      "regularised f; covariance error classes P persistent / s start-up / e end "
                      "(errors > 1e-13 only)")
                print(f"  {pn:>7} " + "".join(f"{nm:>{W}}" for nm in NAMES))
                for iv in sorted(blk, key=int):
                    row = blk[iv]
                    line = f"  {row['value']:>7.0e} "
                    for nm in NAMES:
                        d = row[nm]
                        lc = d["loc_cov"]
                        s = (f"a{d['n_abort']} i{d['n_indefinite']} r{d['n_regularised']} "
                             f"f{d['n_forward_regularised']} P{lc['persistent']}s{lc['start-up']}e{lc['end']}")
                        line += f"{s:>{W}}"
                    print(line)
                print("  [mean classes] P/s/e of the mean error")
                for iv in sorted(blk, key=int):
                    row = blk[iv]
                    print(f"  {row['value']:>7.0e} " + "".join(
                        f"{'P%ds%de%d' % (row[nm]['loc_mean']['persistent'], row[nm]['loc_mean']['start-up'], row[nm]['loc_mean']['end']):>{W}}"
                        for nm in NAMES))
                causes = []
                for iv in sorted(blk, key=int):
                    row = blk[iv]
                    for nm in NAMES:
                        for k, v in row[nm]["abort_causes"].items():
                            st = row[nm]["abort_steps"].get(k)
                            causes.append(f"    {row['value']:.0e} {nm:>4}: {v} model(s): {k}"
                                          + (f", steps {st[0]}-{st[1]}" if st else "")
                                          + (f"; tolerance artefact on {row[nm]['n_abort_tolerance_artefact']}"
                                             if row[nm]["n_abort_tolerance_artefact"] else ""))
                if causes:
                    print("  [aborts] cause = exception:matrix [failed check] diagnosis code")
                    print("\n".join(causes))
    print_consistency(results)


def print_consistency(results):
    """2F/DWY of the library vs the 2F*/DWY* consistency control."""
    cs = results.get("consistency_summary") or consistency_summary(results)
    print("\n=== Consistency check: library 2F/DWY vs the 2F*/DWY* control (previous version's "
          "symmetrised backward filter) ===")
    print("  per regime and smoother: cases | both complete | both abort | library-only aborts | "
          f"control-only aborts | bitwise equal on all seeds | output difference cov / mean: median, "
          f"max | #cases with cov / mean difference > {CONS_ROUNDOFF:g} | #cases with errors vs the "
          f"reference differing by > {CONS_RATIO:g}x")
    for reg in REGIMES:
        for lab in ("random_families", "base_model"):
            for base, d in cs[reg][lab].items():
                cd, md = d["cov_diff"] or {}, d["mean_diff"] or {}
                print(f"  {reg} {lab:>15} {base:>3}: {d['n_cases']:>3} | {d['both_complete']:>3} | "
                      f"{d['both_abort']:>2} | {d['library_only_aborts']:>2} | {d['control_only_aborts']:>2} | "
                      f"{d['n_bitwise_all_seeds']:>3} | {_g(cd.get('median'))}, {_g(cd.get('max'))} / "
                      f"{_g(md.get('median'))}, {_g(md.get('max'))} | {d['n_cov_diff_gt_roundoff']} / "
                      f"{d['n_mean_diff_gt_roundoff']} | {d['n_error_ratio_gt']}")
    out = cs["disagreements"]
    print(f"  {cs['n_beyond_roundoff_or_disagreeing']} smoother x case records differ by more than "
          f"{CONS_ROUNDOFF:g} or in abort status; of these, abort status or errors (> {CONS_RATIO:g}x) "
          f"differ on {len(out)}: regime gen q #model value smoother | aborting seeds library/control | "
          "cov error library/control | mean error library/control | direct difference cov/mean")
    for x in out:
        who = "base" if x["gen"] is None else f"{x['gen']} q={x['q']} #{x['model_id']}"
        if x["gen"] is None:
            who += f" (2,{x['q']})"
        print(f"    {x['regime']} {who} {x['value']:.0e} {x['smoother']:>3} | {x['abort_library']}/"
              f"{x['abort_control']} | {_g(x['cov_library'])}/{_g(x['cov_control'])} | "
              f"{_g(x['mean_library'])}/{_g(x['mean_control'])} | {_g(x['cov_diff_max'])}/"
              f"{_g(x['mean_diff_max'])}")


def _g(x):
    return "-" if x is None else f"{x:.1e}"


def _base_rows(results, reg, q):
    return sorted([c for c in results["cases"] if c["regime"] == reg and c["q"] == q],
                  key=lambda c: c["index"])


def regime_note_random(results, reg, iv=None):
    """Factual sentence: random families, one regime, one sweep value (default: the most
    extreme), pooled and per generator; covariance and mean errors, classes, aborts."""
    rs = results["random_summary"][reg]
    vals = RF_VALUES[reg]
    iv = len(vals) - 1 if iv is None else iv
    pn = REGIMES[reg]["param"]
    row = rs["all"]["all"][str(iv)]
    nmod = row["n_models"]
    parts = []
    for nm in NAMES:
        d = row[nm]
        if d["cov"] is None:
            parts.append(f"{nm}: aborts on {d['n_abort']}/{nmod}")
            continue
        lc = d["loc_cov"]
        s = (f"{nm}: cov median {_g(d['cov']['median'])}, p90 {_g(d['cov']['p90'])}, max "
             f"{_g(d['cov']['max'])} (interior max {_g(d['cov_mid']['max'])}; persistent on "
             f"{lc['persistent']}, start-up on {lc['start-up']}, end on {lc['end']} of the "
             f"{d['n_ran']} runs); mean max {_g(d['mean']['max'])} (interior {_g(d['mean_mid']['max'])})")
        if d["n_abort"]:
            s += f"; aborts on {d['n_abort']}/{nmod}"
        if d["n_indefinite"]:
            s += f"; indefinite raw covariance on {d['n_indefinite']}/{d['n_ran']}"
        parts.append(s)
    per_gen = []
    for gen in GENS:
        for q in QS:
            r = rs[gen][str(q)][str(iv)]
            bad = [f"{nm} {r[nm]['n_abort']}" for nm in NAMES if r[nm]["n_abort"]]
            worst = max(((r[nm]["cov"] or {}).get("max") or 0, nm) for nm in NAMES)
            per_gen.append(f"{gen} q={q} ({r['n_models']} models): largest cov max {worst[1]} "
                           f"{_g(worst[0])}" + (f", aborts {', '.join(bad)}" if bad else ", no abort"))
    return (f"Random families, regime {reg}, {pn} = {vals[iv]:g}, {nmod} models (G1+G2, q=1,2; "
            f"{results['meta']['random_families']['seeds']} seeds each; errors = max over seeds; "
            "statistics over the models on which the smoother ran): " + "; ".join(parts)
            + ". Per generator and q: " + "; ".join(per_gen) + ".")


def regime_note_base(results, reg):
    """Factual sentence: base model, one regime, most extreme value, both q."""
    out = []
    for q in QS:
        rows = _base_rows(results, reg, q)
        if not rows:
            continue
        c = rows[-1]
        pn = REGIMES[reg]["param"]
        parts = []
        for nm in NAMES:
            d = c["smoothers"][nm]
            if "cov_err_raw" not in d:
                f = d["failures"][0]
                parts.append(f"{nm} aborts ({f['matrix']}, step {f['step']})")
                continue
            lc = d["localisation"]
            parts.append(f"{nm} cov {_g(d['cov_err_raw']['max'])} ({lc['cov_err_raw']['class']}), mean "
                         f"{_g(d['mean_err']['max'])} ({lc['mean_err']['class']})")
        out.append(f"(2,{q}) {pn}={c['value']:g}: " + "; ".join(parts))
    return (f"Base model, regime {reg}, most extreme value, max over {results['meta']['seeds']} seeds: "
            + " | ".join(out) + ".")


def _where_or_matrix(f):
    """'on <matrix>' for a failed library check, 'in <function>' for an exception raised
    outside the library's named checks (e.g. numpy 'Singular matrix' in _cond_xy)."""
    if f.get("matrix") is not None:
        return f"on {f['matrix']}"
    w = f.get("where")
    return f"in {w.split(':')[1]}" if w and w.count(":") >= 2 else f"in {w}"


def abort_cause_notes(results):
    """Every abort (base model and random families), grouped by smoother, check and
    diagnosis, with the tolerance-artefact count."""
    groups = {}
    for c in results["cases"]:
        for nm in NAMES:
            for f in c["smoothers"][nm]["failures"]:
                dg = f.get("diagnosis", {})
                cs = dg.get("cause", {})
                k = (c["regime"], nm, f["type"], _where_or_matrix(f),
                     "/".join(cs.get("failed_checks") or []),
                     dg.get("diagnosis_code"), bool(cs.get("tolerance_artefact")))
                groups.setdefault(k, {"base": 0, "random": 0})["base"] += 1
    for c in results["random_cases"]:
        for nm in NAMES:
            for f in c["smoothers"][nm].get("failures", []):
                k = (c["regime"], nm, f["type"], _where_or_matrix(f),
                     "/".join(f.get("failed_checks") or []),
                     f.get("diagnosis_code"), bool(f.get("tolerance_artefact")))
                groups.setdefault(k, {"base": 0, "random": 0})["random"] += 1
    lines = []
    for k in sorted(groups, key=lambda k: (list(REGIMES).index(k[0]), NAMES.index(k[1]))):
        reg, nm, typ, mat, chk, code, art = k
        v = groups[k]
        lines.append(f"{reg} {nm}: {typ} {mat} [{chk}], diagnosis {code}, tolerance artefact {art}: "
                     f"{v['base']} base-model cases, {v['random']} random-family cases")
    return lines


def key_findings(results):
    """Synthesis statements. Every number is read from the results; every statement names
    the set it is measured on (random families pooled: 80 models = G1 + G2, q = 1 and 2,
    20 models each; per generator / q splits are in random_summary; base model separately)."""
    S_ = results["random_summary"]
    rc = results["random_cases"]

    def cell(reg, iv, nm, gen="all", q="all"):
        return S_[reg][gen][str(q)][str(iv)][nm]

    def mx(reg, nms, key, ivs=None, gen="all", q="all"):
        ivs = range(len(RF_VALUES[reg])) if ivs is None else ivs
        v = [cell(reg, i, nm, gen, q)[key]["max"] for nm in nms for i in ivs
             if cell(reg, i, nm, gen, q)[key] is not None]
        return max(v) if v else None

    def st(reg, iv, nm, key="cov"):
        d = cell(reg, iv, nm)
        s = d[key]
        return "all abort" if s is None else f"median {_g(s['median'])}, p90 {_g(s['p90'])}, max {_g(s['max'])}"

    def ab(reg, iv, nm, gen="all", q="all"):
        return cell(reg, iv, nm, gen, q)["n_abort"]

    if any(len(S_[reg]["all"]["all"]) < len(RF_VALUES[reg]) for reg in REGIMES):
        return ["(partial sweep, e.g. --quick: key findings not computed)"]
    nmod = S_["D"]["all"]["all"]["0"]["n_models"]
    L = {reg: len(RF_VALUES[reg]) - 1 for reg in REGIMES}
    out = []
    # ---- D
    i = L["D"]
    lc = {nm: cell("D", i, nm)["loc_cov"] for nm in NAMES}
    npers = {nm: sum(cell("D", j, nm)["loc_cov"][k] for j in range(i + 1) for k in ("persistent", "end"))
             for nm in NAMES}
    pers_txt = ("no D covariance error of the six library smoothers is classed persistent or end "
                "at any s" if not any(npers.values()) else
                "covariance errors classed persistent/end (model x value cases): "
                + ", ".join(f"{nm} {v}" for nm, v in npers.items() if v))
    out.append(
        f"D (random families, {nmod} models): at s = 1e10 the raw covariance error is RTS {st('D', i, 'RTS')}; "
        f"BF {st('D', i, 'BF')}; MBF {st('D', i, 'MBF')}; 2F {st('D', i, '2F')}; DWY {st('D', i, 'DWY')}; "
        f"VAR {st('D', i, 'VAR')}. In the interior window the maximum over all s and all models is "
        + ", ".join(f"{nm} {_g(mx('D', [nm], 'cov_mid'))}" for nm in NAMES)
        + f"; {pers_txt} (start-up/persistent at s = 1e10: "
        + ", ".join(f"{nm} {lc[nm]['start-up']}/{lc[nm]['persistent']}" for nm in NAMES)
        + "). Aborts at "
        f"s = 1e10: 2F {ab('D', i, '2F')}, DWY {ab('D', i, 'DWY')}; at 1e8: 2F {ab('D', i - 1, '2F')}, DWY "
        f"{ab('D', i - 1, 'DWY')}; at s <= 1e6: 2F {sum(ab('D', j, '2F') for j in range(i - 1))}, DWY "
        f"{sum(ab('D', j, 'DWY') for j in range(i - 1))}; RTS, BF, "
        f"MBF and VAR abort on {sum(ab('D', j, nm) for j in range(i + 1) for nm in ('RTS', 'BF', 'MBF', 'VAR'))} "
        f"model x value cases in D.")
    # ---- S
    i = L["S"]
    d = cell("S", i, "DWY")
    okj = [cell("S", j, "DWY")["n_ran"] > 0 and cell("S", j, "DWY")["loc_cov"]["persistent"]
           >= 0.95 * cell("S", j, "DWY")["n_ran"] for j in range(i + 1)]
    j0 = next((j for j in range(i + 1) if all(okj[j:])), None)
    first80 = RF_VALUES["S"][j0] if j0 is not None else float("nan")
    v = cell("S", i, "VAR")
    out.append(
        f"S (random families, {nmod} models): DWY's covariance error is persistent on "
        f"{d['loc_cov']['persistent']} of the {d['n_ran']} models on which it runs at eps = 1e-10 "
        f"({st('S', i, 'DWY')}; interior {st('S', i, 'DWY', 'cov_mid')}), and on >= 95% of the models from "
        f"eps = {first80:g} on. VAR at eps = 1e-10: {st('S', i, 'VAR')}, "
        f"start-up on {v['loc_cov']['start-up']} and persistent on {v['loc_cov']['persistent']} of the models "
        f"(interior {st('S', i, 'VAR', 'cov_mid')}). 2F: {st('S', i, '2F')} (start-up on "
        f"{cell('S', i, '2F')['loc_cov']['start-up']}, persistent on {cell('S', i, '2F')['loc_cov']['persistent']}). "
        f"RTS {st('S', i, 'RTS')}; BF {st('S', i, 'BF')}; MBF {st('S', i, 'MBF')}. Aborts in S: 2F "
        f"{sum(ab('S', j, '2F') for j in range(i + 1))}, DWY {sum(ab('S', j, 'DWY') for j in range(i + 1))} "
        f"model x value cases; the other smoothers "
        f"{sum(ab('S', j, nm) for j in range(i + 1) for nm in ('RTS', 'BF', 'MBF', 'VAR'))}.")
    # models with persistent RTS/BF/MBF error in S
    per = sorted({(c["gen"], c["q"], c["model_id"]) for c in rc if c["regime"] == "S"
                  and any(c["smoothers"][nm].get("loc_cov") == "persistent" for nm in ("RTS", "BF", "MBF"))})
    if per:
        cs = [c for c in rc if c["regime"] == "S" and (c["gen"], c["q"], c["model_id"]) in per and c["index"] == i]
        out.append("S, the only random models on which RTS/BF/MBF covariance errors are persistent: "
                   + "; ".join(f"{c['gen']} q={c['q']} #{c['model_id']} (at eps=1e-10: RTS "
                               f"{_g(c['smoothers']['RTS'].get('cov_mid'))}, BF {_g(c['smoothers']['BF'].get('cov_mid'))}, "
                               f"MBF {_g(c['smoothers']['MBF'].get('cov_mid'))} interior; exact cond P^xx_n|n median "
                               f"{c['cond_exact']['Pf[2F]'][1]:.1e})" for c in cs) + ".")
    # ---- C
    i = L["C"]
    vv = [cell("C", j, "VAR") for j in range(i + 1)]
    crR = [c["model"]["cond_R"] for c in rc if c["regime"] == "C" and c["index"] == i]
    m_gq = results["meta"]["random_families"]["models_per_generator_and_q"]
    abFD = {nm: [(gen, q, sum(ab("C", j, nm, gen, q) for j in range(i + 1))) for gen in GENS for q in QS]
            for nm in ("2F", "DWY")}
    ab_txt = "; ".join(f"{nm}: " + ", ".join(f"{g} q={q} {n}" for g, q, n in v) for nm, v in abFD.items())
    # 2F / DWY on the C cases where they complete: largest errors over the whole sweep
    parts = []
    for nm in ("2F", "DWY"):
        okc = [c for c in rc if c["regime"] == "C" and "cov" in c["smoothers"][nm]]
        wc = max(okc, key=lambda c: c["smoothers"][nm]["cov"])
        wm = max(okc, key=lambda c: max(c["smoothers"][nm]["mean"]))
        parts.append(f"{nm} covariance max {wc['smoothers'][nm]['cov']:.1e} ({wc['gen']} q={wc['q']} "
                     f"#{wc['model_id']}, 1-c = {wc['value']:g}), mean max "
                     f"{max(wm['smoothers'][nm]['mean']):.1e} ({wm['gen']} q={wm['q']} #{wm['model_id']}, "
                     f"1-c = {wm['value']:g})")
    out.append(
        f"C (random families, {nmod} models): VAR's covariance error at 1-c = 1e-8 is {st('C', i - 1, 'VAR')} "
        f"(persistent on {vv[i - 1]['loc_cov']['persistent']} of {vv[i - 1]['n_ran']}); VAR aborts on "
        f"{vv[i - 1]['n_abort']} models at 1e-8 and {vv[i]['n_abort']} at 1e-10 (at 1e-10 it runs on "
        f"{vv[i]['n_ran']}: {st('C', i, 'VAR')}). RTS, BF and MBF: max over all "
        f"1-c and models {_g(mx('C', ['RTS'], 'cov'))}, {_g(mx('C', ['BF'], 'cov'))}, {_g(mx('C', ['MBF'], 'cov'))}, "
        f"aborts {sum(ab('C', j, nm) for j in range(i + 1) for nm in ('RTS', 'BF', 'MBF'))}, while cond R at "
        f"1-c = 1e-10 is {min(crR):.1e}-{max(crR):.1e}. 2F/DWY: aborts over all 1-c (model x value cases, "
        f"of {m_gq * len(RF_VALUES['C'])} per generator and q) {ab_txt}; where they complete, over all 1-c: "
        + "; ".join(parts) + f"; at 1-c = 1e-10: 2F {st('C', i, '2F')}, DWY {st('C', i, 'DWY')}.")
    # ---- M
    i = L["M"]
    d2, dd = cell("M", i, "2F"), cell("M", i, "DWY")
    worst = max((c for c in rc if c["regime"] == "M" and c["index"] == i and "cov" in c["smoothers"]["DWY"]),
                key=lambda c: c["smoothers"]["DWY"]["cov"])
    out.append(
        f"M (random families, {nmod} models): at 1-rho = 1e-4 2F is {st('M', i, '2F')} (persistent on "
        f"{d2['loc_cov']['persistent']} of {d2['n_ran']}), DWY {st('M', i, 'DWY')} (persistent on "
        f"{dd['loc_cov']['persistent']} of {dd['n_ran']}); largest on {worst['gen']} q={worst['q']} "
        f"#{worst['model_id']} (exact cond Sigma_n median {worst['cond_exact']['Sigma[2F,DWY]'][1]:.1e}). "
        f"2F/DWY abort on {ab('M', i, '2F')}/{ab('M', i, 'DWY')} models at 1e-4; all 1-rho together: 2F "
        f"{sum(ab('M', j, '2F') for j in range(i + 1))}, DWY {sum(ab('M', j, 'DWY') for j in range(i + 1))}, "
        f"other smoothers {sum(ab('M', j, nm) for j in range(i + 1) for nm in ('RTS', 'BF', 'MBF', 'VAR'))} "
        f"model x value cases. RTS, BF, MBF, VAR: max over all 1-rho and models "
        + ", ".join(f"{_g(mx('M', [nm], 'cov'))}" for nm in ("RTS", "BF", "MBF", "VAR"))
        + "; interior " + ", ".join(f"{_g(mx('M', [nm], 'cov_mid'))}" for nm in ("RTS", "BF", "MBF", "VAR")) + ".")
    # ---- indefinite raw covariances, every regime
    ind = []
    for reg in REGIMES:
        for j in range(len(RF_VALUES[reg])):
            for nm in NAMES:
                n_i = cell(reg, j, nm)["n_indefinite"]
                if n_i:
                    ind.append(f"{reg} {REGIMES[reg]['param']}={RF_VALUES[reg][j]:g} {nm} {n_i}")
    out.append("Indefinite raw smoothed covariance (number of random models, of "
               f"{nmod}, every case where it occurs): " + ("; ".join(ind) if ind else "none") + ".")
    # ---- means
    out.append(
        "Means (random families): interior-window mean error max over all values and models - "
        + "; ".join(f"{reg}: " + ", ".join(f"{nm} {_g(mx(reg, [nm], 'mean_mid'))}" for nm in NAMES)
                    for reg in REGIMES)
        + ". Whole-record max - " + "; ".join(f"{reg}: " + ", ".join(f"{nm} {_g(mx(reg, [nm], 'mean'))}" for nm in NAMES)
                                              for reg in REGIMES) + ".")
    # ---- kappa (base model) and VAR pivots
    for q in QS:
        rows = _base_rows(results, "C", q)
        if rows:
            c = rows[-1]
            k, dl = c["kappa"].get("VAR", {}), c["cond_exact"].get("Delta[VAR]", {})
            out.append(f"Base model C (2,{q}) at 1-c = 1e-10: VAR cancellation factor kappa max "
                       f"{_g(k.get('max'))} at n = {k.get('argmax')}, median {_g(k.get('median'))}; "
                       f"exact cond Delta_n max {_g(dl.get('max'))}, median {_g(dl.get('median'))}.")
        rows = _base_rows(results, "S", q)
        if rows:
            c = rows[-1]
            k = c["kappa"].get("DWY", {})
            out.append(f"Base model S (2,{q}) at eps = 1e-10: DWY cancellation factor kappa max "
                       f"{_g(k.get('max'))}, median {_g(k.get('median'))}; exact cond of the matrices DWY "
                       f"inverts: Sigma_n <= {_g(c['cond_exact']['Sigma[2F,DWY]']['max'])}, P^b_n-1|n <= "
                       f"{_g(c['cond_exact']['Pb_pred[DWY]']['max'])}.")
    return out


def library_note(results):
    meta = results["meta"]
    lib = meta.get("library") or {}
    sc = meta.get("replicate_selfcheck", [])
    runs = [v for s_ in sc for k, v in s_.items() if k in ("2F", "DWY")]
    n_bit = sum(bool(v.get("bitwise_identical")) for v in runs)
    n_fail = sum("same_failure" in v for v in runs)
    n_same = sum(bool(v.get("same_failure")) for v in runs)
    return (f"Library: awesomepkf at {lib.get('path')} (prg.__version__ {lib.get('prg.__version__')}), git HEAD "
            f"{lib.get('git_head')}, source tree {'with uncommitted changes' if lib.get('dirty') else 'clean'}"
            f" (modified: {', '.join(lib.get('modified_files') or []) or 'none'}). A line-by-line replicate of "
            f"its 2F/DWY backward filter (Sigma_n, Q^b_n, P^b_n-1|n and the conditioned covariances "
            f"symmetrised at every step) reproduces the library bitwise (smoothed means and covariances) on "
            f"{n_bit} of the {len(runs) - n_fail} self-check runs that complete ({len(sc)} cases, 2F and DWY) "
            f"and aborts identically on {n_same} of the {n_fail} that abort.")


def regularisation_note(results):
    """In-place regularisations: script wrapper count vs the library's own record."""
    n, agree, n_ev = 0, 0, 0
    for c in results["cases"]:
        for d in list(c["smoothers"].values()) + list(c["consistency_control"].values()):
            rc = d.get("regularisation_count_check") or {}
            n += 1
            agree += bool(rc.get("agree"))
            n_ev += rc.get("wrapper") or 0
    for c in results["random_cases"]:
        for d in list(c["smoothers"].values()) + list(c["consistency_control"].values()):
            n += 1
            agree += bool(d.get("reg_lib_agree"))
            n_ev += (d.get("reg") or 0) + (d.get("fwd_reg") or 0)
    return (f"In-place covariance regularisations: the count recorded by this script's wrapper equals "
            f"the length of the library's PKF.covariance_regularisations list on {agree} of {n} "
            f"smoother x case records (all seeds; {n_ev} regularisations in total, aborted runs "
            f"included up to the abort).")


def consistency_note(results):
    """2F/DWY of the library vs the 2F*/DWY* control: one factual sentence per regime."""
    cs = results["consistency_summary"]
    parts = []
    for reg in REGIMES:
        sub = []
        for lab, tag in (("random_families", "random"), ("base_model", "base")):
            for base, d in cs[reg][lab].items():
                cd = d["cov_diff"] or {}
                md = d["mean_diff"] or {}
                sub.append(
                    f"{base} {tag}: {d['n_cases']} cases, both complete {d['both_complete']}, both abort "
                    f"{d['both_abort']}, library-only abort {d['library_only_aborts']}, control-only abort "
                    f"{d['control_only_aborts']}; bitwise equal {d['n_bitwise_all_seeds']}; output "
                    f"difference cov median {_g(cd.get('median'))} max {_g(cd.get('max'))}, mean median "
                    f"{_g(md.get('median'))} max {_g(md.get('max'))}; cov difference > {CONS_ROUNDOFF:g} in "
                    f"{d['n_cov_diff_gt_roundoff']} (there, difference / larger error: median "
                    f"{_g((d.get('cov_diff_over_max_error') or {}).get('median'))}), errors vs reference "
                    f"differ by > {CONS_RATIO:g}x in {d['n_error_ratio_gt']} (library more accurate in "
                    f"{d.get('n_library_more_accurate_gt')}, control in {d.get('n_control_more_accurate_gt')})")
        parts.append(f"{reg}: " + "; ".join(sub))
    out = cs["disagreements"]
    worst = cs["largest_differences"][:6]

    def one(x):
        who = f"base (2,{x['q']})" if x["gen"] is None else f"{x['gen']} q={x['q']} #{x['model_id']}"
        if x["abort_status_differs"]:
            return (f"{x['regime']} {who} at {x['value']:g} {x['smoother']}: aborting seeds library "
                    f"{x['abort_library']} / control {x['abort_control']}")
        return (f"{x['regime']} {who} at {x['value']:g} {x['smoother']}: cov error library "
                f"{_g(x['cov_library'])} / control {_g(x['cov_control'])}, mean {_g(x['mean_library'])} / "
                f"{_g(x['mean_control'])}, direct difference cov {_g(x['cov_diff_max'])}")
    ab = [x for x in out if x["abort_status_differs"]]
    return ("Consistency check (not part of the study): the library 2F/DWY against the control used "
            "by the previous version of this script (2F*/DWY*: backward filter with P^b_n-1|n and "
            "P^b_n symmetrised once per step, Sigma_n, Q^b_n and P^y_n as computed), on every case; "
            "output difference = max over seeds of max_n ||P_lib - P_ctrl||_2/||P_ref,n||_2 (cov) and "
            "max_n ||x_lib - x_ctrl||_2/max_n ||x_ref,n||_2 (mean). " + " | ".join(parts)
            + f". {cs['n_beyond_roundoff_or_disagreeing']} smoother x case records differ by more than "
            f"{CONS_ROUNDOFF:g} or in abort status; where the outputs differ by more than {CONS_ROUNDOFF:g} "
            f"the difference is of the order of the errors themselves (ratio to the larger error capped at 2 "
            f"by the triangle inequality), i.e. both outputs are inaccurate there by similar amounts. Abort "
            f"status differs on {len(ab)}"
            + (" (" + "; ".join(one(x) for x in ab[:8]) + (" ..." if len(ab) > 8 else "") + ")" if ab else "")
            + ("; largest output differences: " + "; ".join(one(x) for x in worst) if worst else "") + ".")


def build_notes(results):
    """Summary notes and suggested captions. Every statement is generated from the data and
    scoped to what was measured (base model vs random families, with counts)."""
    meta = results["meta"]
    rf = meta["random_families"]
    nmod = rf["models_per_generator_and_q"]
    notes = [
        f"Scope. Base model: one fixed (A0, R0) per (p,q) in {{(2,1),(2,2)}}, {meta['seeds']} seeds per "
        f"sweep value. Random families: generators G1 and G2, {nmod} models per (generator, q), "
        f"{rf['seeds']} seeds per model, every regime ({rf['n_cases']} model x value cases). Every number "
        "below is a median / 90th percentile / max over these sets only; nothing is claimed beyond them.",
        library_note(results),
    ]
    xc = results["crosscheck"] + results.get("crosscheck_random", [])
    worst = max(max(c["kf_rts_N400"]["mean"], c["kf_rts_N400"]["cov"], c["kf_rts_N400"]["lag1"] or 0)
                for c in xc)
    worst_d = max(max(c[f"dense_conditioning_N{N_DENSE}"]["mean"], c[f"dense_conditioning_N{N_DENSE}"]["cov"])
                  for c in xc)
    notes.append(f"Reference. Over {len(xc)} cross-checked cases (base: most extreme value of each regime "
                 f"and q; random: worst model per regime, generator and q), the 60-digit lifted reference "
                 f"and an independent 60-digit KF+RTS agree to {worst:.1e} (relative, mean / covariance / "
                 f"lag-one), and dense conditioning on N={N_DENSE} to {worst_d:.1e}.")
    notes.extend(key_findings(results))
    notes.append(mean_vs_cov_note(results))
    notes.append("Abort causes (all aborts of the run, six library smoothers): "
                 + (" | ".join(abort_cause_notes(results)) or "none"))
    for blk in results.get("scale_control", []):
        f = blk.get("failure_at_k_star") or {}
        if blk["k_star_first_abort"] is None:
            notes.append(f"Control U (2,{blk['q']}): (R, Pz0) scaled by 2^-k, record by 2^-k/2, k = "
                         f"{SCALE_EXPS[0]}..{SCALE_EXPS[-1]}: no smoother aborts at any k; outputs "
                         f"(rescaled back) deviate from k = 0 by at most {_g(blk['max_dev_even_k_running'])} "
                         f"(even k, exact power-of-two scaling) and {_g(blk['max_dev_odd_k_running'])} (odd k).")
        else:
            cs_ = classify_abort(f) if f else {}
            notes.append(f"Control U (2,{blk['q']}): first abort at k* = {blk['k_star_first_abort']}: "
                         f"{f.get('type')} on {f.get('matrix')}, check {cs_.get('check')}; tolerance "
                         f"artefact {cs_.get('tolerance_artefact')}. Runs that complete deviate from k = 0 "
                         f"by {_g(blk['max_dev_even_k_running'])} (even k) and "
                         f"{_g(blk['max_dev_odd_k_running'])} (odd k).")
    notes.append(consistency_note(results))
    notes.append(regularisation_note(results))
    for reg in REGIMES:
        notes.append(regime_note_base(results, reg))
        notes.append(regime_note_random(results, reg))
    return notes, build_caption(results), build_caption_letter(results)


def build_caption(results):
    rf = results["meta"]["random_families"]
    m = rf["models_per_generator_and_q"]
    n_all = 2 * len(QS) * m
    return (f"Accuracy of the six library smoothers against a 60-digit reference, N=400, on "
            f"{n_all} random models per regime ({m} models from each of two generators, G1 and G2, "
            f"at each of (p,q)=(2,1) and (2,2); {rf['seeds']} simulated records per model). "
            f"Covariance error: max_n ||P_n-P_ref,n||_2/||P_ref,n||_2 of the raw output (before the "
            f"library's in-place regularisation), over the whole record (first column) and over "
            f"{MID[0]}≤n≤{MID[1]} (second column); an error present in the first column but not in "
            f"the second lies outside that window (start-up or end of record). Mean error (third "
            f"column): max_n ||x̂_n-x_ref,n||_2/max_n ||x_ref,n||_2 over the whole record (max over "
            f"the {rf['seeds']} records of each model); this column does not separate start-up from "
            f"interior errors. Line: median over the models on which the "
            f"smoother ran (not drawn where fewer than half ran); open marker: max over them. Fourth "
            f"column (same x axis): number of models (of {n_all}, square-root scale) on which the "
            f"smoother aborts (×) or returns an indefinite raw covariance (○, dashed). "
            f"Grey line: unit roundoff 1.1e-16.")


def print_notes(results):
    print("\n=== Summary notes (also in the JSON) ===")
    for s in results.get("summary_notes", []):
        print("- " + s)
    print("\n=== Suggested caption (figures/conditioning_exact) ===")
    print(results.get("suggested_caption", ""))
    print("\n=== Suggested caption (figures/conditioning_letter) ===")
    print(results.get("suggested_caption_letter", ""))


# ----------------------------------------------------------------------------------
def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        if np.isnan(f):
            return None
        if np.isinf(f):
            return "inf" if f > 0 else "-inf"
        return f
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


DEFINITIONS = {
    "differences": "every error against the reference is formed as (estimate - hi) - lo "
                   "with (hi, lo) the double-double split of the 60-digit value, so "
                   "roundoff-level errors are resolved to a few percent",
    "mean_err": "max_n ||xhat_n - xref_n||_2 / max_n ||xref_n||_2",
    "_mid": f"same maximum restricted to the interior window n in [{MID[0]},{MID[1]}] "
            "(normalisation unchanged)",
    "cov_err_raw": "max_n ||P_n - Pref_n||_2/||Pref_n||_2, P_n captured before the "
                   "library's in-place regularisation",
    "cov_err_returned": "same, on the covariance the library returns",
    "lag1_err": "max_n ||C_n - Cref_n||_2/sqrt(||Pref_n|| ||Pref_{n+1}||), "
                "C_n = Cov(x_{n+1},x_n|y): RTS from Gk_smooth, VAR from Mk_smooth",
    "cond_exact": "2-norm condition numbers of the EXACT matrices each smoother "
                  "inverts (60-digit values rounded to float64), max/median/argmax "
                  "over n; seed-independent",
    "cond_float": "same for the float64 matrices the library forms (a diverged "
                  "float recursion shows up here, not in cond_exact); base model only",
    "intermediate_err": "max_n ||float - exact||_2/||exact||_2 of the library's "
                        "intermediate covariances; uses the exact value rounded to float64, so "
                        "entries below ~1e-15 are resolution-limited",
    "kappa": "(exact intermediates) max AND median over n of ||subtracted term||_2/||result||_2 "
             "for the covariance updates that are differences (RTS: Pf M'P_{n+1|n}^-1 M Pf; BF: "
             "P^xx_{n|n-1}-P_{n|N}; MBF: Pf Lambda Pf = Pf-P_{n|N}; DWY: D Pb_pred D'; 2F: "
             "||(P^y_n)^-1|| ||P_{n|N}||; VAR: Thomas pivot subtraction). A large max with a "
             "small median is a boundary effect (e.g. VAR in C: max at n = N)",
    "Delta_roundoff_ratio": "max_n ||Delta_n(float) - Delta_n(exact)||_2 / "
                            "lambda_min(Delta_n(exact)) for the float replicate of VAR's pivot "
                            "recursion (library operation order) and for a variant using "
                            "np.linalg.inv for Delta^-1; >= 1 means the pivot's definiteness is "
                            "decided by rounding",
    "localisation": "per smoother and error: class (small / start-up / end / persistent), "
                    "argmax n of the worst seed, max and interior max. start-up / end are location "
                    "labels (argmax n < 100 / > 300 with a small interior error), not causes: in S, "
                    "C and M the prior is stationary, so an error located at n < 100 there is not a "
                    "prior transient",
    "abort_cause": "library check behind each abort: InvertibleMatrix test (which one), "
                   "Cholesky failure (scipy cho_factor) or covariance check; tolerance_artefact = "
                   "fixed-threshold test failed on a float64 matrix that is Cholesky-"
                   "factorisable; diagnosis_code compares the failing float64 matrix with its "
                   "exact counterpart (exact_singular: exact cond >= 1e15; float_off: float "
                   "rel. err >= 1 while the exact matrix is PD; rounding_vs_eigengap: rel. err x "
                   "exact cond >= 0.1; threshold_on_wellposed; other)",
    "consistency_control": "per case, for 2F and DWY: the control 2F*/DWY* (library classes "
                           "with the backward filter used as a control by the previous version "
                           "of this script: 2.15.1 filter + one symmetrisation of P^b_{n-1|n} and "
                           "P^b_n per step; swapped in in-process) with its errors in the same "
                           "format as the smoothers, and stock_vs_control = aborting seeds of "
                           "each, seeds where both complete, seeds bitwise equal, max over seeds "
                           "of max_n ||x_lib - x_ctrl||/max_n ||x_ref|| (mean_diff_max) and of "
                           "max_n ||P_lib - P_ctrl||_2/||P_ref,n||_2 (cov_diff_max, raw covariances)",
    "regularisation_count_check": "number of in-place covariance regularisations seen by the "
                                  "script's wrapper vs the length of the library's "
                                  "PKF.covariance_regularisations list (all seeds)",
}


def replicate_selfcheck(task):
    """The "v2160" replicate of the backward filter must reproduce the library bitwise (2F
    and DWY smoothed means and covariances): it documents what the library's backward filter
    does in the version used, and it is the base of the "control" mode (which differs from it
    only by the symmetrisations listed in backward_filter_replicate)."""
    regime, q, value, gen, model_id = task
    model = build_model(regime, q, value, gen, model_id)
    param = make_param(model, q)
    X, Y = simulate(model, q, 0)
    res = {"regime": regime, "q": q, "value": value, "gen": gen, "model_id": model_id}
    for name in ("2F", "DWY"):
        s1, e1 = run_library(SMOOTHERS[name], param, X, Y, q)
        s2, e2 = run_library(SMOOTHERS[name], param, X, Y, q, bfilter=_v2160_backward_filter)
        if e1 is not None or e2 is not None:
            res[name] = {"stock_error": e1 and e1["matrix"], "replicate_error": e2 and e2["matrix"],
                         "same_failure": (e1 and (e1["type"], e1["matrix"], e1["step"])) ==
                                         (e2 and (e2["type"], e2["matrix"], e2["step"]))}
            continue
        a = [np.array(h["Xkp1_smooth"]) for h in s1.history]
        b = [np.array(h["Xkp1_smooth"]) for h in s2.history]
        Pa = [np.array(h["PXXkp1_smooth"]) for h in s1.history]
        Pb_ = [np.array(h["PXXkp1_smooth"]) for h in s2.history]
        res[name] = {"bitwise_identical": bool(all(np.array_equal(x, y) for x, y in zip(a, b))
                                               and all(np.array_equal(x, y) for x, y in zip(Pa, Pb_)))}
    return res


def pick_random_crosschecks(rc):
    """Per (regime, generator, q): the model with the largest covariance error of any of
    the six library smoothers at the most extreme sweep value."""
    out = []
    for reg in REGIMES:
        iv = len(RF_VALUES[reg]) - 1
        for gen in GENS:
            for q in QS:
                at = [c for c in rc if c["regime"] == reg and c["gen"] == gen and c["q"] == q
                      and c["index"] == iv]
                if not at:
                    continue
                def worst(c):
                    v = [c["smoothers"][nm]["cov"] for nm in NAMES if "cov" in c["smoothers"][nm]]
                    return max(v) if v else -1.0
                c = max(sorted(at, key=lambda c: c["model_id"]), key=worst)
                out.append((reg, q, c["value"], gen, c["model_id"]))
    return out


def write_json(results, path):
    """indent=1 everywhere except random_cases: one compact line per case (size)."""
    rest = {k: v for k, v in results.items() if k != "random_cases"}
    txt = json.dumps(rest, indent=1)
    rc = ",\n".join(json.dumps(c, separators=(",", ":")) for c in results["random_cases"])
    txt = txt[:txt.rindex("}")].rstrip() + ',\n "random_cases": [\n' + rc + "\n ]\n}\n"
    json.loads(txt)   # sanity
    Path(path).write_text(txt)


def _round_sig(o, sig=4):
    """Round every float in a JSON-like structure to `sig` significant digits (size)."""
    if isinstance(o, dict):
        return {k: _round_sig(v, sig) for k, v in o.items()}
    if isinstance(o, list):
        return [_round_sig(v, sig) for v in o]
    if isinstance(o, float):
        return float(f"{o:.{sig}g}")
    return o


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=int, default=10, help="seeds per base-model case")
    ap.add_argument("--rf-models", type=int, default=RF_MODELS,
                    help="random models per generator and q")
    ap.add_argument("--rf-seeds", type=int, default=RF_SEEDS, help="seeds per random-family case")
    ap.add_argument("--workers", type=int, default=min(10, os.cpu_count() or 1))
    ap.add_argument("--quick", action="store_true",
                    help="smoke test: 2 seeds, 2 values per regime, 2 random models")
    ap.add_argument("--no-figure", action="store_true")
    ap.add_argument("--letter-fig", action="store_true",
                    help="also draw the compact letter figure (figures/conditioning_letter)")
    ap.add_argument("--letter-layout", choices=["2x2wide", "1x4", "2x2"], default="2x2wide",
                    help="letter figure layout: full page width as 2x2 (default) or one row "
                         "(1x4), or one column (2x2)")
    ap.add_argument("--letter-fig-out", default=str(OUT_LETTER_FIG),
                    help="letter figure path without extension")
    ap.add_argument("--from-json", action="store_true",
                    help="reprint / redraw / rebuild the notes from the saved JSON")
    ap.add_argument("--fig-q", type=int, default=1, help="q of the optional base-model figure")
    ap.add_argument("--out", default=str(OUT_JSON))
    ap.add_argument("--fig-out", default=str(OUT_FIG), help="figure path without extension")
    ap.add_argument("--base-fig-out", default=None,
                    help="optional base-model figure path without extension (scratch use)")
    args = ap.parse_args()

    if args.from_json:
        results = json.loads(Path(args.out).read_text())
    else:
        t0 = time.time()
        n_seeds = 2 if args.quick else args.seeds
        n_rf = 2 if args.quick else args.rf_models
        rf_seeds = min(args.rf_seeds, n_seeds)
        tasks = []
        for reg, spec in REGIMES.items():
            vals = spec["values"]
            idx = [0, len(vals) - 1] if args.quick else range(len(vals))
            for q in QS:
                for i in idx:
                    tasks.append((reg, q, i, vals[i], n_seeds, None, None, False))
        rf_tasks = []
        for reg in REGIMES:
            vals = RF_VALUES[reg]
            idx = [0, len(vals) - 1] if args.quick else range(len(vals))
            for gen in GENS:
                for q in QS:
                    for j in range(n_rf):
                        for i in idx:
                            rf_tasks.append((reg, q, i, vals[i], rf_seeds, gen, j, True))
        xtasks = [(reg, q, spec["values"][-1], None, None) for reg, spec in REGIMES.items() for q in QS]
        ks = [0, 25, 26, 51] if args.quick else SCALE_EXPS
        stasks = [(q, k) for q in QS for k in ks]
        rtasks = [("C", 1, 1e-1, None, None), ("C", 1, 1e-2, None, None), ("S", 2, 1e-10, None, None),
                  ("D", 1, 1e8, None, None), ("M", 2, 1e-4, "G2", 0), ("C", 2, 1e-6, "G1", 1),
                  ("C", 1, 1e-4, "G2", 4)]
        lib_start = library_info()
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            fc = [ex.submit(run_case, t) for t in tasks]          # longest tasks first
            fr = [ex.submit(run_case, t) for t in rf_tasks]
            fx = [ex.submit(crosscheck_case, t) for t in xtasks]
            fs = [ex.submit(scale_probe, t) for t in stasks]
            fk = [ex.submit(replicate_selfcheck, t) for t in rtasks]
            cases = [f.result() for f in fc]
            t_base = time.time() - t0
            rcases = [f.result() for f in fr]
            t_rf = time.time() - t0
            fx2 = [ex.submit(crosscheck_case, t) for t in pick_random_crosschecks(rcases)]
            cross = [f.result() for f in fx]
            cross_rf = [f.result() for f in fx2]
            scale = assemble_scale_control([f.result() for f in fs])
            selfcheck = [f.result() for f in fk]
        import matplotlib
        import scipy
        results = {
            "meta": {
                "script": "experiments/conditioning_exact.py",
                "date": datetime.now().isoformat(timespec="seconds"),
                "N": N, "p": P, "qs": list(QS), "seeds": n_seeds, "mp_dps": DPS,
                "interior_window": list(MID),
                "localisation_rule": f"small if max <= {LOC_THR:g}; persistent if interior-window "
                                     f"error >= {LOC_RATIO:g} x max or >= {LOC_ABS:g} (absolute); "
                                     f"else start-up (argmax n < {MID[0]}) or end (argmax n > "
                                     f"{MID[1]})",
                "library": lib_start,
                "library_unchanged_during_run": (
                    library_info().get("git_diff_prg_sha256") == lib_start.get("git_diff_prg_sha256")
                    and lib_start.get("git_diff_prg_sha256") is not None),
                "consistency_rule": f"library 2F/DWY vs control 2F*/DWY*: output difference > "
                                    f"{CONS_ROUNDOFF:g} counted as beyond round-off; errors vs the "
                                    f"reference 'differ' when their ratio exceeds {CONS_RATIO:g}",
                "mean_vs_cov_rule": f"mean and covariance errors (pooled median and max) 'differ' "
                                    f"when their ratio exceeds {MEAN_COV_FACTOR:g} and the larger "
                                    f"exceeds {MEAN_COV_FLOOR:g}",
                "random_families": {
                    "generators": {g: GENERATORS[g].__doc__ for g in GENS},
                    "models_per_generator_and_q": n_rf, "seeds": rf_seeds,
                    "values": {k: v for k, v in RF_VALUES.items()},
                    "n_cases": len(rf_tasks),
                    "statistics": "per model: max over its seeds of each error (covariances are "
                                  "seed-independent); per cell: median / 90th percentile (numpy "
                                  "linear interpolation) / max over the models on which the "
                                  "smoother ran; aborting models are counted, not included",
                    "reductions": f"{rf_seeds} seeds per random-family case ({n_seeds} for the base "
                                  "model); 6 sweep values per regime (regime C starts at 1-c = 1e-1, "
                                  "the base model at 0.5); no dense cond(J), float-matrix condition "
                                  "numbers, cancellation factors or VAR inv-variant in the compact "
                                  "random-family record (exact condition numbers and float-vs-exact "
                                  "P^b are kept)"},
                "consistency_control": {"2F*": "library Linear_PKS_MF with the control backward "
                                               "filter (and the 2.15.1 _cond_xy for P^y_n) swapped "
                                               "in (in-process)",
                                        "DWY*": "library Linear_PKS_DWY with the control backward "
                                                "filter swapped in (in-process)"},
                "replicate_selfcheck": selfcheck,
                "quick": bool(args.quick),
                "runtime_s": {"total": time.time() - t0, "base_done_at": t_base,
                              "random_families_done_at": t_rf},
                "workers": args.workers,
                "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                             "scipy": scipy.__version__, "mpmath": mp.__version__,
                             "matplotlib": matplotlib.__version__,
                             "prg_path": str(Path(sys.modules["prg"].__file__).parent)},
                "regimes": {k: {kk: vv for kk, vv in v.items() if kk != "xlabel"}
                            for k, v in REGIMES.items()},
                "base": {"A0": {str(q): A0[q].tolist() for q in QS},
                         "Cxx": CXX.tolist(), "Cyy": {str(q): CYY[q].tolist() for q in QS},
                         "K": {str(q): KDIR[q].tolist() for q in QS}, "c_base": C_BASE,
                         "R0": {str(q): noise_matrix(q).tolist() for q in QS},
                         "rho_A0": {str(q): spectral_radius(A0[q]) for q in QS},
                         "henrici_departure_rel_A0": {str(q): henrici(A0[q]) for q in QS}},
                "definitions": DEFINITIONS,
            },
            "cases": cases, "random_cases": rcases, "crosscheck": cross,
            "crosscheck_random": cross_rf, "scale_control": scale,
        }
        results = _clean(results)
    results["random_cases"] = _round_sig(results["random_cases"])   # compact records: 4 digits
    results["random_summary"] = _round_sig(_clean(summarise_random(results["random_cases"])))
    results["consistency_summary"] = _round_sig(_clean(consistency_summary(results)))
    results["mean_vs_cov"] = _clean(mean_vs_cov_groups(results))
    fam, lfam = None, None
    if not args.no_figure:
        fam = make_figure(results, args.fig_out)
        if args.base_fig_out:
            make_figure_base(results, args.fig_q, args.base_fig_out)
    if args.letter_fig:
        lfam, lsize = make_letter_figure(results, args.letter_fig_out, layout=args.letter_layout)
        results.setdefault("figure", {})["letter_size_in"] = [round(v, 3) for v in lsize]
    notes, caption, caption_letter = build_notes(results)
    results["summary_notes"] = notes
    results["suggested_caption"] = caption
    results["suggested_caption_letter"] = caption_letter
    results = _clean(results)
    write_json(results, args.out)
    print(f"saved {args.out}  (runtime {results['meta']['runtime_s']['total']:.0f} s)")
    print_summary(results)
    print_random_families(results)
    print_notes(results)
    if fam is not None:
        print(f"saved {args.fig_out}.pdf/.png (font {fam})")
        if args.base_fig_out:
            print(f"saved {args.base_fig_out}.pdf/.png")
    if lfam is not None:
        print(f"saved {args.letter_fig_out}.pdf/.png (font {lfam}, {lsize[0]:.2f} x {lsize[1]:.2f} in)")

if __name__ == "__main__":
    main()
