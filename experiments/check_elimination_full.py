#!/usr/bin/env python
r"""Reproducible numerical check of Proposition 1 (letter Sec. III) on random GPMCs.

What is checked
---------------
Random stable Gaussian pairwise Markov chains with FULL pairwise structure
(A^{xy} != 0, A^{yy} != 0, R^{xy} != 0), (p, q) in {1,2,3,4} x {1,2}, several seeds,
both TIME-INVARIANT (A_n = A, R_n = R) and TIME-VARYING (a different stable A_n and a
different positive-definite R_n drawn at every step).  Conventions (letter, eq. (1)):

    Z_n = A_n Z_{n-1} + W_n,  W_n ~ N(0, R_n),  n = 1..N,   Z_0 ~ N(m_0, P_0),
    M_n = A_n[:, :p] (columns-X block),  E = [I_p; 0],  F = [0; I_q],
    c_n = F y_n - A_n[:, p:] y_{n-1} = ( -A^{xy}_n y_{n-1} ; y_n - A^{yy}_n y_{n-1} ).

The reference is the exact posterior p(x_{0:N} | y_{0:N}): J and eta are read off the
joint precision of z_{0:N} (forward factorisation), and are themselves cross-checked
against plain Gaussian conditioning of the joint prior covariance of z_{0:N}.

Mapping to the letter: script rows (a), (b), (c) -> Proposition 1(a) (top-down pivots,
RTS multiplier, MBF coordinates); (d) -> Proposition 1(b) (DWY, bottom-up);
(e) -> Proposition 1(c) (2F, twisted factorization).

Identities (rows of the printed table; "gate" rows must hold to round-off):
  J      explicit time-varying blocks of J and eta; J from the BACKWARD factorisation;
         explicit blocks in backward-chain quantities; the prior-chain split identity.
  (a)    top-down pivots  Delta_n = (P^xx_{n|n})^-1 + M_{n+1}' R_{n+1}^-1 M_{n+1}  (n<N),
         Delta_N = (P^xx_{N|N})^-1, and the modified RHS; the naive (M_n, R_n)
         indexing is reported (info only).
  (b)    back-substitution multiplier  -Delta_n^-1 J_{n,n+1} = G_n E, the full gain
         Delta_n^-1 M_{n+1}' R_{n+1}^-1 = G_n, Delta_n^-1 = P^xx_{n|n} - G_n P_{n+1|n} G_n',
         and the RTS mean/covariance recursions (propagated from n = N).
  (c)    MBF adjoint recursion (as implemented in awesomepkf _mbf_backward_pass, written
         time-varying): lambda_n = (P^xx_{n|n})^-1 (x_{n|N} - x_{n|n}), Lambda_n, and the
         couple adjoint (mu_n, Gamma_n); against the LIBRARY's Linear_PKS_MBF internals
         (captured with sys.settrace, nothing is modified) and outputs on the
         time-invariant models.
  (d)    bottom-up pivots Delta^b_n = (P^b_n)^-1 + (M^b_{n-1})' (Q^b_{n-1})^-1 M^b_{n-1}
         (n>0), Delta^b_0 = (P^b_0)^-1; bottom-up RHS; DWY multiplier and recursions;
         Delta^b_n = U_n + J^-_n with U_n = (P^b_n)^-1 - (P^y_n)^-1.
  (e)    twisted identity (P^xx_{n|N})^-1 = Delta_n + Delta^b_n - J_nn, the 2F fusion,
         and their information-vector counterparts.

Residuals are relative and sequence-normalised:  max_n ||X_n - Y_n||_max / max_n ||Y_n||_max.
Every identity is evaluated in float64 on all cases and, unless --no-mp, re-evaluated in
50-digit mpmath arithmetic on time-varying cases (same model, same data), which separates
an exact identity (residual ~1e-48) from a numerical coincidence.

Exit status: 0 if every gated residual is <= 1e-10, 1 otherwise, 2 if awesomepkf is
not importable (unless --no-lib).

Run (from anywhere):
    PYTHONPATH=/path/to/awesomePKF /path/to/awesomePKF/.venv/bin/python -B \
        experiments/check_elimination_full.py
"""
from __future__ import annotations

import argparse
import inspect
import logging
import sys
import time

import numpy as np

TOL = 1e-10
PQ = [(p, q) for p in (1, 2, 3, 4) for q in (1, 2)]


# ======================================================================
# Arithmetic back-ends (float64 / mpmath on numpy object arrays)
# ======================================================================
class FloatBK:
    name = "float64"

    def arr(self, x):
        return np.array(x, dtype=float)

    def zeros(self, shape):
        return np.zeros(shape)

    def eye(self, n):
        return np.eye(n)

    def inv(self, M):
        return np.linalg.inv(M)


class MpBK:
    def __init__(self, dps: int):
        import mpmath
        self.mp = mpmath.mp
        self.mp.dps = dps
        self.mpf = mpmath.mpf
        self.name = f"mp{dps}"

    def arr(self, x):
        x = np.asarray(x, dtype=float)
        out = np.empty(x.shape, dtype=object)
        for idx, v in np.ndenumerate(x):
            out[idx] = self.mpf(float(v))      # binary64 values are exact in mpf
        return out

    def zeros(self, shape):
        out = np.empty(shape, dtype=object)
        out.fill(self.mpf(0))
        return out

    def eye(self, n):
        out = self.zeros((n, n))
        for i in range(n):
            out[i, i] = self.mpf(1)
        return out

    def inv(self, M):
        Mi = self.mp.inverse(self.mp.matrix(M.tolist()))
        return np.array(Mi.tolist(), dtype=object)


def _amax(X) -> float:
    return float(np.max(np.abs(np.asarray(X))))


def rel(X, Y) -> float:
    """Sequence-normalised relative residual max_n|X_n-Y_n| / max_n|Y_n|."""
    if not isinstance(X, (list, tuple)):
        X, Y = [X], [Y]
    num = max(_amax(np.asarray(x) - np.asarray(y)) for x, y in zip(X, Y, strict=True))
    den = max(_amax(y) for y in Y)
    return num / den if den > 0 else num


# ======================================================================
# Random GPMC with full pairwise structure + simulation (float64)
# ======================================================================
def random_case(p: int, q: int, N: int, seed: int, tv: bool) -> dict:
    d = p + q
    rng = np.random.default_rng([seed, p, q, int(tv), 20261007])

    # "Full pairwise structure": every off-diagonal block is present with a non-negligible
    # weight (redraw otherwise) -- back-action A^xy, observation memory A^yy, A^yx, and
    # noise correlation R^xy.  This only guarantees that every term of the identities is
    # exercised; it does not affect what is being verified.
    def draw_A():
        while True:
            G = rng.normal(size=(d, d))
            G *= rng.uniform(0.5, 0.95) / np.max(np.abs(np.linalg.eigvals(G)))
            nA = np.linalg.norm(G)
            if min(np.linalg.norm(G[:p, p:]), np.linalg.norm(G[p:, :p]),
                   np.linalg.norm(G[p:, p:])) >= 0.15 * nA:
                return G

    def draw_R():
        while True:
            C = rng.normal(size=(d, d))
            Rm = C @ C.T / d + 0.2 * np.eye(d)
            sd = np.sqrt(np.diag(Rm))
            if np.max(np.abs(Rm[:p, p:] / np.outer(sd[:p], sd[p:]))) >= 0.2:
                return Rm

    if tv:
        A = [None] + [draw_A() for _ in range(N)]
        R = [None] + [draw_R() for _ in range(N)]
    else:
        A1, R1 = draw_A(), draw_R()
        A = [None] + [A1] * N
        R = [None] + [R1] * N
    m0 = rng.normal(size=d)
    C0 = rng.normal(size=(d, d))
    P0 = C0 @ C0.T / d + 0.5 * np.eye(d)
    z = [m0 + np.linalg.cholesky(P0) @ rng.normal(size=d)]
    for n in range(1, N + 1):
        z.append(A[n] @ z[-1] + np.linalg.cholesky(R[n]) @ rng.normal(size=d))
    return {"p": p, "q": q, "N": N, "seed": seed, "tv": tv, "A": A, "R": R,
            "m0": m0, "P0": P0, "x": [v[:p] for v in z], "y": [v[p:] for v in z]}


# ======================================================================
# All identities for one case, in a given arithmetic
# ======================================================================
def analyse(case: dict, bk) -> tuple[dict, dict]:
    p, q, N = case["p"], case["q"], case["N"]
    d = p + q
    inv, Z, I = bk.inv, bk.zeros, bk.eye
    A = [None] + [bk.arr(a) for a in case["A"][1:]]
    R = [None] + [bk.arr(r) for r in case["R"][1:]]
    m0, P0 = bk.arr(case["m0"]), bk.arr(case["P0"])
    y = [bk.arr(v) for v in case["y"]]
    ix, iy = slice(0, p), slice(p, d)
    E = Z((d, p)); E[:p, :] = I(p)
    F = Z((d, q)); F[p:, :] = I(q)
    M = [None] + [A[n][:, :p] for n in range(1, N + 1)]
    Ri = [None] + [inv(R[n]) for n in range(1, N + 1)]
    c = [None] + [F @ y[n] - A[n][:, p:] @ y[n - 1] for n in range(1, N + 1)]
    cat = lambda a, b: np.concatenate([a, b])

    def bd(Pxx):                       # blkdiag(Pxx, 0_q)
        out = Z((d, d)); out[:p, :p] = Pxx
        return out

    def cond_on_y(m, P, yy):
        S = P[iy, iy]; K = P[ix, iy] @ inv(S)
        return m[ix] + K @ (yy - m[iy]), P[ix, ix] - K @ P[iy, ix], S, K

    blk = lambda X, i, j: X[p * i:p * (i + 1), p * j:p * (j + 1)]
    seg = lambda v, i: v[p * i:p * (i + 1)]
    res: dict = {}

    # ---------- exact reference: joint precision of z_{0:N} (forward factorisation) ----
    D = d * (N + 1)
    Lam, h = Z((D, D)), Z(D)
    P0i = inv(P0)
    Lam[:d, :d] += P0i; h[:d] += P0i @ m0
    for n in range(1, N + 1):
        a, b = slice(d * (n - 1), d * n), slice(d * n, d * (n + 1))
        Lam[b, b] += Ri[n]; Lam[a, a] += A[n].T @ Ri[n] @ A[n]
        Lam[a, b] -= A[n].T @ Ri[n]; Lam[b, a] -= Ri[n] @ A[n]
    xi = np.concatenate([np.arange(d * n, d * n + p) for n in range(N + 1)])
    yi = np.concatenate([np.arange(d * n + p, d * (n + 1)) for n in range(N + 1)])
    yv = np.concatenate(y)
    J = Lam[np.ix_(xi, xi)]
    eta = h[xi] - Lam[np.ix_(xi, yi)] @ yv
    Jinv = inv(J)
    xh = Jinv @ eta
    xs = [seg(xh, n) for n in range(N + 1)]                  # x_{n|N}
    Ps = [blk(Jinv, n, n) for n in range(N + 1)]              # P^xx_{n|N}

    # independent route: condition the joint PRIOR covariance of z_{0:N} on y_{0:N}
    mpri, Sig = [m0], [P0]
    for n in range(1, N + 1):
        mpri.append(A[n] @ mpri[-1]); Sig.append(A[n] @ Sig[-1] @ A[n].T + R[n])
    C = Z((D, D))
    for n in range(N + 1):
        C[d * n:d * (n + 1), d * n:d * (n + 1)] = Sig[n]
        for m in range(n + 1, N + 1):
            C[d * m:d * (m + 1), d * n:d * (n + 1)] = A[m] @ C[d * (m - 1):d * m, d * n:d * (n + 1)]
            C[d * n:d * (n + 1), d * m:d * (m + 1)] = C[d * m:d * (m + 1), d * n:d * (n + 1)].T
    mfull = np.concatenate(mpri)
    Kxy = C[np.ix_(xi, yi)] @ inv(C[np.ix_(yi, yi)])
    x_cov = mfull[xi] + Kxy @ (yv - mfull[yi])
    P_cov = C[np.ix_(xi, xi)] - Kxy @ C[np.ix_(yi, xi)]
    res["J.post"] = max(rel(Jinv, P_cov), rel(xh, x_cov))

    # ---------- pairwise Kalman filter (time-varying) ----------
    xf, Pf = [None] * (N + 1), [None] * (N + 1)
    zp, Pp, S, K, io = ([None] * (N + 1) for _ in range(5))
    xf[0], Pf[0], _, _ = cond_on_y(m0, P0, y[0])
    for n in range(1, N + 1):
        zp[n] = A[n] @ cat(xf[n - 1], y[n - 1])
        Pp[n] = A[n] @ bd(Pf[n - 1]) @ A[n].T + R[n]
        S[n] = Pp[n][iy, iy]; K[n] = Pp[n][ix, iy] @ inv(S[n]); io[n] = y[n] - zp[n][iy]
        xf[n] = zp[n][ix] + K[n] @ io[n]
        Pf[n] = Pp[n][ix, ix] - K[n] @ S[n] @ K[n].T
    Pfi = [inv(P) for P in Pf]

    # ---------- explicit time-varying J blocks and eta ----------
    Jx, Jn = Z(J.shape), Z(J.shape)            # Jn: letter eq.(4) read with same index n
    etax = Z(eta.shape)
    for n in range(N + 1):
        fut = M[n + 1].T @ Ri[n + 1] @ M[n + 1] if n < N else Z((p, p))
        past = Pfi[0] if n == 0 else E.T @ Ri[n] @ E
        Jx[p * n:p * (n + 1), p * n:p * (n + 1)] = past + fut
        if 0 < n < N:
            Jn[p * n:p * (n + 1), p * n:p * (n + 1)] = past + M[n].T @ Ri[n] @ M[n]
        else:
            Jn[p * n:p * (n + 1), p * n:p * (n + 1)] = past + fut
        if n >= 1:
            off = -E.T @ Ri[n] @ M[n]
            for X in (Jx, Jn):
                X[p * n:p * (n + 1), p * (n - 1):p * n] = off
                X[p * (n - 1):p * n, p * n:p * (n + 1)] = off.T
        e_past = Pfi[0] @ xf[0] if n == 0 else -E.T @ Ri[n] @ c[n]
        e_fut = M[n + 1].T @ Ri[n + 1] @ c[n + 1] if n < N else Z(p)
        etax[p * n:p * (n + 1)] = e_past + e_fut
    res["J.blocks"] = rel(Jx, J)
    res["J.eta"] = rel(etax, eta)
    res["J.naive"] = rel(Jn, J)

    # ---------- (a) top-down block-Thomas sweep on the assembled J ----------
    Dl, ct = [None] * (N + 1), [None] * (N + 1)
    Dl[0], ct[0] = blk(J, 0, 0), seg(eta, 0)
    for n in range(1, N + 1):
        T = blk(J, n, n - 1) @ inv(Dl[n - 1])
        Dl[n] = blk(J, n, n) - T @ blk(J, n - 1, n)
        ct[n] = seg(eta, n) - T @ ct[n - 1]
    Dpred = [Pfi[n] + (M[n + 1].T @ Ri[n + 1] @ M[n + 1] if n < N else Z((p, p)))
             for n in range(N + 1)]
    res["a.pivot"] = rel(Dl, Dpred)
    if N >= 2:      # naive indexing differs from the correct one only for 0 < n < N
        res["a.pivot.naive"] = rel(Dl[1:N], [Pfi[n] + M[n].T @ Ri[n] @ M[n] for n in range(1, N)])
    ctpred = [Pfi[n] @ xf[n] + (M[n + 1].T @ Ri[n + 1] @ c[n + 1] if n < N else Z(p))
              for n in range(N + 1)]
    res["a.rhs"] = rel(ct, ctpred)

    # ---------- (b) multiplier, gain, RTS recursions ----------
    G = [Pf[n] @ M[n + 1].T @ inv(Pp[n + 1]) for n in range(N)]
    Dli = [inv(X) for X in Dl]
    res["b.mult"] = rel([-Dli[n] @ blk(J, n, n + 1) for n in range(N)], [G[n] @ E for n in range(N)])
    res["b.gain"] = rel([Dli[n] @ M[n + 1].T @ Ri[n + 1] for n in range(N)], G)
    res["b.pivinv"] = rel(Dli[:N], [Pf[n] - G[n] @ Pp[n + 1] @ G[n].T for n in range(N)])
    xR, PR = [None] * (N + 1), [None] * (N + 1)
    xR[N], PR[N] = xf[N], Pf[N]
    for n in range(N - 1, -1, -1):
        xR[n] = xf[n] + G[n] @ (cat(xR[n + 1], y[n + 1]) - zp[n + 1])
        PR[n] = Pf[n] + G[n] @ (bd(PR[n + 1]) - Pp[n + 1]) @ G[n].T
    res["b.rts.mean"] = rel(xR, xs)
    res["b.rts.cov"] = rel(PR, Ps)

    # ---------- (c) MBF adjoint recursion (awesomepkf _mbf_backward_pass, time-varying) ----
    lam, Lm = [None] * (N + 1), [None] * (N + 1)
    mu, Gam = [None] * (N + 1), [None] * (N + 1)
    lam[N], Lm[N] = Z(p), Z((p, p))
    for n in range(N, 0, -1):
        Si = inv(S[n])
        Fn = Z((d, p)); Fn[:p, :] = I(p); Fn[p:, :] = -K[n].T          # (I_p; -K_n^T)
        mu[n] = cat(lam[n], Si @ io[n] - K[n].T @ lam[n])
        Gam[n] = Fn @ Lm[n] @ Fn.T
        Gam[n][p:, p:] = Gam[n][p:, p:] + Si
        lam[n - 1] = M[n].T @ mu[n]
        Lm[n - 1] = M[n].T @ Gam[n] @ M[n]
    lam_ref = [Pfi[n] @ (xs[n] - xf[n]) for n in range(N + 1)]
    Lam_ref = [Pfi[n] @ (Pf[n] - Ps[n]) @ Pfi[n] for n in range(N + 1)]
    res["c.lam"] = rel(lam, lam_ref)
    res["c.Lam"] = rel(Lm, Lam_ref)
    Ppi = [None] + [inv(Pp[n]) for n in range(1, N + 1)]
    res["c.mu"] = rel(mu[1:], [Ppi[n] @ (cat(xs[n], y[n]) - zp[n]) for n in range(1, N + 1)])
    res["c.Gam"] = rel(Gam[1:], [Ppi[n] @ (Pp[n] - bd(Ps[n])) @ Ppi[n] for n in range(1, N + 1)])

    # ---------- (d) backward chain, J from the backward factorisation ----------
    Ab = [Sig[n] @ A[n + 1].T @ inv(Sig[n + 1]) for n in range(N)]
    Qb = [Sig[n] - Ab[n] @ Sig[n + 1] @ Ab[n].T for n in range(N)]
    Qbi = [inv(Qx) for Qx in Qb]
    cb = [mpri[n] - Ab[n] @ mpri[n + 1] for n in range(N)]
    Mb = [Ab[n][:, :p] for n in range(N)]
    Lb, hb = Z((D, D)), Z(D)
    SNi = inv(Sig[N])
    Lb[d * N:, d * N:] += SNi; hb[d * N:] += SNi @ mpri[N]
    for n in range(N):        # residual z_n - Ab_n z_{n+1} - cb_n ~ N(0, Qb_n)
        a, b = slice(d * n, d * (n + 1)), slice(d * (n + 1), d * (n + 2))
        Lb[a, a] += Qbi[n]; Lb[b, b] += Ab[n].T @ Qbi[n] @ Ab[n]
        Lb[a, b] -= Qbi[n] @ Ab[n]; Lb[b, a] -= Ab[n].T @ Qbi[n]
        hb[a] += Qbi[n] @ cb[n]; hb[b] -= Ab[n].T @ Qbi[n] @ cb[n]
    res["J.back"] = rel(Lb[np.ix_(xi, xi)], J)
    res["J.back.eta"] = rel(hb[xi] - Lb[np.ix_(xi, yi)] @ yv, eta)

    # prior conditioned on y_n, backward filter
    xy_, Py_ = [None] * (N + 1), [None] * (N + 1)
    for n in range(N + 1):
        xy_[n], Py_[n], _, _ = cond_on_y(mpri[n], Sig[n], y[n])
    Pyi = [inv(P) for P in Py_]
    xb, Pb = [None] * (N + 1), [None] * (N + 1)
    zpb, Ppb = [None] * N, [None] * N
    xb[N], Pb[N], _, _ = cond_on_y(mpri[N], Sig[N], y[N])
    for n in range(N, 0, -1):
        Ppb[n - 1] = Mb[n - 1] @ Pb[n] @ Mb[n - 1].T + Qb[n - 1]
        zpb[n - 1] = Ab[n - 1] @ cat(xb[n], y[n]) + cb[n - 1]
        xb[n - 1], Pb[n - 1], _, _ = cond_on_y(zpb[n - 1], Ppb[n - 1], y[n - 1])
    Pbi = [inv(P) for P in Pb]

    # explicit blocks in backward-chain quantities
    Jbx = Z(J.shape)
    for n in range(N + 1):
        own = E.T @ Qbi[n] @ E if n < N else Pyi[N]
        prev = Mb[n - 1].T @ Qbi[n - 1] @ Mb[n - 1] if n > 0 else Z((p, p))
        Jbx[p * n:p * (n + 1), p * n:p * (n + 1)] = own + prev
        if n < N:
            off = -E.T @ Qbi[n] @ Mb[n]
            Jbx[p * n:p * (n + 1), p * (n + 1):p * (n + 2)] = off
            Jbx[p * (n + 1):p * (n + 2), p * n:p * (n + 1)] = off.T
    res["J.back.blocks"] = rel(Jbx, J)
    # prior-chain split: E'R_n^-1 E = (P^y_n)^-1 + Mb_{n-1}' Qb_{n-1}^-1 Mb_{n-1} (n>=1),
    #                    E'Qb_0^-1 E = (P^y_0)^-1 + M_1' R_1^-1 M_1
    lhs = [E.T @ Qbi[0] @ E] + [E.T @ Ri[n] @ E for n in range(1, N + 1)]
    rhs = [Pyi[0] + M[1].T @ Ri[1] @ M[1]] + \
          [Pyi[n] + Mb[n - 1].T @ Qbi[n - 1] @ Mb[n - 1] for n in range(1, N + 1)]
    res["J.split"] = rel(lhs, rhs)

    # bottom-up sweep
    Db, cbt = [None] * (N + 1), [None] * (N + 1)
    Db[N], cbt[N] = blk(J, N, N), seg(eta, N)
    for n in range(N - 1, -1, -1):
        T = blk(J, n, n + 1) @ inv(Db[n + 1])
        Db[n] = blk(J, n, n) - T @ blk(J, n + 1, n)
        cbt[n] = seg(eta, n) - T @ cbt[n + 1]
    Dbpred = [Pbi[n] + (Mb[n - 1].T @ Qbi[n - 1] @ Mb[n - 1] if n > 0 else Z((p, p)))
              for n in range(N + 1)]
    res["d.pivot"] = rel(Db, Dbpred)
    cpr = [None] + [F @ y[n - 1] - Ab[n - 1][:, p:] @ y[n] - cb[n - 1] for n in range(1, N + 1)]
    cbpred = [Pbi[n] @ xb[n] + (Mb[n - 1].T @ Qbi[n - 1] @ cpr[n] if n > 0 else Z(p))
              for n in range(N + 1)]
    res["d.rhs"] = rel(cbt, cbpred)
    Jminus = [Pfi[0]] + [E.T @ Ri[n] @ E for n in range(1, N + 1)]
    res["d.U"] = rel(Db, [Pbi[n] - Pyi[n] + Jminus[n] for n in range(N + 1)])
    Dw = [None] + [Pb[n] @ Mb[n - 1].T @ inv(Ppb[n - 1]) for n in range(1, N + 1)]
    Dbi = [inv(X) for X in Db]
    res["d.mult"] = rel([-Dbi[n] @ blk(J, n, n - 1) for n in range(1, N + 1)],
                        [Dw[n] @ E for n in range(1, N + 1)])
    xD, PD = [None] * (N + 1), [None] * (N + 1)
    xD[0], PD[0] = xb[0], Pb[0]
    for n in range(1, N + 1):
        xD[n] = xb[n] + Dw[n] @ (cat(xD[n - 1], y[n - 1]) - zpb[n - 1])
        PD[n] = Pb[n] + Dw[n] @ (bd(PD[n - 1]) - Ppb[n - 1]) @ Dw[n].T
    res["d.dwy.mean"] = rel(xD, xs)
    res["d.dwy.cov"] = rel(PD, Ps)

    # ---------- (e) twisted factorisation and 2F fusion ----------
    Psi = [inv(P) for P in Ps]
    res["e.twist"] = rel([Dl[n] + Db[n] - blk(J, n, n) for n in range(N + 1)], Psi)
    res["e.2F"] = rel([Pfi[n] + Pbi[n] - Pyi[n] for n in range(N + 1)], Psi)
    info_x = [Psi[n] @ xs[n] for n in range(N + 1)]
    res["e.twist.mean"] = rel([ct[n] + cbt[n] - seg(eta, n) for n in range(N + 1)], info_x)
    res["e.2F.mean"] = rel([Pfi[n] @ xf[n] + Pbi[n] @ xb[n] - Pyi[n] @ xy_[n]
                            for n in range(N + 1)], info_x)

    keep = {"xs": xs, "Ps": Ps, "xf": xf, "Pf": Pf, "S": S, "K": K, "io": io,
            "lam": lam, "Lam": Lm, "mult": [-Dli[n] @ blk(J, n, n + 1) for n in range(N)],
            "G": G, "condJ": None}
    if bk.name == "float64":
        keep["condJ"] = float(np.linalg.cond(J))
        keep["condPp"] = max(float(np.linalg.cond(Pp[n])) for n in range(1, N + 1))
    return res, keep


# ======================================================================
# Library (awesomepkf) checks on time-invariant cases
# ======================================================================
def load_library():
    from prg.classes import linear_pks as LP
    from prg.classes.param_linear import ParamLinear
    from prg.models.linear._amq import LinearAmQ
    src, start = inspect.getsourcelines(LP._mbf_backward_pass)
    hits = [start + i for i, line in enumerate(src) if "Xf + Pf @ lam" in line]
    if len(hits) != 1:
        raise RuntimeError("cannot locate the MBF recovery line in _mbf_backward_pass")
    return {"LP": LP, "ParamLinear": ParamLinear, "LinearAmQ": LinearAmQ,
            "mbf_code": LP._mbf_backward_pass.__code__, "mbf_line": hits[0]}


class _WarnCounter(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.n = 0

    def emit(self, record):
        self.n += 1


def library_checks(case: dict, own: dict, L: dict) -> tuple[dict, dict]:
    p, q, N = case["p"], case["q"], case["N"]
    d = p + q
    LP = L["LP"]
    m = L["LinearAmQ"](p, q, A=case["A"][1], mQ=case["R"][1], mz0=case["m0"].reshape(d, 1),
                       Pz0=case["P0"], pairwiseModel=True)
    prm = m.get_params().copy(); prm.pop("dim_x"); prm.pop("dim_y")
    param = L["ParamLinear"](0, p, q, **prm)
    data = [(k, case["x"][k].reshape(p, 1), case["y"][k].reshape(q, 1)) for k in range(N + 1)]
    xs, Ps, xf, Pf = own["xs"], own["Ps"], own["xf"], own["Pf"]
    res, info = {}, {}
    variants = {"RTS": LP.Linear_PKS_RTS, "BF": LP.Linear_PKS_BF, "MBF": LP.Linear_PKS_MBF,
                "2F": LP.Linear_PKS_MF, "DWY": LP.Linear_PKS_DWY, "VAR": LP.Linear_PKS_VAR}
    hist = {}
    cap: dict = {}
    for name, cls in variants.items():
        sm = cls(param)
        if name == "MBF":
            code, line = L["mbf_code"], L["mbf_line"]

            def local(frame, event, arg):
                if event == "line" and frame.f_lineno == line:
                    loc = frame.f_locals
                    cap[loc["n"] - 1] = (np.array(loc["lam"], float).ravel().copy(),
                                         np.array(loc["Lam"], float).copy())
                return local

            def glob(frame, event, arg):
                return local if frame.f_code is code else None

            old = sys.gettrace()
            sys.settrace(glob)
            try:
                sm.process_N_data_smoother(N=N, data_generator=iter(data))
            finally:
                sys.settrace(old)
        else:
            sm.process_N_data_smoother(N=N, data_generator=iter(data))
        hist[name] = sm.history
        xl = [np.asarray(r["Xkp1_smooth"], float).ravel() for r in sm.history]
        Pl = [np.asarray(r["PXXkp1_smooth"], float).reshape(p, p) for r in sm.history]
        info[f"lib.{name}.mean"] = rel(xl, xs)
        info[f"lib.{name}.cov"] = rel(Pl, Ps)

    h = hist["MBF"]
    xfl = [np.asarray(r["Xkp1_update"], float).ravel() for r in h]
    Pfl = [np.asarray(r["PXXkp1_update"], float).reshape(p, p) for r in h]
    info["lib.filter"] = max(rel(xfl, xf), rel(Pfl, Pf),
                             rel([np.asarray(h[n]["Skp1"], float).reshape(q, q) for n in range(1, N + 1)],
                                 own["S"][1:]),
                             rel([np.asarray(h[n]["Kkp1"], float).reshape(p, q) for n in range(1, N + 1)],
                                 own["K"][1:]))
    # (c) library MBF internals (lambda_n, Lambda_n), n = N-1..0, captured from the pass
    if sorted(cap) != list(range(N)):
        raise RuntimeError(f"MBF capture incomplete: {sorted(cap)}")
    Pfi = [np.linalg.inv(P) for P in Pf]
    lam_ref = [Pfi[n] @ (xs[n] - xf[n]) for n in range(N)]
    Lam_ref = [Pfi[n] @ (Pf[n] - Ps[n]) @ Pfi[n] for n in range(N)]
    res["c.lib.lam"] = rel([cap[n][0] for n in range(N)], lam_ref)
    res["c.lib.Lam"] = rel([cap[n][1] for n in range(N)], Lam_ref)
    res["c.lib.own"] = max(rel([cap[n][0] for n in range(N)], own["lam"][:N]),
                           rel([cap[n][1] for n in range(N)], own["Lam"][:N]))
    xsl = [np.asarray(r["Xkp1_smooth"], float).ravel() for r in h]
    res["c.lib.out"] = rel([np.linalg.solve(Pfl[n], xsl[n] - xfl[n]) for n in range(N + 1)],
                           own["lam"])
    # (b) library RTS gain (X rows of the joint gain) vs J back-substitution multiplier
    hR = hist["RTS"]
    Gl = [np.asarray(hR[n]["Gk_smooth"], float).reshape(p, d) for n in range(N)]
    res["b.lib.gain"] = rel([g[:, :p] for g in Gl], own["mult"])
    return res, info


# ======================================================================
# Driver
# ======================================================================
ROWS = [
    # key,            gated, description
    ("J.post",        True,  "J^-1, J^-1 eta = Gaussian conditioning of joint prior cov"),
    ("J.blocks",      True,  "J_nn=E'R_n^-1E+M_{n+1}'R_{n+1}^-1M_{n+1}, J_{n,n-1}, J_00, J_NN"),
    ("J.eta",         True,  "eta_n explicit (time-varying)"),
    ("J.naive",       False, "[info] J_nn with same-index M_n,R_n (eq.(4) read TV)"),
    ("J.back",        True,  "(d) J from backward factorisation = J"),
    ("J.back.eta",    True,  "(d) eta from backward factorisation = eta"),
    ("J.back.blocks", True,  "(d) J blocks in backward-chain quantities (Qb, Mb, P^y_N)"),
    ("J.split",       True,  "E'R_n^-1E = (P^y_n)^-1 + Mb_{n-1}'Qb_{n-1}^-1Mb_{n-1}"),
    ("a.pivot",       True,  "(a) Delta_n=(Pf_n)^-1+M_{n+1}'R_{n+1}^-1M_{n+1}, Delta_N=(Pf_N)^-1"),
    ("a.pivot.naive", False, "[info] same with naive (M_n, R_n), 0<n<N"),
    ("a.rhs",         True,  "(a) RHS c~_n=(Pf_n)^-1 xf_n + M_{n+1}'R_{n+1}^-1 c_{n+1}"),
    ("b.mult",        True,  "(b) -Delta_n^-1 J_{n,n+1} = G_n E"),
    ("b.gain",        True,  "(b) Delta_n^-1 M_{n+1}'R_{n+1}^-1 = G_n  (p x (p+q))"),
    ("b.pivinv",      True,  "(b) Delta_n^-1 = Pf_n - G_n P_{n+1|n} G_n'"),
    ("b.rts.mean",    True,  "(b) RTS mean recursion (propagated from N)"),
    ("b.rts.cov",     True,  "(b) RTS covariance recursion (propagated from N)"),
    ("b.lib.gain",    True,  "(b) awesomepkf RTS Gk_smooth E = multiplier  [TI]"),
    ("c.lam",         True,  "(c) MBF recursion: lambda_n = (Pf_n)^-1 (x_{n|N}-x_{n|n})"),
    ("c.Lam",         True,  "(c) MBF recursion: Lambda_n = Pf^-1 (Pf - P_{n|N}) Pf^-1"),
    ("c.mu",          True,  "(c) mu_n = P_{n|n-1}^-1 (z_{n|N} - z_{n|n-1})"),
    ("c.Gam",         True,  "(c) Gamma_n = P_{n|n-1}^-1 (P_{n|n-1}-P^zz_{n|N}) P_{n|n-1}^-1"),
    ("c.lib.lam",     True,  "(c) awesomepkf MBF internal lambda_n = ref  [TI]"),
    ("c.lib.Lam",     True,  "(c) awesomepkf MBF internal Lambda_n = ref  [TI]"),
    ("c.lib.own",     True,  "(c) awesomepkf MBF internals = own recursion  [TI]"),
    ("c.lib.out",     True,  "(c) (Pf)^-1(x^MBF_{n|N}-x_{n|n}) from lib output = own  [TI]"),
    ("d.pivot",       True,  "(d) Delta^b_n=(Pb_n)^-1+Mb_{n-1}'Qb_{n-1}^-1Mb_{n-1}, D^b_0=(Pb_0)^-1"),
    ("d.rhs",         True,  "(d) bottom-up RHS = (Pb_n)^-1 xb_n + Mb'Qb^-1 c^b'_{n-1}"),
    ("d.U",           True,  "(d) Delta^b_n = U_n + J^-_n, U_n=(Pb_n)^-1-(P^y_n)^-1"),
    ("d.mult",        True,  "(d) -(Delta^b_n)^-1 J_{n,n-1} = D_n E (DWY gain)"),
    ("d.dwy.mean",    True,  "(d) DWY mean recursion (propagated from 0)"),
    ("d.dwy.cov",     True,  "(d) DWY covariance recursion (propagated from 0)"),
    ("e.twist",       True,  "(e) (P_{n|N})^-1 = Delta_n + Delta^b_n - J_nn"),
    ("e.2F",          True,  "(e) (P_{n|N})^-1 = (Pf_n)^-1 + (Pb_n)^-1 - (P^y_n)^-1"),
    ("e.twist.mean",  True,  "(e) P_{n|N}^-1 x_{n|N} = c~_n + c~^b_n - eta_n"),
    ("e.2F.mean",     True,  "(e) P_{n|N}^-1 x_{n|N} = Pf^-1xf + Pb^-1xb - Py^-1xy"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--N", type=int, default=20, help="horizon (default 20)")
    ap.add_argument("--seeds", type=int, default=5, help="seeds per (p,q) and per TI/TV")
    ap.add_argument("--mp-dps", type=int, default=50, help="mpmath digits (default 50)")
    ap.add_argument("--mp-N", type=int, default=8, help="horizon of the mpmath cases")
    ap.add_argument("--no-mp", action="store_true", help="skip the mpmath re-evaluation")
    ap.add_argument("--no-lib", action="store_true", help="skip the awesomepkf checks")
    args = ap.parse_args()

    t_all = time.perf_counter()
    L = None
    if not args.no_lib:
        try:
            L = load_library()
        except ImportError as e:
            print(f"awesomepkf (prg) not importable: {e}. Set PYTHONPATH or use --no-lib.")
            return 2
        wc = _WarnCounter()
        logging.getLogger("prg").addHandler(wc)
        logging.getLogger("prg").setLevel(logging.WARNING)
        logging.getLogger("prg").propagate = False

    worst = {k: {"TI": None, "TV": None, "MP": None} for k, _, _ in ROWS}
    info_w: dict = {}
    struct = {"axy": [], "ayy": [], "rxy": [], "condJ": [], "condPp": []}

    def upd(col, key, val, label):
        if key not in worst:
            raise KeyError(key)
        cur = worst[key][col]
        if cur is None or not np.isfinite(val) or val > cur[0]:
            worst[key][col] = (val if np.isfinite(val) else np.inf, label)

    def wmax(key):
        vals = [worst[key][c][0] for c in ("TI", "TV", "MP") if worst[key][c] is not None]
        return max(vals) if vals else np.nan

    t0 = time.perf_counter()
    ncase = 0
    t_lib = 0.0
    for tv in (False, True):
        for (p, q) in PQ:
            for seed in range(args.seeds):
                case = random_case(p, q, args.N, seed, tv)
                label = f"{'TV' if tv else 'TI'} p={p} q={q} s={seed}"
                res, keep = analyse(case, FloatBK())
                ncase += 1
                for k, v in res.items():
                    upd("TV" if tv else "TI", k, v, label)
                for n in range(1, args.N + 1):
                    An, Rn = case["A"][n], case["R"][n]
                    struct["axy"].append(np.linalg.norm(An[:p, p:]) / np.linalg.norm(An))
                    struct["ayy"].append(np.linalg.norm(An[p:, p:]) / np.linalg.norm(An))
                    dr = np.sqrt(np.diag(Rn))
                    struct["rxy"].append(np.max(np.abs(Rn[:p, p:] / np.outer(dr[:p], dr[p:]))))
                struct["condJ"].append(keep["condJ"]); struct["condPp"].append(keep["condPp"])
                if L is not None and not tv:
                    tl = time.perf_counter()
                    lres, linfo = library_checks(case, keep, L)
                    t_lib += time.perf_counter() - tl
                    for k, v in lres.items():
                        upd("TI", k, v, label)
                    for k, v in linfo.items():
                        if k not in info_w or v > info_w[k][0]:
                            info_w[k] = (v, label)
    t_float = time.perf_counter() - t0 - t_lib

    t_mp = 0.0
    n_mp = 0
    if not args.no_mp:
        t1 = time.perf_counter()
        bk = MpBK(args.mp_dps)
        for (p, q) in [(1, 1), (2, 1), (3, 2), (4, 2)]:
            case = random_case(p, q, args.mp_N, 0, True)
            res, _ = analyse(case, bk)
            n_mp += 1
            for k, v in res.items():
                upd("MP", k, v, f"TV p={p} q={q} s=0 N={args.mp_N}")
        t_mp = time.perf_counter() - t1

    # ---------------- report ----------------
    print(f"Proposition 1 check: {ncase} float64 cases ((p,q) in {{1..4}}x{{1,2}}, "
          f"{args.seeds} seeds, TI and TV, N={args.N}); "
          + (f"{n_mp} mpmath-{args.mp_dps} TV cases (N={args.mp_N}); " if n_mp else "")
          + ("awesomepkf checks on the TI cases." if L is not None else "no library checks."))
    print(f"Model structure: ||A^xy||_F/||A||_F in [{min(struct['axy']):.2f}, {max(struct['axy']):.2f}], "
          f"||A^yy||_F/||A||_F in [{min(struct['ayy']):.2f}, {max(struct['ayy']):.2f}], "
          f"max |corr(R^xy)| per step in [{min(struct['rxy']):.2f}, {max(struct['rxy']):.2f}],\n"
          "                 "
          f"cond(J) in [{min(struct['condJ']):.1e}, {max(struct['condJ']):.1e}], "
          f"max_n cond(P_n|n-1) in [{min(struct['condPp']):.1e}, {max(struct['condPp']):.1e}]")
    print("Relative residual = max_n ||X_n - Y_n||_max / max_n ||Y_n||_max ; gate: <= "
          f"{TOL:.0e} in every column.\n")
    hdr = f"{'id':<14} {'identity':<66} {'TI f64':>8} {'TV f64':>8} {'TV ' + (bk.name if n_mp else 'mp'):>8}  gate"
    print(hdr)
    print("-" * len(hdr))
    fail = []
    for key, gated, desc in ROWS:
        vals = []
        for col in ("TI", "TV", "MP"):
            w = worst[key][col]
            vals.append(f"{w[0]:8.1e}" if w is not None else f"{'-':>8}")
        if gated and all(worst[key][c] is None for c in ("TI", "TV", "MP")):
            g = "skip"            # only possible for the [TI] library rows under --no-lib
        elif gated:
            mx = wmax(key)
            ok = np.isfinite(mx) and mx <= TOL
            g = "ok" if ok else "FAIL"
            if not ok:
                fail.append(key)
        else:
            g = "info"
        print(f"{key:<14} {desc:<66} {vals[0]} {vals[1]} {vals[2]}  {g}")

    if info_w:
        print("\n[info] awesomepkf smoothers vs exact posterior (TI cases, worst case):")
        names = ["RTS", "BF", "MBF", "2F", "DWY", "VAR"]
        print(f"  {'':<6}" + "".join(f"{n:>9}" for n in names))
        print(f"  {'mean':<6}" + "".join(f"{info_w[f'lib.{n}.mean'][0]:9.1e}" for n in names))
        print(f"  {'cov':<6}" + "".join(f"{info_w[f'lib.{n}.cov'][0]:9.1e}" for n in names))
        print(f"  library forward filter vs own filter (x, P, S, K): {info_w['lib.filter'][0]:.1e}")
        print(f"  library warnings logged (regularisations etc.): {wc.n}")

    print("\nWorst cases of the gated rows (largest column):")
    top = sorted(((wmax(k), k) for k, g, _ in ROWS if g and not np.isnan(wmax(k))),
                 reverse=True)[:3]
    for v, k in top:
        col = max((c for c in ("TI", "TV", "MP") if worst[k][c] is not None),
                  key=lambda c: worst[k][c][0])
        print(f"  {k:<14} {v:.1e}  ({worst[k][col][1]})")
    if any(worst[k][c] is None for k, g, _ in ROWS if g and not k.startswith(("b.lib", "c.lib"))
           for c in ("TI", "TV")):
        print("  WARNING: some gated identity was not evaluated in a float64 column")

    t_tot = time.perf_counter() - t_all
    print(f"\nRuntime: total {t_tot:.1f} s  (float64 identities {t_float:.1f} s, "
          f"awesomepkf {t_lib:.1f} s, mpmath {t_mp:.1f} s)")
    if fail:
        print(f"FAILED identities: {', '.join(fail)}")
        return 1
    print("All gated identities hold to round-off.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
