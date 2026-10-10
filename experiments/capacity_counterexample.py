#!/usr/bin/env python
"""Does a classical model always reproduce the pairwise smoother? (Sec. III)

For a scalar GPMC, a member of C = {A^xy = 0} has the same N -> infinity (Wiener) smoother
as the truth only if the numerator of the truth's transfer function, a Laurent polynomial
t_{-1}/z + t_0 + t_1 z (`scalar_laurent`), has real roots (and the member's A^yy, the inverse
of a root, then lies in the unit disc for a stationary member). We draw random stable scalar
GPMCs and count the three cases: no real exact member (complex roots), a real but
non-stationary one, a stationary one. For the draws without any exact member, we search
the stable members of C (A^xx, A^yy in (-1, 1), A^yx real, R > 0) for the smallest Wiener
state-MSE penalty over the pairwise smoother (multi-start Nelder--Mead on a fixed frequency
grid, then an exact re-evaluation with `asym_metrics`).

Run (from the paper directory): python -B experiments/capacity_counterexample.py
Output: experiments/capacity_counterexample_results.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import classical_vs_pairwise_exact as cv  # noqa: E402

OUT_JSON = Path(__file__).resolve().parent / "capacity_counterexample_results.json"
W = np.linspace(-np.pi, np.pi, 1024, endpoint=False) + np.pi / 1024
Z = np.exp(-1j * W)


def spec_blocks(A, R):
    """Phi_xx, Phi_xy, Phi_yy (times 2 pi) of a 2x2 GPMC on the grid, vectorized."""
    a, b, c, d = A[0, 0], A[0, 1], A[1, 0], A[1, 1]
    det = (1 - a * Z) * (1 - d * Z) - b * c * Z * Z
    m11, m12, m21, m22 = (1 - d * Z) / det, b * Z / det, c * Z / det, (1 - a * Z) / det
    r11, r12, r22 = R[0, 0], R[0, 1], R[1, 1]
    pxx = r11 * abs(m11) ** 2 + 2 * r12 * np.real(m11 * np.conj(m12)) + r22 * abs(m12) ** 2
    pyy = r11 * abs(m21) ** 2 + 2 * r12 * np.real(m21 * np.conj(m22)) + r22 * abs(m22) ** 2
    pxy = (r11 * m11 * np.conj(m21) + r12 * (m11 * np.conj(m22) + m12 * np.conj(m21))
           + r22 * m12 * np.conj(m22))
    return pxx, pxy, pyy


def wiener_mse(At, Rt, Am, Rm):
    pxx, pxy, pyy = spec_blocks(At, Rt)
    _, mxy, myy = spec_blocks(Am, Rm)
    H = mxy / myy
    return float(np.mean(pxx - 2 * np.real(H * np.conj(pxy)) + abs(H) ** 2 * pyy))


def unpack(th):
    A = np.array([[np.tanh(th[0]), 0.0], [th[1], np.tanh(th[2])]])
    L = np.array([[np.exp(th[3]), 0.0], [th[4], np.exp(th[5])]])
    return A, L @ L.T


def pack(A, R):
    """Inverse of `unpack` for a stable member of C (|A^xx|, |A^yy| < 1, R > 0)."""
    L = np.linalg.cholesky(0.5 * (R + R.T))
    clip = lambda v: float(np.clip(v, -0.999, 0.999))
    return np.array([np.arctanh(clip(A[0, 0])), A[1, 0], np.arctanh(clip(A[1, 1])),
                     np.log(L[0, 0]), L[1, 0], np.log(L[1, 1])])


def best_stable_C(At, Rt, rng, starts=20):
    """Smallest Wiener penalty over the stable members of C. Starts: the likelihood fit (KL
    projection) and the ablated model -- so the result is never worse than either -- then
    random points scaled to the truth's noise level."""
    base = wiener_mse(At, Rt, At, Rt)
    seeds = [pack(*cv.project_classC_closed(At, Rt, 1, 1)), pack(*cv.ablate(At, Rt, 1))]
    scale = 0.5 * np.log(np.trace(Rt) / 2)
    for _ in range(starts):
        th = rng.standard_normal(6) * 0.7
        th[[3, 5]] += scale
        seeds.append(th)
    best = None
    for th0 in seeds:
        r = minimize(lambda th: wiener_mse(At, Rt, *unpack(th)), th0,
                     method="Nelder-Mead", options={"maxiter": 6000, "xatol": 1e-9, "fatol": 1e-13})
        if best is None or r.fun < best.fun:
            best = r
    Am, Rm = unpack(best.x)
    exact = cv.asym_metrics(At, Rt, Am, Rm, 1)["mse"] / cv.asym_metrics(At, Rt, At, Rt, 1)["mse"]
    return 100 * (best.fun / base - 1), 100 * (exact - 1)


def main(n_draws=300, n_search=12, seed=7):
    # validation of the fast grid evaluation against the exact routine of the cost script
    At, Rt = cv.A_true(1.0), cv.R_TRUE
    Am, Rm = cv.ablate(At, Rt, 1)
    fast = wiener_mse(At, Rt, Am, Rm) / wiener_mse(At, Rt, At, Rt)
    exact = cv.asym_metrics(At, Rt, Am, Rm, 1)["mse"] / cv.asym_metrics(At, Rt, At, Rt, 1)["mse"]
    print(f"grid check (ablated, Fig. 1 model): fast {fast:.6f}  exact {exact:.6f}")

    rng = np.random.default_rng(seed)
    counts = {"no_real_member": 0, "real_unstable_member": 0, "stable_member": 0}
    nomatch = []
    while sum(counts.values()) < n_draws:
        A = rng.uniform(-0.8, 0.8, (2, 2))
        if np.max(np.abs(np.linalg.eigvals(A))) > 0.9:
            continue
        G = rng.standard_normal((2, 3))
        R = G @ G.T / 3
        (tm1, t0, t1), _ = cv.scalar_laurent(A, R)
        if t0 * t0 - 4 * tm1 * t1 < 0:
            counts["no_real_member"] += 1
            nomatch.append((A, R))
            continue
        stable = any(m is not None and m[3] for m in
                     (cv.classC_match_member(A, R, float(a)) for a in np.linspace(-0.99, 0.99, 199)))
        counts["stable_member" if stable else "real_unstable_member"] += 1
    print(counts, flush=True)
    pens = []
    srng = np.random.default_rng(seed + 1)
    for k, (A, R) in enumerate(nomatch[:n_search]):
        fast, exact = best_stable_C(A, R, srng)
        pens.append(exact)
        print(f"  no-match draw {k + 1}: best stable member of C, penalty {exact:.1f}% "
              f"(grid {fast:.1f}%)", flush=True)
    pens = np.array(pens)
    res = {"n_draws": n_draws, "counts": counts, "seed": seed,
           "draw": "A entries U(-0.8, 0.8), spectral radius <= 0.9; R = G G^T / 3, G 2x3 N(0,1)",
           "best_stable_penalty_pct": {"values": pens.tolist(), "median": float(np.median(pens)),
                                       "min": float(pens.min()), "max": float(pens.max())}}
    OUT_JSON.write_text(json.dumps(res, indent=1))
    print(json.dumps(res["best_stable_penalty_pct"], indent=1))
    print(f"saved {OUT_JSON}")


if __name__ == "__main__":
    main()
