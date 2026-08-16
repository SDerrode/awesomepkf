"""Cost of the classical "skip the update" recipe under missing observations.

Produces the numbers quoted in the CHANGELOG entry on missing-observation
support: in a pairwise model ``y`` is a component of the Markov chain, so a
gap must be handled by *marginalising* over the missing ``y`` (carrying the
FULL joint covariance across the gap, what ``Linear_PKF.process_filter``
does). The classical recipe — skip the update and rebuild the joint as
``blkdiag(P^xx, 0)`` — silently zeroes the Y and cross blocks the gap leaves
behind. This script measures what that costs.

Protocol, per gap rate ``r``:

1. draw ``B`` random stable couples (p=4, q=2): dense ``A`` rescaled to a
   spectral radius drawn in [0.6, 0.95], dense random PSD ``Q``;
2. simulate ``N`` steps with the released simulator, drop each ``y_k``
   (k >= 1) independently with probability ``r``;
3. filter twice from the SAME first estimate:
   * **exact** — the released ``Linear_PKF.process_filter`` (all-NaN gaps),
   * **naive** — the same joint recursion but with the blkdiag rebuild on
     gap steps (the ablation);
4. report the distribution over couples of the relative excess state RMSE
   ``100 * (RMSE_naive / RMSE_exact - 1)`` (median, 9th decile, max), both
   over ALL steps and over the GAP steps only, plus the number of *diverged*
   naive runs (non-finite estimates or a singular innovation covariance),
   which are excluded from the percentiles.

Note the gap-step-only column is a negative control: at the gap step itself
both recipes return the same X posterior (= the prior); they differ in the
JOINT they carry forward, so the damage lands on the steps AFTER each gap —
the all-steps column is the meaningful cost.

Run (with ``prg`` importable):  python experiments/missing_obs_ablation.py
Optional flags:                 python experiments/missing_obs_ablation.py <B> <N>
"""
from __future__ import annotations

import sys

import numpy as np

from prg.classes.linear_pkf import Linear_PKF
from prg.classes.param_linear import ParamLinear
from prg.models.linear._amq import LinearAmQ
from prg.utils.exceptions import PKFError

P_DIM, Q_DIM = 4, 2                 # couple dimensions of the CHANGELOG claim
GAP_RATES = (0.1, 0.2, 0.3)         # "moderate gap rates"
SR_RANGE = (0.6, 0.95)              # spectral radius of the random couples


def _pd(Q: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    Q = 0.5 * (Q + Q.T)
    lo = float(np.min(np.linalg.eigvalsh(Q)))
    return Q + (eps - lo) * np.eye(Q.shape[0]) if lo < eps else Q


def _random_param(rng: np.random.Generator) -> ParamLinear:
    """Random stable pairwise couple (p=4, q=2), house construction."""
    d = P_DIM + Q_DIM
    A = rng.standard_normal((d, d))
    A *= rng.uniform(*SR_RANGE) / np.max(np.abs(np.linalg.eigvals(A)))
    G = rng.standard_normal((d, d)) / np.sqrt(d)
    Q = _pd(G @ G.T)
    m = LinearAmQ(P_DIM, Q_DIM, A=A, mQ=Q, mz0=np.zeros((d, 1)),
                  Pz0=np.eye(d), pairwiseModel=True)
    p = m.get_params().copy()
    p.pop("dim_x")
    p.pop("dim_y")
    return ParamLinear(0, P_DIM, Q_DIM, **p)


def _gapped(records, gaps):
    """Yield (k, x, y) with y replaced by all-NaN on gap steps."""
    for k, x, y in records:
        if k in gaps:
            y = np.full_like(np.asarray(y, dtype=float), np.nan)
        yield k, x, y


def _naive_estimates(param, hist0, Y, gaps):
    """The ablation: joint recursion identical to the exact filter on
    observed steps, but rebuilding blkdiag(P^xx, 0) across each gap.

    Returns the per-step state estimates (k = 1..N), or None if the run
    diverged (singular innovation covariance or non-finite estimate).
    """
    p = param.dim_x
    A = np.asarray(param.A, dtype=float)
    Qeff = param.B @ param.mQ @ param.B.T
    d = A.shape[0]

    z = np.zeros((d, 1))
    P = np.zeros((d, d))
    z[:p] = hist0["Xkp1_update"]
    z[p:] = hist0["ykp1"]
    P[:p, :p] = hist0["PXXkp1_update"]

    out = []
    for k in range(1, len(Y)):
        z = A @ z
        P = A @ P @ A.T + Qeff
        if k in gaps:
            # Classical recipe: posterior = prior for X, then zero the Y and
            # cross blocks — the joint is forced back to blkdiag(P^xx, 0).
            P[:p, p:] = 0.0
            P[p:, :p] = 0.0
            P[p:, p:] = 0.0
        else:
            S = P[p:, p:]
            try:
                G = np.linalg.solve(S.T, P[:, p:].T).T   # [Pxy; Pyy] S^-1
            except np.linalg.LinAlgError:
                return None
            z = z + G @ (Y[k] - z[p:])
            P = P - G @ P[p:, :]
        if not np.all(np.isfinite(z)):
            return None
        out.append(z[:p].copy())
    return out


def _rmse(estimates, X, keep=None):
    """State RMSE over steps k = 1..N; ``keep`` restricts to a step subset."""
    pairs = zip(estimates, X[1:], strict=True)
    err = np.hstack([e - x for k, (e, x) in enumerate(pairs, start=1)
                     if keep is None or k in keep])
    return float(np.sqrt(np.mean(err**2)))


def main(B: int = 200, N: int = 400, seed: int = 1) -> dict:
    print(f"missing-obs ablation: B={B} random couples "
          f"(p={P_DIM}, q={Q_DIM}), N={N} steps, seed={seed}")
    print(f"{'rate':>5} | {'all: med %':>10} {'q90 %':>7} {'max %':>7} | "
          f"{'gaps: med %':>11} {'q90 %':>7} | diverged")
    print("-" * 74)

    results: dict = {}
    for rate in GAP_RATES:
        exc_all, exc_gap, diverged, skipped = [], [], 0, 0
        for i in range(B):
            rng = np.random.default_rng((seed, i))
            param = _random_param(rng)
            sim = Linear_PKF(param, sKey=seed + i).simulate_N_data(N)
            X = [x for _, x, _ in sim]
            Y = [y for _, _, y in sim]

            mask_rng = np.random.default_rng((seed, i, int(rate * 100)))
            gaps = {k for k in range(1, N + 1)
                    if mask_rng.random() < rate}

            pkf = Linear_PKF(param, sKey=seed + i)
            try:
                list(pkf.process_filter(data_generator=_gapped(sim, gaps)))
            except PKFError:
                skipped += 1     # couple too ill-conditioned for either recipe
                continue
            exact = [pkf.history[k]["Xkp1_update"] for k in range(1, N + 1)]

            naive = _naive_estimates(param, pkf.history[0], Y, gaps)
            if naive is None:
                diverged += 1
                continue

            exc_all.append(
                100.0 * (_rmse(naive, X) / _rmse(exact, X) - 1.0)
            )
            exc_gap.append(
                100.0 * (_rmse(naive, X, gaps) / _rmse(exact, X, gaps) - 1.0)
            )

        stats = {
            "median": float(np.median(exc_all)),
            "q90": float(np.percentile(exc_all, 90)),
            "max": float(np.max(exc_all)),
            "gap_median": float(np.median(exc_gap)),
            "gap_q90": float(np.percentile(exc_gap, 90)),
            "diverged": diverged, "skipped": skipped, "n": len(exc_all),
        }
        results[rate] = stats
        note = f" ({skipped} skipped)" if skipped else ""
        print(f"{rate:>5.0%} | {stats['median']:>9.1f} {stats['q90']:>7.1f} "
              f"{stats['max']:>7.1f} | "
              f"{stats['gap_median']:>10.1f} {stats['gap_q90']:>7.1f} | "
              f"{diverged}/{B}{note}")

    return results


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:]]
    main(*args)
