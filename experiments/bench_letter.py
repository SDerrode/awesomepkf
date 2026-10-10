#!/usr/bin/env python
"""Run time of the six awesomepkf smoothers on random stable pairwise models of growing size
(p, q), N time steps, median over repetitions: total (forward filter + smoothing pass, wall
clock), the forward filter alone, and their difference (the smoothing pass).

Models: A Gaussian, rescaled to spectral radius 0.8; R = D C D with C a random correlation
matrix (normalised Wishart) and unit variances; prior = stationary law.

Run (from the letter directory):
  PYTHONPATH=/path/to/awesomePKF python -B experiments/bench_letter.py
Output: experiments/bench_letter_results.json and a printed table (ms and ratio to RTS).
Absolute times depend on the machine and on the library's per-step diagnostics; the ratios
are what the letter quotes.
"""
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import solve_discrete_lyapunov

from prg import __version__ as prg_version
from prg.classes.linear_pkf import Linear_PKF
from prg.classes.linear_pks import Linear_PKS
from prg.classes.param_linear import ParamLinear
from prg.models.linear._amq import LinearAmQ

OUT = Path(__file__).resolve().parent / "bench_letter_results.json"
METHODS = ["RTS", "BF", "MBF", "2F", "DWY", "VAR"]
SIZES = [(1, 1), (2, 1), (2, 2), (5, 5), (10, 5), (20, 10)]
N, REPS = 1000, 5


def random_param(p, q, seed=0):
    rng = np.random.default_rng(seed)
    d = p + q
    A = rng.standard_normal((d, d))
    A *= 0.8 / np.max(np.abs(np.linalg.eigvals(A)))
    G = rng.standard_normal((d, d + 2))
    W = G @ G.T
    s = 1.0 / np.sqrt(np.diag(W))
    R = s[:, None] * W * s[None, :]
    R = 0.5 * (R + R.T)
    P0 = solve_discrete_lyapunov(A, R)
    m = LinearAmQ(p, q, A=A, mQ=R, mz0=np.zeros((d, 1)), Pz0=0.5 * (P0 + P0.T),
                  pairwiseModel=True)
    pp = m.get_params().copy()
    pp.pop("dim_x")
    pp.pop("dim_y")
    return ParamLinear(0, p, q, **pp)


def time_method(param, sim, method):
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        Linear_PKS(param, method=method).process_N_data_smoother(N=None, data_generator=iter(sim))
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def time_filter(param, sim):
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        list(Linear_PKF(param, sKey=1).process_filter(N=None, data_generator=iter(sim)))
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main():
    res = {"meta": {"N": N, "reps": REPS, "awesomepkf": prg_version,
                    "python": sys.version.split()[0], "numpy": np.__version__,
                    "machine": platform.machine(), "processor": platform.processor()},
           "times_s": {}, "filter_s": {}, "smoothing_pass_s": {}}
    print(f"N={N}, median of {REPS} runs; ms (ratio to RTS)")
    for p, q in SIZES:
        param = random_param(p, q)
        sim = list(Linear_PKF(param, sKey=1).simulate_N_data(N))
        time_method(param, sim[:50], "RTS")          # warm-up
        t = {m: time_method(param, sim, m) for m in METHODS}
        tf = time_filter(param, sim)
        sp = {m: t[m] - tf for m in METHODS}
        key = f"{p},{q}"
        res["times_s"][key], res["filter_s"][key], res["smoothing_pass_s"][key] = t, tf, sp
        print(f"(p,q)=({p:2d},{q:2d}) total: " + "  ".join(
            f"{m} {t[m] * 1e3:5.0f} ({t[m] / t['RTS']:.2f})" for m in METHODS), flush=True)
        print(f"{'':12s}filter {tf * 1e3:5.0f} ms ({tf / t['RTS']:.0%} of RTS total); smoothing pass: "
              + "  ".join(f"{m} {sp[m] * 1e3:4.0f} ({sp[m] / sp['RTS']:.2f})" for m in METHODS),
              flush=True)
    OUT.write_text(json.dumps(res, indent=1))
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
