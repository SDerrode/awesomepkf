#!/usr/bin/env python
"""Does a Joseph-form fusion cure the two-filter smoother (2F)?

Runs the random-family part of conditioning_exact.py (80 pairwise models per regime, the
same four regimes, sweep values, 60-digit reference and metrics) with one extra smoother,
"2FJ": the library's 2F backward filter (shared with DWY, unchanged) followed by a
Joseph-form fusion instead of the library's information-form fusion.

With the future information U_n = (P^b_n)^{-1} - (P^y_n)^{-1} (positive semidefinite in
exact arithmetic, zero at n = N) and nu_n = (P^b_n)^{-1} xb_n - (P^y_n)^{-1} xy_n, the
fusion is a Kalman update of the forward filter (x_{n|n}, P_{n|n}):
    K_n = P_{n|n} (I + U_n P_{n|n})^{-1},   L_n = K_n U_n,
    x_{n|N} = x_{n|n} + K_n (nu_n - U_n x_{n|n}),
    P_{n|N} = (I - L_n) P_{n|n} (I - L_n)^T + K_n U_n K_n^T,
which inverts neither P_{n|n} nor U_n (the library's fusion inverts P_{n|n}, P^b_n, P^y_n
and the fused information matrix).

Run (from the letter directory, ~25 min on 10 cores; --quick ~1 min):
  PYTHONPATH=/path/to/awesomePKF python -B experiments/conditioning_exact_joseph.py
Output: experiments/conditioning_exact_joseph_results.json and a printed 2F vs 2FJ table.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from scipy.linalg import LinAlgError, cho_factor, cho_solve  # noqa: E402

import conditioning_exact as ce  # noqa: E402
import prg.classes.linear_pks as _lpk  # noqa: E402
from prg.utils.exceptions import CovarianceError  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT_JSON = HERE / "conditioning_exact_joseph_results.json"


def _joseph_twofilter_pass(s, N_records):
    """2F with the library's backward filter and a Joseph-form fusion (in place)."""
    bf = _lpk._dwy_backward_filter(s, N_records)
    dx = bf["dx"]
    Sig, mz, ys, xb, Pb = bf["Sig"], bf["mz"], bf["ys"], bf["xb"], bf["Pb"]
    eye = s.eye_dim_x

    def _inv_pd(M, k, name):
        try:
            return cho_solve(cho_factor(M), eye)
        except (LinAlgError, ValueError) as e:
            raise CovarianceError(f"Step {k}: Cholesky failed for {name} (2FJ).",
                                  matrix_name=name, step=k) from e

    for n in range(N_records):
        rec = s.history[n]
        Xf, Pf = rec["Xkp1_update"], rec["PXXkp1_update"]
        Xy, Py = _lpk._cond_xy(mz[n], Sig[n], ys[n], dx)
        Ib = _inv_pd(Pb[n], rec["k"], "Pb_backward")
        Iy = _inv_pd(Py, rec["k"], "Py_prior")
        U = Ib - Iy
        U = 0.5 * (U + U.T)
        nu = Ib @ xb[n] - Iy @ Xy
        K = np.linalg.solve((eye + U @ Pf).T, Pf.T).T          # P_f (I + U P_f)^{-1}
        L = K @ U
        Xs = Xf + K @ (nu - U @ Xf)
        IL = eye - L
        Ps = IL @ Pf @ IL.T + K @ U @ K.T
        Ps = 0.5 * (Ps + Ps.T)
        s._check_covariance(Ps, rec["k"], name="PXXkp1_smooth")
        s.history.update_record(n, Xkp1_smooth=Xs, PXXkp1_smooth=Ps)


class Linear_PKS_2FJ(_lpk.Linear_PKS_MF):
    def _smoothing_pass(self, N_records):
        _joseph_twofilter_pass(self, N_records)


# Patch the engine at module level (spawned workers re-import this module and patch too).
ce.SMOOTHERS["2FJ"] = Linear_PKS_2FJ
if "2FJ" not in ce.NAMES:
    ce.NAMES.append("2FJ")
    ce.ALL_NAMES.insert(len(ce.NAMES) - 1, "2FJ")


def _cell(c):
    cov = c.get("cov")
    s = f"{cov['median']:.0e}/{cov['max']:.0e}" if cov else "   --   "
    return s + (f" a{c['n_abort']}" if c["n_abort"] else "") + \
        (f" i{c['n_indefinite']}" if c["n_indefinite"] else "")


def report(res):
    summ = res["random_summary"]
    for reg in ce.REGIMES:
        print(f"\n=== {reg} ({ce.REGIMES[reg]['title']}): covariance error median/max over models ===")
        blk = summ[reg]["all"]["all"]
        for iv in sorted(blk, key=int):
            row = blk[iv]
            print(f"  {row['value']:8.0e}  2F {_cell(row['2F']):24s}  2FJ {_cell(row['2FJ']):24s}"
                  f"  RTS {_cell(row['RTS'])}")
    # per case: 2FJ vs 2F
    r = []
    for c in res["random_cases"]:
        a, b = c["smoothers"]["2F"], c["smoothers"]["2FJ"]
        if a.get("cov") and b.get("cov"):
            r.append(np.log10(b["cov"] / a["cov"]))
    r = np.array(r)
    print(f"\nper case log10(err_2FJ / err_2F): median {np.median(r):+.2f}, "
          f"5%/95% {np.percentile(r, 5):+.2f}/{np.percentile(r, 95):+.2f}, "
          f"min {r.min():+.2f}, max {r.max():+.2f} (n={r.size})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rf-models", type=int, default=ce.RF_MODELS)
    ap.add_argument("--rf-seeds", type=int, default=ce.RF_SEEDS)
    ap.add_argument("--workers", type=int, default=min(10, os.cpu_count() or 1))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--from-json", action="store_true")
    ap.add_argument("--out", default=str(OUT_JSON))
    args = ap.parse_args()
    if args.from_json:
        res = json.loads(Path(args.out).read_text())
    else:
        t0 = time.time()
        n_rf = 2 if args.quick else args.rf_models
        tasks = []
        for reg in ce.REGIMES:
            vals = ce.RF_VALUES[reg]
            idx = [0, len(vals) - 1] if args.quick else range(len(vals))
            for gen in ce.GENS:
                for q in ce.QS:
                    for j in range(n_rf):
                        for i in idx:
                            tasks.append((reg, q, i, vals[i], args.rf_seeds, gen, j, True))
        lib = ce.library_info()
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            rcases = [f.result() for f in [ex.submit(ce.run_case, t) for t in tasks]]
        res = {"meta": {"script": "experiments/conditioning_exact_joseph.py", "library": lib,
                        "N": ce.N, "p": ce.P, "qs": list(ce.QS), "mp_dps": ce.DPS,
                        "models_per_generator_and_q": n_rf, "seeds": args.rf_seeds,
                        "quick": bool(args.quick), "runtime_s": time.time() - t0,
                        "workers": args.workers},
               "random_cases": ce._round_sig(ce._clean(rcases))}
        res["random_summary"] = ce._round_sig(ce._clean(ce.summarise_random(res["random_cases"])))
        ce.write_json(res, args.out)
        print(f"saved {args.out} (runtime {res['meta']['runtime_s']:.0f} s)")
    report(res)


if __name__ == "__main__":
    main()
