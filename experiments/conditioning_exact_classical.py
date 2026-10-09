#!/usr/bin/env python
"""Accuracy study of conditioning_exact.py, re-run on CLASSICAL (textbook) models.

Same engine (six awesomepkf smoothers, 60-digit lifted reference, same metrics, same
statistics, same four regimes and sweep values), but the random families are textbook
state-space models written as GPMCs:

    x_{n+1} = F x_n + w_{n+1},  w ~ N(0, Q);   y_n = H x_n + v_n,  v ~ N(0, Rv),

i.e. Z_{n+1} = A Z_n + W_{n+1} with A = [[F, 0], [H F, 0]] (no back-action, A^xy = 0, and
A^yy = 0) and R = [[Q, Q H^T], [H Q, H Q H^T + Rv]].

Generators (20 models per (generator, q), q = 1, 2, p = 2, 3 seeds per model):
  T1: F Gaussian rescaled to rho ~ U[0.6, 0.9] (relative Henrici departure > 0.3), H Gaussian,
      Q and Rv random correlation matrices (normalised Wishart), unit variances.
  T2: F = U T U^T (Schur form, real eigenvalues ~ U[-0.85, 0.85], max |lambda| >= 0.5,
      strictly-upper entries N(0, s^2), s ~ U[0.3, 1]), H Gaussian with column scales
      logU[0.5, 2], Q and Rv with random spectra and standard deviations ~ logU[0.5, 2].
Regimes, mapped on the physical parameters:
  D  broad prior: Pz0 = s I, s = 1 .. 1e10 (data from the stationary law);
  S  vanishing state noise: Q -> eps Q, eps = 1 .. 1e-10;
  C  strongly correlated noise: Rv -> delta Rv, delta = 1e-1 .. 1e-10 (nearly noiseless
     measurements: the largest canonical correlation c between the X- and Y-noise channels
     of the pairwise form tends to 1; 1-c is recorded per model);
  M  poles near the unit circle (regime U in the letter): F -> (rho / rho(F)) F,
     rho = 0.9 .. 0.9999.
In S, C and M the smoother prior is the stationary law, as in conditioning_exact.py.

Requires awesomepkf >= 2.16.2: earlier versions declared any covariance with condition
number >= 1e12 invalid, refused 26 of these models at construction and "regularised" the
filter covariances in place under the broadest prior (spurious O(1) RTS errors).

Run (from a clone of awesomePKF, ~25 min on 10 cores; --quick ~1 min):
  python experiments/conditioning_exact_classical.py
Output: experiments/conditioning_exact_classical_results.json and, if
experiments/conditioning_exact_results.json exists (conditioning_exact.py), a printed
side-by-side comparison with the pairwise study.
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

import conditioning_exact as ce  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT_JSON = HERE / "conditioning_exact_classical_results.json"
PAIRWISE_JSON = HERE / "conditioning_exact_results.json"
P = ce.P


def textbook_t1(q, j):
    rng = np.random.default_rng(90_000 + 100 * q + j)
    while True:
        F = rng.standard_normal((P, P))
        F *= rng.uniform(0.6, 0.9) / ce.spectral_radius(F)
        if ce.henrici(F) > 0.3:
            break
    H = rng.standard_normal((q, P))
    return {"F": F, "H": H, "Q": ce._random_corr(rng, P), "Rv": ce._random_corr(rng, q)}


def textbook_t2(q, j):
    rng = np.random.default_rng(95_000 + 100 * q + j)
    while True:
        lam = rng.uniform(-0.85, 0.85, P)
        if np.max(np.abs(lam)) >= 0.5:
            break
    T = np.diag(lam) + np.triu(rng.standard_normal((P, P)), 1) * rng.uniform(0.3, 1.0)
    U = ce._haar_orthogonal(rng, P)
    F = U @ T @ U.T
    H = rng.standard_normal((q, P)) * np.exp(rng.uniform(np.log(0.5), np.log(2.0), P))[None, :]
    sq = np.exp(rng.uniform(np.log(0.5), np.log(2.0), P))
    sv = np.exp(rng.uniform(np.log(0.5), np.log(2.0), q))
    Q = sq[:, None] * ce._random_corr_spectrum(rng, P) * sq[None, :]
    Rv = sv[:, None] * ce._random_corr_spectrum(rng, q) * sv[None, :]
    return {"F": F, "H": H, "Q": 0.5 * (Q + Q.T), "Rv": 0.5 * (Rv + Rv.T)}


TEXTBOOK = {"T1": textbook_t1, "T2": textbook_t2}


def pairwise_form(F, H, Q, Rv):
    q = H.shape[0]
    A = np.block([[F, np.zeros((P, q))], [H @ F, np.zeros((q, q))]])
    R = np.block([[Q, Q @ H.T], [H @ Q, H @ Q @ H.T + Rv]])
    return A, 0.5 * (R + R.T)


def canonical_corr(R, q):
    Rxx, Rxy, Ryy = R[:P, :P], R[:P, P:], R[P:, P:]
    Wx = np.linalg.inv(np.linalg.cholesky(Rxx))
    Wy = np.linalg.inv(np.linalg.cholesky(Ryy))
    return float(np.linalg.norm(Wx @ Rxy @ Wy.T, 2))


def build_model_textbook(regime, q, value, gen=None, model_id=None):
    sp = TEXTBOOK[gen](q, model_id)
    F, H, Q, Rv = sp["F"], sp["H"], sp["Q"], sp["Rv"]
    if regime == "S":
        Q = value * Q
    elif regime == "C":
        Rv = value * Rv
    elif regime == "M":
        F = F * ((1.0 - value) / ce.spectral_radius(F))
    A, R = pairwise_form(F, H, Q, Rv)
    Sig_inf = ce.stationary_cov(A, R)
    Pz0 = value * np.eye(P + q) if regime == "D" else Sig_inf.copy()
    return {"A": A, "R": R, "Pz0": Pz0, "Pz0_data": Sig_inf}


# Patch the engine (module level, so that spawned worker processes patch it too).
ce.build_model = build_model_textbook
ce.GENS = tuple(TEXTBOOK)
ce.GENERATORS = TEXTBOOK


def compare(cl, pw):
    """Print, per regime and sweep value, pooled (all generators, all q) median / max
    covariance error, aborts and indefinite counts: pairwise vs classical."""
    for reg in ce.REGIMES:
        print(f"\n=== regime {reg} ({ce.REGIMES[reg]['title']}) ===")
        cb, pb = cl[reg]["all"]["all"], pw[reg]["all"]["all"]
        for iv in sorted(cb, key=int):
            v = cb[iv]["value"]
            line = [f"{v:8.0e}"]
            for name in ce.NAMES:
                c, p = cb[iv][name], pb[iv][name]
                def fmt(x):
                    cov = x.get("cov")
                    med = f"{cov['median']:.0e}/{cov['max']:.0e}" if cov else "   --   "
                    tags = (f" a{x['n_abort']}" if x["n_abort"] else "") + \
                           (f" i{x['n_indefinite']}" if x["n_indefinite"] else "")
                    return med + tags
                line.append(f"{name}: pw {fmt(p)} | cl {fmt(c)}")
            print("  " + "\n           ".join(line))


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
            for gen in TEXTBOOK:
                for q in ce.QS:
                    for j in range(n_rf):
                        for i in idx:
                            tasks.append((reg, q, i, vals[i], args.rf_seeds, gen, j, True))
        # The library refuses to build a model whose R or Pz0 fails its covariance check
        # (cond >= 1e12): such sweep points are skipped and listed.
        skipped, kept = [], []
        for t in tasks:
            reg, q, i, v, _, gen, j, _ = t
            try:
                ce.make_param(build_model_textbook(reg, q, v, gen, j), q)
                kept.append(t)
            except ValueError as e:
                m = build_model_textbook(reg, q, v, gen, j)
                skipped.append({"regime": reg, "q": q, "value": v, "gen": gen, "model_id": j,
                                "cond_R": float(np.linalg.cond(m["R"])), "error": str(e)})
        tasks = kept
        print(f"{len(skipped)} sweep points refused by the library:", skipped)
        lib = ce.library_info()
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            rcases = [f.result() for f in [ex.submit(ce.run_case, t) for t in tasks]]
            cross = [f.result() for f in
                     [ex.submit(ce.crosscheck_case, t) for t in ce.pick_random_crosschecks(rcases)]]
        one_minus_c = {}
        for gen in TEXTBOOK:
            for q in ce.QS:
                for j in range(n_rf):
                    for v in ce.RF_VALUES["C"]:
                        m = build_model_textbook("C", q, v, gen, j)
                        one_minus_c.setdefault(f"{v:g}", []).append(1.0 - canonical_corr(m["R"], q))
        res = {"meta": {"script": "experiments/conditioning_exact_classical.py",
                        "library": lib, "N": ce.N, "p": P, "qs": list(ce.QS), "mp_dps": ce.DPS,
                        "models_per_generator_and_q": n_rf, "seeds": args.rf_seeds,
                        "generators": {g: f.__doc__ or "" for g, f in TEXTBOOK.items()},
                        "values": ce.RF_VALUES, "quick": bool(args.quick),
                        "runtime_s": time.time() - t0, "skipped_by_library": skipped, "workers": args.workers,
                        "regime_C_one_minus_c": {k: [float(np.min(v)), float(np.median(v)),
                                                     float(np.max(v))]
                                                 for k, v in one_minus_c.items()}},
               "random_cases": ce._round_sig(ce._clean(rcases)),
               "crosscheck_random": ce._clean(cross)}
        res["random_summary"] = ce._round_sig(ce._clean(ce.summarise_random(res["random_cases"])))
        ce.write_json(res, args.out)
        print(f"saved {args.out} (runtime {res['meta']['runtime_s']:.0f} s)")
    print("regime C, 1-c per delta (min/median/max):", res["meta"]["regime_C_one_minus_c"])
    worst = max(v for x in res["crosscheck_random"] for k, d in x.items()
                if isinstance(d, dict) and k != "model" for v in d.values()
                if isinstance(v, float))
    print(f"reference cross-checks: largest discrepancy {worst:.1e}")
    if PAIRWISE_JSON.exists():   # side-by-side with the pairwise study, if it was run
        compare(res["random_summary"], json.loads(PAIRWISE_JSON.read_text())["random_summary"])
    else:
        print(f"(run conditioning_exact.py first to compare with {PAIRWISE_JSON.name})")


if __name__ == "__main__":
    main()
