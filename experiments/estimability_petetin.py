#!/usr/bin/env python
"""Estimability without testability on the pairwise model of Petetin and Desbouvries (Sec. VII).

Their model (IEEE TSP 2014, Eqs. (69)-(70), with d = a and the KL-optimal c), as in
petetin_kl_comparison.py: A = [[a, -b c], [0, a]] (A^yx = 0, so the back-action A^xy = -b c is
unobservable and no y-based test has power), with a = 0.8, b = 1, R = 1 and the state-noise
variance Q swept so that R/Q runs from 0.05 to 20. For each R/Q we compute, exactly as in
Sec. III (N -> infinity Wiener smoother under the true law, `asym_metrics`), the state-MSE
penalty over the pairwise smoother of: the ablated model (A^xy set to 0), the likelihood fit
(KL projection onto C, x observed in training), and the best stable member of C (Nelder--Mead
search, fast grid evaluation of capacity_counterexample.py, exact re-evaluation).

Run (from the paper directory): python -B experiments/estimability_petetin.py
Output: experiments/estimability_petetin_results.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import capacity_counterexample as cc  # noqa: E402
import classical_vs_pairwise_exact as cv  # noqa: E402
import petetin_kl_comparison as pk  # noqa: E402

OUT_JSON = Path(__file__).resolve().parent / "estimability_petetin_results.json"


def main():
    a, b, R = 0.8, 1.0, 1.0
    rows = []
    rng = np.random.default_rng(3)
    for ratio in (0.05, 0.2, 1.0, 5.0, 20.0):
        Q = R / ratio
        At, Rt = pk.petetin_couple(a, b, Q, R)
        Rt = 0.5 * (Rt + Rt.T)
        base = cv.asym_metrics(At, Rt, At, Rt, 1)["mse"]
        row = {"R_over_Q": ratio, "Axy": float(At[0, 1]),
               "kl_rate_test": pk.paper_kl(At, Rt)}
        for name, (Am, Rm) in (("ablated", cv.ablate(At, Rt, 1)),
                               ("likelihood_fit", cv.project_classC_closed(At, Rt, 1, 1))):
            row[name] = 100 * (cv.asym_metrics(At, Rt, Am, Rm, 1)["mse"] / base - 1)
        row["best_stable_C"] = cc.best_stable_C(At, Rt, rng)[1]
        rows.append(row)
        print(f"R/Q={ratio:5.2f}  A^xy={row['Axy']:+.3f}  KL_rate(test)={row['kl_rate_test']:.1e}  "
              f"ablated {row['ablated']:6.1f}%  likelihood fit {row['likelihood_fit']:5.1f}%  "
              f"best stable C {row['best_stable_C']:5.1f}%", flush=True)
    OUT_JSON.write_text(json.dumps({"a": a, "b": b, "R": R, "rows": rows}, indent=1))
    print(f"saved {OUT_JSON}")


if __name__ == "__main__":
    main()
