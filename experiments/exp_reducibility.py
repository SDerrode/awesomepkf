#!/usr/bin/env python3
"""Reducibility of the back-action gauge (paper App. E, "When the back-action is a gauge").

For a stable pairwise transition A (blocks x=latent p, y=observed q), a shear L that removes
the back-action is a real solution of the nonsymmetric algebraic Riccati equation (App. E)

    A^xy + L A^yy - A^xx L - L A^yx L = 0 ,           (NARE)

equivalently (Prop. E.1) a real A-invariant q-dimensional subspace transverse to X. This
script enumerates those solutions via invariant subspaces and reports, per (p,q) cell:

  * the fraction of draws admitting >= 1 real solution (the back-action is a removable gauge),
    in two regimes -- iid N(0,1) entries, and "conditioned on stability" -- which coincide
    because reducibility depends only on eigenvector geometry and the real/complex pattern,
    both invariant under the positive rescaling A -> A/(rho*1.05) used for the stable regime;
  * the mean number of real solutions;
  * median / 99th-percentile / max Frobenius norm of the least-norm solution;
  * (noise) among reducible draws with Q > 0, the fraction whose reduced form (A^xy=0) reaches
    Q^xy = 0 -- i.e. lands in the strictly classical subclass -- via Q^xy -> Q^xy + L Q^yy.

Reproduces the numbers quoted in App. E. Anchor: at p=q=1 the fraction is 1/sqrt(2) ~ 0.707,
the probability that a 2x2 iid-Gaussian matrix has real eigenvalues (Edelman).

numpy only; no awesomepkf dependency. Writes reducibility_results.json next to this file.
Usage:  python exp_reducibility.py [--ndraw 4000] [--seed 0]
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


def real_solutions(A, p, q, tol=1e-8):
    """Real L (p x q) solving the NARE, via A-invariant q-dim subspaces transverse to X."""
    w, V = np.linalg.eig(A)
    reals, pairs, used = [], [], np.zeros(len(w), bool)
    for i in np.argsort(w.imag):
        if used[i]:
            continue
        if abs(w[i].imag) < tol:
            reals.append(i)
            used[i] = True
        else:
            j = min((k for k in range(len(w)) if not used[k] and k != i
                     and abs(w[k] - np.conj(w[i])) < 1e-6), default=None)
            if j is None:
                return None                      # numerically ambiguous spectrum; skip draw
            pairs.append((i, j))
            used[i] = used[j] = True
    sols, nr, npair = [], len(reals), len(pairs)
    for r in range(nr + 1):
        c = q - r
        if c < 0 or c % 2 or c // 2 > npair:
            continue
        for rr in itertools.combinations(range(nr), r):
            for pp in itertools.combinations(range(npair), c // 2):
                cols = [V[:, reals[i]].real for i in rr]
                for k in pp:
                    a, _ = pairs[k]
                    cols += [V[:, a].real, V[:, a].imag]
                B = np.column_stack(cols)
                BY = B[p:, :]
                if abs(np.linalg.det(BY)) < tol:
                    continue                     # not transverse to X (no graph)
                sols.append(-B[:p, :] @ np.linalg.inv(BY))
    return sols


def _draw_A(rng, d, stable):
    A = rng.standard_normal((d, d))
    return A / (np.abs(np.linalg.eigvals(A)).max() * 1.05) if stable else A


def _draw_Q(rng, d):
    M = rng.standard_normal((d, d))
    return M @ M.T + 1e-3 * d * np.eye(d)


def run_cell(p, q, ndraw, seed):
    d = p + q
    res = {"p": p, "q": q}
    for tag, stable in (("iid", False), ("stable", True)):
        rng = np.random.default_rng(seed)
        hit, counts, least = 0, [], []
        for _ in range(ndraw):
            s = real_solutions(_draw_A(rng, d, stable), p, q)
            if s is None:
                continue
            counts.append(len(s))
            if s:
                hit += 1
                least.append(min(np.linalg.norm(L) for L in s))
        counts, least = np.array(counts), np.array(least)
        res[tag] = {"frac": hit / len(counts), "mean_n": float(counts.mean()),
                    "med": float(np.median(least)), "p99": float(np.percentile(least, 99)),
                    "mx": float(least.max())}
    # noise: fraction of reducible draws reaching the strictly classical Q^xy = 0
    rng = np.random.default_rng(seed + 1)
    reduc, qxy0 = 0, 0
    for _ in range(ndraw):
        A = _draw_A(rng, d, False)
        s = real_solutions(A, p, q)
        if not s:
            continue
        reduc += 1
        Q = _draw_Q(rng, d)
        Qxy, Qyy = Q[:p, p:], Q[p:, p:]
        if any(np.linalg.norm(Qxy + L @ Qyy) < 1e-6 for L in s):
            qxy0 += 1
    res["frac_Qxy0"] = qxy0 / max(reduc, 1)
    return res


def main(ndraw=4000, seed=0):
    grid = [(1, 1), (2, 1), (1, 2), (2, 2), (3, 3), (3, 5), (5, 3), (4, 4)]
    rows = [run_cell(p, q, ndraw, seed) for (p, q) in grid]
    hdr = f"{'(p,q)':>7} | {'frac_iid':>8} {'frac_stab':>9} {'mean_n':>7} " \
          f"{'med|L|':>7} {'p99':>7} {'max':>10} {'Qxy=0':>7}"
    print(hdr)
    for r in rows:
        i = r["iid"]
        print(f"({r['p']},{r['q']})".rjust(7) + f" | {i['frac']:8.3f} "
              f"{r['stable']['frac']:9.3f} {i['mean_n']:7.2f} {i['med']:7.2f} "
              f"{i['p99']:7.1f} {i['mx']:10.0f} {r['frac_Qxy0']:7.3f}")
    out = Path(__file__).resolve().parent / "reducibility_results.json"
    out.write_text(json.dumps({"ndraw": ndraw, "seed": seed, "cells": rows}, indent=1))
    print("\nsaved", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ndraw", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.ndraw, a.seed)
