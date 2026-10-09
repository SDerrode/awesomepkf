# Paper experiments

Self-contained scripts that regenerate the figures and tables of two companion papers:

- the full paper *Smoothing, Learning, and Testing the Gaussian Pairwise Markov Model*
  (figure/table numbers in the main table below);
- the smoothing letter *Six Smoothers for Gaussian Pairwise Markov Chains, and How to
  Choose One* (in preparation; see [the letter's mapping](#smoothing-letter)).

They import **only** the public library and are fully deterministic.

## Setup

From a clone of this repository:

```bash
python -m venv .venv && source .venv/bin/activate    # or use the shipped .venv
pip install -e .                                     # installs awesomepkf + deps
```

Run everything **from the repository root** (so `prg` is importable). The
`classical_vs_couple*`, `em_identification`, `lrt_vector`, `bench_smoothers` and
`discriminating_models` scripts import `prg`; `em_realdata` locates the datasets under
`data/` (override the root with the `AWESOMEPKF_ROOT` environment variable); `em_lrt.py`
is fully self-contained.

## Scripts, figures, and how to reproduce them

**The default parameters of each script are exactly the ones used for the published
figures** — run with no flags to reproduce them; pass the flags only to trade runtime
for precision.

| Script | Reproduces | Command (published settings) | Runtime |
|---|---|---|---|
| `pmm_schematic.py` | Fig. 1 (model schematic) | `python experiments/pmm_schematic.py` | ~2 s (matplotlib diagram) |
| `discriminating_models.py` | Fig. 2 + conditioning table (MBF safety) | `python experiments/discriminating_models.py` | ~20 s (drives the six library smoothers; also prints the q=2 check) |
| `bench_smoothers.py` | Fig. 3 (timing/memory) | `python experiments/bench_smoothers.py` | ~5 min (six smoothers, N up to 1.6e4) |
| `classical_vs_couple.py` | Fig. 4 | `python experiments/classical_vs_couple.py` | ~9 min (200 seeds × 9 couplings; `--replot` re-renders from the cache instantly) |
| `classical_vs_couple_multi.py` | Table IV | `python experiments/classical_vs_couple_multi.py` | ~5 min |
| `em_identification.py` | Fig. 5 | `python experiments/em_identification.py` | **~40 min** (50 seeds × 100 iters, N=2000; the E-step is the library smoother, not a hand-rolled one) |
| `em_lrt.py` | Fig. 6 | `python experiments/em_lrt.py` | ~1 min (direct-MLE null + power) |
| `em_realdata.py` | Fig. 7, Table V | `python experiments/em_realdata.py` | ~2 min (needs the chemostat/S&P data under `data/`) |
| `lrt_vector.py` | Remark 3 (vector `chi2_pq`) | `python experiments/lrt_vector.py` | ~8 min (library LRT, two cases) |
| `backaction_oscillator.py` | Fig. 8 (fitted poles) | `python experiments/backaction_oscillator.py` | ~15 min (40 realizations × 4 MLE fits × 2 systems); `... 8` for a quick look |
| `backaction_tradeoff.py` | Fig. 9 | `python experiments/backaction_tradeoff.py` | ~10 s (self-contained) |
| `petetin_kl_comparison.py` | Fig. 10 (estimability vs. testability, cf. Petetin–Desbouvries 2014) | `python experiments/petetin_kl_comparison.py` | ~5 s (self-contained) |
| `rmse_vs_lrt.py` | Fig. 11 (RMSE comparison vs. the test) | `python experiments/rmse_vs_lrt.py` | ~8 min (MC, 4 decision rules) |
| `missing_obs_ablation.py` | CHANGELOG numbers (missing-observation ablation: exact marginalisation vs the classical blkdiag recipe) | `python experiments/missing_obs_ablation.py` | ~3 min (600 runs: 200 couples × 3 gap rates, N=400) |

The "one estimate" check (Table III — all six smoothers agree to round-off) is reproduced
by `python -m prg.run_dwy_equivalence` (defaults `--N 500 --seeds 20`), with a shorter-horizon
walkthrough in `notebooks/tutorial_09_linear_smoothers.ipynb`; the learning/testing story of Figs. 5-6
is walked through in `notebooks/tutorial_10_learning_and_testing.ipynb`.

<a id="smoothing-letter"></a>
### Smoothing letter

The letter has its own four scripts (added in 2.16.0; `conditioning_exact_classical.py`
in 2.16.2). `conditioning_exact.py`, `conditioning_exact_classical.py` and
`classical_vs_pairwise_exact.py` write a JSON with every number next to the script and
their figures under `figures/` (both git-ignored); `check_elimination_full.py` only
prints its table of residuals.

| Letter | Command | Runtime |
|---|---|---|
| Fig. 1 (accuracy of the six smoothers against a 60-digit reference, four stress regimes, 80 random models each) | `python experiments/conditioning_exact.py --letter-fig` (full-width 2x2 panels; `--letter-layout 1x4` or `2x2` for the other layouts) | ~25 min on 10 cores (`--quick` ~1.5 min; `--from-json --letter-fig --no-figure` redraws instantly) |
| Sec. IV, classical models (the same study on 80 textbook models y = Hx + v per regime; same ranking and choice rule) | `python experiments/conditioning_exact_classical.py` (prints a side-by-side comparison if `conditioning_exact.py` was run first) | ~25 min on 10 cores (`--quick` ~1 min) |
| Table II (cost of ignoring back-action; exact MSE/NEES, no Monte Carlo) | `python experiments/classical_vs_pairwise_exact.py` | ~22 min (`--no-oracle --no-mc --no-capacity`: ~20 s, same Table II numbers) |
| Proposition 1 (three elimination orders; identities to round-off and to 1e-49 in 50-digit arithmetic, time-varying models) | `python experiments/check_elimination_full.py` | ~7 s (exit code 1 if an identity fails) |

The letter's 2F/DWY results require **2.16.0 or later**: earlier versions did not
symmetrise the backward filter that 2F and DWY share, and aborted (or returned wrong
results) under strongly correlated noise. The classical-model study requires **2.16.2 or
later**: earlier versions declared any covariance with condition number >= 1e12 invalid,
refused 26 of its models and regularised the filter covariances in place under the
broadest prior (spurious O(1) RTS errors).

Runtimes measured on a laptop from a clone (`pip install -e .`, Python 3.14).

## Expected results (to verify a run)

| Script | Key numbers you should see |
|---|---|
| `bench_smoothers.py` | all six **O(N)** (slopes 0.99–1.01), within **~24 %** in runtime (RTS fastest, 2F slowest); peak memory MBF ~33 MB (lightest) to DWY ~60 MB |
| `classical_vs_couple.py` | at ρ=1: best-fit classical MSE **+77 %**, naive ablation **+138 %**; pairwise NEES ≈ 0.99, refit ≈ 0.87 |
| `classical_vs_couple_multi.py` | at ρ=1, ΔMSE(refit) in **37–123 %** across (p,q) and noise (ablated up to **277 %**; x2y1 ×4: 70 %/160 %, NEES abl. 2.38); couple stays calibrated |
| `em_identification.py` | `A^xy = 0.395 ± 0.041`, `A^yy = 0.403 ± 0.022` (true 0.4/0.4); monotone log-likelihood |
| `em_lrt.py` | empirical size **0.037** at α=0.05, mean Λ ≈ 0.985 (χ²₁ mean 1); power rising to **1.0** by A^xy=0.45; predicted χ²₁(λ) curve overlays the empirical power |
| `em_realdata.py` | chemostat Λ **70.6/102.1**, S&P **95.2/280.5**, wind **3.3 (keep) / 13.6**, negative control (circularly shifted driver) **0.74 (keep)**; learned A^xy CI ≈ **[−0.50, −0.29]**, held-out **+10.3 %** vs classical (Clark–West t=2.90, p=0.002; plain DM is *not* significant) |
| `lrt_vector.py` | x2y2 (q=p=2): mean Λ ≈ 3.8 (dof pq=4), size ≈ 0.04 — tracks χ²₄; x2y1 (q=1<p): mean Λ ≈ 1 < pq, size ≈ 0.01 — conservative |
| `backaction_oscillator.py` | out-of-class oscillator: pairwise lowers held-out error **~35 %** (Diebold–Mariano p<1e-20), complex poles in **40/40** runs vs classical real **0/40**; in-class control (A^xy=0 truth): **≈0 %** (n.s.) |
| `backaction_tradeoff.py` | LRT power saturates by A^xy≈0.4; the classical state-MSE penalty keeps rising to ~20 % (testability ≠ estimability) |
| `petetin_kl_comparison.py` | Petetin Eq.(71) rises to **0.47** at R/Q=20 while the paper's KL_rate ≡ **0** on their A^yx=0 couple; KL_rate climbs **0→0.46** as A^yx opens a y-footprint |
| `rmse_vs_lrt.py` | H0 size: naive in-sample **1.00**, held-out **0.32**, DM **0.02**, LRT **0.04** (only DM/LRT calibrated); identity Λ≈N·log(MSE0/MSE1) corr **0.997** |
| `discriminating_models.py` | under noise starvation cond(P)→**5.9e9** (RTS), cond(Σ)→**3.3e7** (2F/DWY), cond(S)≡**1** (BF/MBF — but at q=1 that is a 1×1 tautology, not evidence), cond(R)≈1.5 (VAR); 2F/DWY smoothed-state error →**1.7e-6** (4.2e-7 before 2.16.0) while BF/MBF/VAR stay ≲**1e-12**. Block **R3** (p=q=2, where cond(S) is informative): cond(S) saturates near **8.7** while cond(P) reaches **1.2e7** at eps=1e-8. Before 2.16.0 the filter aborted there on a scale-dependent |det S_n| ≤ 1e-15 test, not on ill-conditioning (cond(S) ≈ 8.7) |
| `missing_obs_ablation.py` | naive blkdiag recipe vs exact marginalisation: median excess state RMSE **0.5–1.1 %** at 10–30 % gap rates, ninth decile **2.5–5.5 %**, worst couple **~80 %**; gap-step-only excess ≈ 0 (negative control — the damage lands on the steps *after* each gap); **0/200** diverged at every rate |

## Outputs

- `em_identification.py` → `experiments/em_coupling.pdf` (+ `.png` preview)
- `em_lrt.py` → `experiments/em_lrt.pdf` (+ `.png` preview)
- `classical_vs_couple.py` → `figures/classical_vs_couple.pdf` (+ `.png`; `--replot` re-renders from the cached `figures/classical_vs_couple_data.npz`)
- `classical_vs_couple_multi.py` → prints a table to stdout
- `lrt_vector.py` → `experiments/lrt_vector_both.json` (+ prints the size table)
- `backaction_oscillator.py` → `figures/backaction_poles.pdf` (+ `.png`; caches `figures/backaction_oscillator_data.npz`, `--replot` re-renders from it)
- `backaction_tradeoff.py` → `figures/backaction_two_quantities.pdf` (+ `.png` preview)
- `petetin_kl_comparison.py` → `figures/petetin_kl_comparison.pdf` (+ `.png` preview; self-contained, numpy/scipy/matplotlib)
- `rmse_vs_lrt.py` → `figures/rmse_vs_lrt.pdf` (+ `.png` preview; self-contained, numpy/scipy/matplotlib)
- `pmm_schematic.py` → `figures/pmm_schematic.pdf` (+ `.png` preview; a matplotlib graphical-model diagram, no numbers)
- `bench_smoothers.py` → `figures/bench_smoothers.pdf` (+ `.png` preview)
- `em_realdata.py` → `figures/em_realdata.pdf` (+ `.png` preview)
- `discriminating_models.py` → `figures/discriminating.pdf` (+ `.png` preview; prints the conditioning table)

These generated files are git-ignored; only the scripts are versioned.

## Library entry points

The partial-EM learning and the back-action test are also packaged as a reusable
API — see `prg.learning.em_partial_dynamics` (`estimate_dynamics_em`,
`back_action_lrt`) and, for the noise block, `prg.learning.em_partial_noise`. The
scripts above are frozen paper-reproduction drivers that implement the same partial
EM directly.
