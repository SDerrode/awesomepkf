# Paper experiments — *Smoothing, Learning, and Testing the Gaussian Pairwise Markov Model*

Self-contained scripts that regenerate the figures and tables of the companion paper.
They import **only** the public library and are fully deterministic.

## Setup

From a clone of this repository:

```bash
python -m venv .venv && source .venv/bin/activate    # or use the shipped .venv
pip install -e .                                     # installs awesomepkf + deps
```

Run everything **from the repository root** (so `prg` is importable). The
`classical_vs_couple*`, `em_identification` and `lrt_vector` scripts import `prg`
(the last two locate the repository root themselves — override with the
`AWESOMEPKF_ROOT` environment variable); `em_lrt.py` is fully self-contained.

## Scripts, figures, and how to reproduce them

**The default parameters of each script are exactly the ones used for the published
figures** — run with no flags to reproduce them; pass the flags only to trade runtime
for precision.

| Script | Reproduces | Command (published settings) | Runtime |
|---|---|---|---|
| `pmm_schematic.py` | Fig. 1 (model schematic) | `python experiments/pmm_schematic.py` | ~2 s (matplotlib diagram) |
| `classical_vs_couple.py` | Fig. 2, Table IV | `python experiments/classical_vs_couple.py` | ~1 min |
| `classical_vs_couple_multi.py` | Table IV | `python experiments/classical_vs_couple_multi.py` | ~2 min |
| `em_identification.py` | Fig. 3 | `python experiments/em_identification.py` | ~3 min (50 seeds × 100 iters, N=2000) |
| `em_lrt.py` | Fig. 4 | `python experiments/em_lrt.py` | ~1 min (direct-MLE null + power) |
| `lrt_vector.py` | Remark 3 (vector `chi2_pq`) | `python experiments/lrt_vector.py` | ~8 min (library LRT, two cases) |
| `backaction_oscillator.py` | Fig. 5 | `python experiments/backaction_oscillator.py` | ~15 min (40 realisations × 4 MLE fits × 2 systems); `... 8` for a quick look |
| `backaction_tradeoff.py` | Fig. 6 | `python experiments/backaction_tradeoff.py` | ~10 s (self-contained) |

The "one estimate" check (Table III — all six smoothers agree to round-off) is in
`notebooks/tutorial_09_linear_smoothers.ipynb`; the learning/testing story of Figs. 2-3
is walked through in `notebooks/tutorial_10_learning_and_testing.ipynb`.

## Expected results (to verify a run)

| Script | Key numbers you should see |
|---|---|
| `classical_vs_couple.py` | at ρ=1: best-fit classical MSE **+77 %**, naive ablation **+138 %**; couple NEES ≈ 0.99, refit ≈ 0.87 |
| `classical_vs_couple_multi.py` | at ρ=1, ΔMSE(refit) in **37–123 %** across (p,q) and noise; couple stays calibrated |
| `em_identification.py` | `A^xy = 0.395 ± 0.041`, `A^yy = 0.403 ± 0.022` (true 0.4/0.4); monotone log-likelihood |
| `em_lrt.py` | empirical size **0.037** at α=0.05, mean Λ ≈ 0.985 (χ²₁ mean 1); power rising to **1.0** by A^xy=0.45 |
| `lrt_vector.py` | x2y2 (q=p=2): mean Λ ≈ 3.8 (dof pq=4), size ≈ 0.04 — tracks χ²₄; x2y1 (q=1<p): mean Λ ≈ 1 < pq, size ≈ 0.01 — conservative |
| `backaction_oscillator.py` | out-of-class oscillator: couple lowers held-out error **~35 %** (Diebold–Mariano p<1e-20), complex poles in **40/40** runs vs classical real **0/40**; in-class control (A^xy=0 truth): **≈0 %** (n.s.) |
| `backaction_tradeoff.py` | LRT power saturates by A^xy≈0.4; the classical state-MSE penalty keeps rising to ~20 % (testability ≠ estimability) |

## Outputs

- `em_identification.py` → `experiments/em_coupling.pdf` (+ `.png` preview)
- `em_lrt.py` → `experiments/em_lrt.pdf` (+ `.png` preview)
- `classical_vs_couple.py` → `figures/classical_vs_couple.pdf` (+ `.png`; `--replot` re-renders from the cached `figures/classical_vs_couple_data.npz`)
- `classical_vs_couple_multi.py` → prints a table to stdout
- `lrt_vector.py` → `experiments/lrt_vector_both.json` (+ prints the size table)
- `backaction_oscillator.py` → `figures/backaction_poles.pdf` (+ `.png`; caches `figures/backaction_oscillator_data.npz`, `--replot` re-renders from it)
- `backaction_tradeoff.py` → `figures/backaction_two_quantities.pdf` (+ `.png` preview)
- `pmm_schematic.py` → `figures/pmm_schematic.pdf` (+ `.png` preview; a matplotlib graphical-model diagram, no numbers)

These generated files are git-ignored; only the scripts are versioned.

## Library entry points

The partial-EM learning and the back-action test are also packaged as a reusable
API — see `prg.learning.em_partial_dynamics` (`estimate_dynamics_em`,
`back_action_lrt`) and, for the noise block, `prg.learning.em_partial_noise`. The
scripts above are frozen paper-reproduction drivers that implement the same partial
EM directly.
