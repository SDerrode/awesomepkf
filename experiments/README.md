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
`classical_vs_couple*` scripts import `prg` directly; the `em_*` scripts locate the
repository root themselves (override with the `AWESOMEPKF_ROOT` environment variable).

## Scripts, figures, and how to reproduce them

**The default parameters of each script are exactly the ones used for the published
figures** — run with no flags to reproduce them; pass the flags only to trade runtime
for precision.

| Script | Reproduces | Command (published settings) | Runtime |
|---|---|---|---|
| `classical_vs_couple.py` | Fig. 1, Table III | `python experiments/classical_vs_couple.py` | ~1 min |
| `classical_vs_couple_multi.py` | Table III | `python experiments/classical_vs_couple_multi.py` | ~2 min |
| `em_identification.py` | Fig. 2 | `python experiments/em_identification.py` | ~3 min (50 seeds, N=2000) |
| `em_lrt.py` | Fig. 3 | `python experiments/em_lrt.py` | ~30 min (350 seeds); `--seeds 40` for a quick look |

The "one estimate" check (Table II — all six smoothers agree to round-off) is in
`notebooks/tutorial_09_linear_smoothers.ipynb`; the learning/testing story of Figs. 2-3
is walked through in `notebooks/tutorial_10_learning_and_testing.ipynb`.

## Expected results (to verify a run)

| Script | Key numbers you should see |
|---|---|
| `classical_vs_couple.py` | at ρ=1: best-fit classical MSE **+77 %**, naive ablation **+138 %**; couple NEES ≈ 0.99, refit ≈ 0.87 |
| `classical_vs_couple_multi.py` | at ρ=1, ΔMSE(refit) in **37–123 %** across (p,q) and noise; couple stays calibrated |
| `em_identification.py` | `A^xy = 0.383 ± 0.034`, `A^yy = 0.408 ± 0.020` (true 0.4/0.4); monotone log-likelihood |
| `em_lrt.py` | empirical size **0.049** at α=0.05, mean Λ ≈ 0.92 (χ²₁ mean 1); power rising to **1.0** by A^xy=0.45 |

## Outputs

- `em_identification.py` → `experiments/em_coupling.png`
- `em_lrt.py` → `experiments/em_lrt.png`
- `classical_vs_couple.py` → `figures/classical_vs_couple.pdf`
- `classical_vs_couple_multi.py` → prints a table to stdout

These generated files are git-ignored; only the scripts are versioned.

## Library entry points

The partial-EM learning and the back-action test are also packaged as a reusable
API — see `prg.learning.em_partial_dynamics` (`estimate_dynamics_em`,
`back_action_lrt`) and, for the noise block, `prg.learning.em_partial_noise`. The
scripts above are frozen paper-reproduction drivers that implement the same partial
EM directly.
