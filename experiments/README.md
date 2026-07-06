# Paper experiments — *Smoothing, Learning, and Testing the Gaussian Pairwise Markov Model*

Self-contained scripts reproducing the figures and tables of the companion paper.
Each script uses only the public library (`prg.classes.linear_pks`,
`prg.classes.linear_pkf`, …); the `classical_vs_couple*` scripts must be run from the
repository root (so that `prg` is importable), while the `em_*` scripts locate the
repository root themselves (`AWESOMEPKF_ROOT` overrides the search).

| Script | Reproduces | What it does |
|---|---|---|
| `classical_vs_couple.py` | Fig. 1, Table III | Sweeps coupling ρ; MSE + NEES of the couple smoother vs. the ablated and best-fit classical smoothers on the same record. |
| `classical_vs_couple_multi.py` | Table III | Repeats the ρ=1 comparison across latent/observed dimensions and observation-noise levels. |
| `em_identification.py` | Fig. 2 | Two-block **partial EM** recovering the couple coefficients (back-action `A^{xy}`, observation memory `A^{yy}`) from the classical initialisation `(0, 0)`; writes `em_coupling.png`. |
| `em_lrt.py` | Fig. 3 | Likelihood-ratio **test** of `H0: A^{xy}=0` (back-action absent): empirical null vs. χ²₁ and power vs. true back-action; writes `em_lrt.png`. |

The "one estimate" check (Table II — all six smoothers agree to round-off) is in
`notebooks/tutorial_09_linear_smoothers.ipynb`.

```bash
# from the repository root
python experiments/classical_vs_couple.py
python experiments/classical_vs_couple_multi.py
python experiments/em_identification.py      # ~3 min (50 seeds, N=2000)
python experiments/em_lrt.py                 # long (350 seeds); use --seeds 40 for a quick look
```

Each script exposes CLI flags (`--seeds`, `--N`, …) to trade runtime for precision.
The `em_*` scripts fix the seeds they sweep, so figures regenerate deterministically.
