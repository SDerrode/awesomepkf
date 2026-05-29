# Sample datasets

Small real-world time series shipped with the repository for tutorials,
tests, and reproducibility of the parameter-learning examples.

## windfarms/

Hourly active-power and wind-speed measurements at a single onshore wind
turbine over October 2022 (586 hourly samples). Two columns are exposed:

- `ActivePower_KWh` — measured active power output.
- `WindSpeed` — co-located wind-speed measurement.

| file | description | size |
|------|-------------|------|
| `site1_202210_Month_586_orig.csv`  | raw values with timestamp index | ~24 KB |
| `site1_202210_Month_586_norm.csv`  | same data, mean-centred and unit-variance per column | ~40 KB |

Used by the `awesomepkf-fit-pkf` CLI and the real-data PKF notebook to
illustrate the method-of-moments estimator (see `prg/learning/`).

The full dataset (multiple sites, monthly/yearly granularity, plus
BuildingTemp and SeattleTemp series) is kept outside this repository — point
the `--data-filename` argument at a local copy if you need the complete
collection.
