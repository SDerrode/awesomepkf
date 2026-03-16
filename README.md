# AwesomePKF

This repository contains a set of programs illustrating the **Pairwise Kalman Filter (PKF)**, a generalization of the classical Kalman Filter, extended to non-linear models. It includes several variants of non-linear filters:

- **Extended Pairwise Kalman Filter (EPKF)**
- **Unscented Pairwise Kalman Filter (UPKF)**, with multiple variants depending on the choice of sigma points
- **Pairwise Particle Filter (PPF)**

---

## Table of Contents

- [AwesomePKF](#awesomepkf)
    - [Table of Contents](#table-of-contents)
    - [Installation](#installation)
    - [Models and Simulations](#models-and-simulations)
    - [Filters](#filters)
        - [Pairwise Kalman Filter (PKF)](#pairwise-kalman-filter-pkf)
        - [Extended Pairwise Kalman Filter (EPKF)](#extended-pairwise-kalman-filter-epkf)
        - [Unscented Pairwise Kalman Filter (UPKF)](#unscented-pairwise-kalman-filter-upkf)
        - [Pairwise Particle Filter (PPF)](#pairwise-particle-filter-ppf)
    - [Usage Examples](#usage-examples)
        - [Simulate Linear Data and Filter with PKF](#simulate-linear-data-and-filter-with-pkf)
        - [Simulate Non-Linear Data and Filter with EPKF, UPKF and PPF](#simulate-non-linear-data-and-filter-with-epkf-upkf-and-ppf)
    - [Folders structure](#folders-structure)

---

## Installation

### From PyPI (recommended)

```bash
pip install awesomepkf
```

### From source

```bash
git clone https://github.com/sderrode/awesomepkf.git
cd awesomepkf
pip install .
```

### Development install

```bash
pip install -e ".[dev]"
```

### Requirements

- Python >= 3.10
- numpy, scipy, matplotlib, pandas, rich, sympy

---

## Quick Start

```python
from prg.classes.Linear_PKF import Linear_PKF
from prg.models.linear.model_x1_y1_AQ_pairwise import Model_x1_y1_AQ_pairwise

model = Model_x1_y1_AQ_pairwise()
pkf = Linear_PKF(model)
# ... run the filter step by step
```

Or use the CLI entry points installed with the package:

```bash
awesomepkf-simulate --N 2000 --linearModelName "model_x1_y1_AQ_pairwise" --dataFileName "testL.csv" --sKey 303
awesomepkf-pkf      --linearModelName "model_x1_y1_AQ_pairwise" --dataFileName "testL.csv" --plot
```

---

## Models and Simulations

The repository provides a program called **run_simulator.py** to simulate data according to **linear and non-linear models**.

---

## Filters

Each filter has two types of programs:

1. Simulate data **and filter it directly**  
2. Filter data **from a previously saved file**  

### Pairwise Kalman Filter (PKF)

- **run_linear_pkf.py** – filter linear data either from simulated data or from a previously saved file (e.g., generated with `run_simulator.py`)  

### Extended Pairwise Kalman Filter (EPKF)

- **run_nonlinear_epkf.py** – filter non-linear data either from simulated data or from a previously saved file (e.g., generated with `run_simulator.py`)  

### Unscented Pairwise Kalman Filter (UPKF)

- **run_nonlinear_upkf.py** – filter non-linear data either from simulated data or from a previously saved file (e.g., generated with `run_simulator.py`)  

### Pairwise Particle Filter (PPF)

- **run_nonlinear_ppf.py** – filter non-linear data either from simulated data or from a previously saved file (e.g., generated with `run_simulator.py`)  

---

## Usage Examples

### Simulate Linear Data and Filter with PKF

```bash
awesomepkf-simulate --N 2000 --linearModelName "model_x1_y1_AQ_pairwise" --dataFileName "testL.csv" --verbose 1 --sKey 303
awesomepkf-pkf      --linearModelName "model_x1_y1_AQ_pairwise" --dataFileName "testL.csv" --verbose 1 --saveHistory --plot
```

### Simulate Non-Linear Data and Filter with EPKF, UPKF and PPF

```bash
awesomepkf-simulate --N 1000 --nonLinearModelName "model_x2_y1_pairwise" --dataFileName "testNL.csv" --verbose 1 --sKey 303

awesomepkf-epkf --nonLinearModelName "model_x2_y1_pairwise" --dataFileName "testNL.csv"                      --verbose 1 --saveHistory --plot
awesomepkf-upkf --nonLinearModelName "model_x2_y1_pairwise" --dataFileName "testNL.csv" --sigmaSet "wan2000"  --verbose 1 --saveHistory --plot
awesomepkf-ppf  --nonLinearModelName "model_x2_y1_pairwise" --dataFileName "testNL.csv" --nbParticles 300     --verbose 1 --saveHistory --plot
```

---

## Folders structure

<!-- PROJECT_STRUCTURE_START -->
```text
./
├── data/
│   ├── datafile/
│   ├── historyTracker/
│   ├── plot/
│   └── clean_dirs.sh
├── ipynb/
│   └── readme.md
├── prg/
│   ├── base_classes/
│   │   ├── __init__.py
│   │   ├── linear_pkf_runner_base.py
│   │   ├── linear_pkf_runner_from_file.py
│   │   ├── linear_pkf_runner_simulation.py
│   │   ├── nonlinear_epkf_runner_base.py
│   │   ├── nonlinear_epkf_runner_from_file.py
│   │   ├── nonlinear_epkf_runner_simulation.py
│   │   ├── nonlinear_ppf_runner_base.py
│   │   ├── nonlinear_ppf_runner_from_file.py
│   │   ├── nonlinear_ppf_runner_simulation.py
│   │   ├── nonlinear_ukf_runner_base.py
│   │   ├── nonlinear_ukf_runner_from_file.py
│   │   ├── nonlinear_ukf_runner_simulation.py
│   │   ├── nonlinear_upkf_runner_base.py
│   │   ├── nonlinear_upkf_runner_from_file.py
│   │   ├── nonlinear_upkf_runner_simulation.py
│   │   ├── runner_base.py
│   │   ├── simulator_base.py
│   │   ├── simulator_linear.py
│   │   └── simulator_nonlinear.py
│   ├── classes/
│   │   ├── HistoryTracker.py
│   │   ├── Linear_PKF.py
│   │   ├── MatrixDiagnostics.py
│   │   ├── NonLinear_EPKF.py
│   │   ├── NonLinear_PPF.py
│   │   ├── NonLinear_UKF.py
│   │   ├── NonLinear_UKF_CN_Linearized.py
│   │   ├── NonLinear_UKF_UN.py
│   │   ├── NonLinear_UPKF.py
│   │   ├── PKF.py
│   │   ├── ParamLinear.py
│   │   ├── ParamNonLinear.py
│   │   ├── SeedGenerator.py
│   │   ├── SigmaPointsSet.py
│   │   └── __init__.py
│   ├── models/
│   │   ├── linear/
│   │   │   ├── __init__.py
│   │   │   ├── base_model_linear.py
│   │   │   ├── model_x1_y1_AQ_augmented.py
│   │   │   ├── model_x1_y1_AQ_classic.py
│   │   │   ├── model_x1_y1_AQ_pairwise.py
│   │   │   ├── model_x1_y1_Sigma_pairwise.py
│   │   │   ├── model_x2_y2_AQ_pairwise.py
│   │   │   ├── model_x2_y2_Sigma_pairwise.py
│   │   │   ├── model_x3_y1_AQ_pairwise.py
│   │   │   └── model_x3_y1_Sigma_pairwise.py
│   │   ├── nonLinear/
│   │   │   ├── __init__.py
│   │   │   ├── base_model_fxhx.py
│   │   │   ├── base_model_gxgy.py
│   │   │   ├── base_model_nonLinear.py
│   │   │   ├── model_x1_y1_ExpSaturant_classic.py
│   │   │   ├── model_x1_y1_augmented.py
│   │   │   ├── model_x1_y1_cubique_classic.py
│   │   │   ├── model_x1_y1_gordon_classic.py
│   │   │   ├── model_x1_y1_pairwise.py
│   │   │   ├── model_x1_y1_sinus_classic.py
│   │   │   ├── model_x2_y1_augmented.py
│   │   │   ├── model_x2_y1_classic.py
│   │   │   ├── model_x2_y1_pairwise.py
│   │   │   ├── model_x2_y1_rapport_classic.py
│   │   │   └── model_x2_y2_pairwise.py
│   │   ├── Generate_MatrixCov.py
│   │   └── __init__.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── csv_to_parquet.py
│   │   ├── exceptions.py
│   │   ├── numerics.py
│   │   ├── parser.py
│   │   ├── plot_settings.py
│   │   └── utils.py
│   ├── __init__.py
│   ├── run_linear_pkf.py
│   ├── run_nonlinear_epkf.py
│   ├── run_nonlinear_ppf.py
│   ├── run_nonlinear_ukf.py
│   ├── run_nonlinear_upkf.py
│   └── run_simulator.py
├── .gitignore
├── LICENSE
├── README.md
└── update_readme_structure.sh

13 directories, 82 files
```
<!-- PROJECT_STRUCTURE_END -->
