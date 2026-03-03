
# AwesomePKF

This repository contains a set of programs illustrating the **Pairwise Kalman Filter (PKF)**, a generalization of the classical Kalman Filter, extended to non-linear models. It includes several variants of non-linear filters:

- **Extended Pairwise Kalman Filter (EPKF)**
- **Unscented Pairwise Kalman Filter (UPKF)**, with multiple variants depending on the choice of sigma points  
- **Pairwise Particle Filter (PPF)**  

---

## Table of Contents

- [AwesomePKF](#awesomepkf)
    - [Table of Contents](#table-of-contents)
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

## Models and Simulations

The repository provides a program called **run_simulator.py** to simulate data according to **linear and non-linear models**.

---

## Filters

Each filter has two types of programs:

1. Simulate data **and filter it directly**  
2. Filter data **from a previously saved file**  

### Pairwise Kalman Filter (PKF)

- **run_linear_pkf.py** – filter linear data either from simulated data or from a previously saved file e.g., generated with `run_simulator.py`)  

### Extended Pairwise Kalman Filter (EPKF)

- **run_nonlinear_epkf.py** – filter non-linear data either from simulated data or from a previously saved file e.g., generated with `run_simulator.py`)  

### Unscented Pairwise Kalman Filter (UPKF)

- **run_nonlinear_upkf.py** – filter non-linear data either from simulated data or from a previously saved file e.g., generated with `run_simulator.py`)  

### Pairwise Particle Filter (PPF)

- **run_nonlinear_ppf.py** – filter non-linear data either from simulated data or from a previously saved file e.g., generated with `run_simulator.py`)  

---

## Usage Examples

### Simulate Linear Data and Filter with PKF

```bash
python3 prg/run_simulator.py --N 2000 --linearModelName "A_mQ_x1_y1" --dataFileName "testL.csv" --verbose 1 --sKey 303
python3 prg/run_linear_pkf.py --linearModelName "A_mQ_x1_y1" --dataFileName "testL.csv" --verbose 1 --saveHistory --plot
```

### Simulate Non-Linear Data and Filter with EPKF, UPKF and PPF

```bash
python3 prg/run_simulator.py   --N 1000 --nonLinearModelName "x2_y1_withRetroactionsOfObservations" --dataFileName "testNL.csv" --verbose 1 --sKey 303 

python3 prg/run_nonlinear_epkf.py --nonLinearModelName "x2_y1_withRetroactionsOfObservations" --dataFileName "testNL.csv" --ell 1              --verbose 1 --saveHistory --plot
python3 prg/run_nonlinear_upkf.py --nonLinearModelName "x2_y1_withRetroactionsOfObservations" --dataFileName "testNL.csv" --sigmaSet "wan2000" --verbose 1 --saveHistory --plot
python3 prg/run_nonlinear_ppf.py   --nonLinearModelName "x2_y1_withRetroactionsOfObservations" --dataFileName "testNL.csv" --nbParticles 300    --verbose 1 --saveHistory --plot
```

---

## Folders structure

<!-- commande : 
    tree -L 4 -I "logs|venv|*.csv|*.pkl|*.png|__pycache__|*.code-workspace|*.ipynb" > folder_strucure.md
-->

<!-- meilleure commande : 
git ls-files | tree --fromfile -F -a --dirsfirst -I "logs|venv|*.csv|*.pkl|*.png|__pycache__|*.code-workspace|*.ipynb|.vscode|.gitkeep|.DS_Store" > structure.txt
-->

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
│   │   ├── NonLinear_UPKF.py
│   │   ├── PKF.py
│   │   ├── ParamLinear.py
│   │   ├── ParamNonLinear.py
│   │   ├── SeedGenerator.py
│   │   ├── SigmaPointsSet.py
│   │   └── __init__.py
│   ├── models/
│   │   ├── linear/
│   │   │   ├── A_mQ_x1_y1.py
│   │   │   ├── A_mQ_x1_y1_VPgreaterThan1.py
│   │   │   ├── A_mQ_x1_y1_augmented.py
│   │   │   ├── A_mQ_x2_y2.py
│   │   │   ├── A_mQ_x3_y1.py
│   │   │   ├── Sigma_x1_y1.py
│   │   │   ├── Sigma_x2_y2.py
│   │   │   ├── Sigma_x3_y1.py
│   │   │   ├── __init__.py
│   │   │   ├── base_model_linear.py
│   │   │   └── generMatrixA_fromVP.py
│   │   ├── nonLinear/
│   │   │   ├── __init__.py
│   │   │   ├── base_model_nonLinear.py
│   │   │   ├── model_cubique.py
│   │   │   ├── model_ext_saturant.py
│   │   │   ├── model_gordon.py
│   │   │   ├── model_sinus.py
│   │   │   ├── model_x1_y1_withRetroactions.py
│   │   │   ├── model_x1_y1_withRetroactions_augmented.py
│   │   │   ├── model_x2_y1.py
│   │   │   ├── model_x2_y1_rapport.py
│   │   │   ├── model_x2_y1_withRetroactionsOfObservations.py
│   │   │   ├── model_x2_y1_withRetroactionsOfObservations_augmented.py
│   │   │   └── model_x2_y2_withRetroactions.py
│   │   ├── Generate_MatrixCov.py
│   │   ├── __init__.py
│   │   ├── testLinear.py
│   │   └── testNonLinear.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── csv_to_parquet.py
│   │   ├── numerics.py
│   │   ├── parser.py
│   │   ├── plot_settings.py
│   │   └── utils.py
│   ├── __init__.py
│   ├── run_linear_pkf.py
│   ├── run_nonlinear_epkf.py
│   ├── run_nonlinear_ppf.py
│   ├── run_nonlinear_upkf.py
│   └── run_simulator.py
├── .gitignore
├── LICENSE
└── README.md

13 directories, 74 files
```
<!-- PROJECT_STRUCTURE_END -->
