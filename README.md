# AwesomePKF

This repository contains a set of programs illustrating the **Pairwise Kalman Filter (PKF)**, a generalization of the classical Kalman Filter, extended here to non-linear models. It includes several variants of non-linear filters:

- **Extended Pairwise Kalman Filter (EPKF)**  
- **Unscented Pairwise Kalman Filter (UPKF)**, with multiple variants depending on the choice of sigma points  
- **Particle Filter (PF)**, not new but interesting for comparison purposes.

---

## Table of Contents

- [AwesomePKF](#awesomepkf)
    - [Table of Contents](#table-of-contents)
- [AwesomePKF](#awesomepkf-1)
    - [Table of Contents](#table-of-contents-1)
    - [Models and Simulations](#models-and-simulations)
    - [Filters](#filters)
        - [Pairwise Kalman Filter (PKF)](#pairwise-kalman-filter-pkf)
        - [Extended Pairwise Kalman Filter (EPKF)](#extended-pairwise-kalman-filter-epkf)
        - [Unscented Pairwise Kalman Filter (UPKF)](#unscented-pairwise-kalman-filter-upkf)
        - [Particle Filter (PF)](#particle-filter-pf)
    - [Usage Examples](#usage-examples)
        - [Simulate Linear Data and Filter with PKF](#simulate-linear-data-and-filter-with-pkf)
        - [Simulate Non-Linear Data and Filter with EPKF, UPKF and PF](#simulate-non-linear-data-and-filter-with-epkf-upkf-and-pf)
    - [Folders structure](#folders-structure)

---

# AwesomePKF

This repository contains a set of programs illustrating the **Pairwise Kalman Filter (PKF)**, a generalization of the classical Kalman Filter, extended to non-linear models. It includes several variants of non-linear filters:

- **Extended Pairwise Kalman Filter (EPKF)**  
- **Unscented Pairwise Kalman Filter (UPKF)**, with multiple variants depending on the choice of sigma points  
- **Particle Filter (PF)**  

---

## Table of Contents

- [AwesomePKF](#awesomepkf)
    - [Table of Contents](#table-of-contents)
- [AwesomePKF](#awesomepkf-1)
    - [Table of Contents](#table-of-contents-1)
    - [Models and Simulations](#models-and-simulations)
    - [Filters](#filters)
        - [Pairwise Kalman Filter (PKF)](#pairwise-kalman-filter-pkf)
        - [Extended Pairwise Kalman Filter (EPKF)](#extended-pairwise-kalman-filter-epkf)
        - [Unscented Pairwise Kalman Filter (UPKF)](#unscented-pairwise-kalman-filter-upkf)
        - [Particle Filter (PF)](#particle-filter-pf)
    - [Usage Examples](#usage-examples)
        - [Simulate Linear Data and Filter with PKF](#simulate-linear-data-and-filter-with-pkf)
        - [Simulate Non-Linear Data and Filter with EPKF, UPKF and PF](#simulate-non-linear-data-and-filter-with-epkf-upkf-and-pf)
    - [Folders structure](#folders-structure)

---

## Models and Simulations

The repository provides several **linear and non-linear models** that can be used with the programs in the `_prg_` folder:

- **simulateLinearData.py** – simulate and save data according to one of the proposed **linear models**  
- **simulateNonLinearData.py** – simulate and save data according to one of the proposed **non-linear models**  

---

## Filters

Each filter has two types of programs:

1. Simulate data **and filter it directly**  
2. Filter data **from a previously saved file**  

### Pairwise Kalman Filter (PKF)

- **filterPKFdata.py** – simulate linear data and filter it with PKF  
- **filterPKFdata_fromfile.py** – filter linear data from a previously saved file (e.g., generated with `simulateLinearData.py`)  

### Extended Pairwise Kalman Filter (EPKF)

- **filterEPKFdata.py** – simulate non-linear data and filter it with EPKF  
- **filterEPKFdata_fromfile.py** – filter non-linear data from a previously saved file (e.g., generated with `simulateNonLinearData.py`)  

### Unscented Pairwise Kalman Filter (UPKF)

- **filterUPKFdata.py** – simulate non-linear data and filter it with UPKF  
- **filterUPKFdata_fromfile.py** – filter non-linear data from a previously saved file  

### Particle Filter (PF)

- **filterPFdata.py** – simulate non-linear data and filter it with PF  
- **filterPFdata_fromfile.py** – filter non-linear data from a previously saved file  

---

## Usage Examples

### Simulate Linear Data and Filter with PKF

```bash
python3 prg/simulateLinearData.py     --verbose 1 --linearModelName "A_mQ_x1_y1" --dataFileName "test.csv" --N 1000 --sKey 303
python3 prg/filterPKFdata_fromfile.py --verbose 1 --linearModelName "A_mQ_x1_y1" --dataFileName "test.csv" --traceplot 
```

### Simulate Non-Linear Data and Filter with EPKF, UPKF and PF

```bash
python3 prg/simulateNonLinearData.py   --verbose 1 --nonLinearModelName "x1_y1_withRetroactions" --dataFileName "testNL.csv" --sKey 303 --N 1000
python3 prg/filterEPKFdata_fromfile.py --verbose 1 --nonLinearModelName "x1_y1_withRetroactions" --dataFileName "testNL.csv" --traceplot
python3 prg/filterUPKFdata_fromfile.py --verbose 1 --nonLinearModelName "x1_y1_withRetroactions" --dataFileName "testNL.csv" --traceplot --sigmaSet "wan2000"
python3 prg/filterPFdata_fromfile.py   --verbose 1 --nonLinearModelName "x1_y1_withRetroactions" --dataFileName "testNL.csv" --traceplot --nbParticles 300
```

---

## Folders structure

<!-- commande : 
    tree -L 4 -I "logs|venv|*.csv|*.pkl|*.png|__pycache__|*.code-workspace|*.ipynb" > folder_strucure.md
-->

.
├── LICENSE
├── data
│   ├── datafile
│   ├── historyTracker
│   └── plot
├── folder_strucure.md
├── ipynb
│   └── readme.md
├── prg
│   ├── classes
│   │   ├── ActiveView.py
│   │   ├── HistoryTracker.py
│   │   ├── Linear_PKF.py
│   │   ├── NonLinear_EPKF.py
│   │   ├── NonLinear_PF.py
│   │   ├── NonLinear_PKF.py
│   │   ├── NonLinear_UPKF.py
│   │   ├── ParamLinear.py
│   │   ├── ParamNonLinear.py
│   │   ├── SeedGenerator.py
│   │   └── SigmaPointsSet.py
│   ├── filterEPKFdata.py
│   ├── filterEPKFdata_fromfile.py
│   ├── filterPFdata.py
│   ├── filterPFdata_fromfile.py
│   ├── filterPKFdata.py
│   ├── filterPKFdata_fromfile.py
│   ├── filterUPKFdata.py
│   ├── filterUPKFdata_fromfile.py
│   ├── models
│   │   ├── linear
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
│   │   ├── nonLinear
│   │   │   ├── __init__.py
│   │   │   ├── base_model_nonLinear.py
│   │   │   ├── model_cubique.py
│   │   │   ├── model_ext_saturant.py
│   │   │   ├── model_gordon.py
│   │   │   ├── model_sinus.py
│   │   │   ├── model_x1_y1_withRetroaction.py
│   │   │   ├── model_x1_y1_withRetroaction_augmented.py
│   │   │   ├── model_x2_y1.py
│   │   │   ├── model_x2_y1_rapport.py
│   │   │   ├── model_x2_y1_withRetroactionsOfObservations.py
│   │   │   └── model_x2_y1_withRetroactionsOfObservations_augmented.py
│   │   ├── testLinear.py
│   │   └── testNonLinear.py
│   ├── others
│   │   ├── csv_to_parquet.py
│   │   ├── parser.py
│   │   ├── plot_settings.py
│   │   └── utils.py
│   ├── simulateLinearData.py
│   ├── simulateNonLinearData.py
│   └── tests
│       ├── Jacobien_TextPourChatGPT.txt
│       ├── bash_augmentation_L.sh
│       ├── bash_augmentation_NL.sh
│       ├── commandes_L.sh
│       ├── commandes_NL.sh
│       └── run_tests.sh
└── readme.md

