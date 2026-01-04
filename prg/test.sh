#!/usr/bin/env bash
# -*- coding: utf-8 -*-

###################
# USAGE
# ./prg/test.sh

# =========================
# Nettoyage des répertoires des fichiers générés
# =========================
find ./data/datafile       -type f ! -name ".gitkeep" -delete
find ./data/historyTracker -type f ! -name ".gitkeep" -delete
find ./data/plot           -type f ! -name ".gitkeep" -delete

# =========================
# Définition des variables
# =========================
N_LINEAR=10000
N_NONLINEAR=10000
LINEAR_MODEL="A_mQ_x1_y1"
NONLINEAR_MODEL="x1_y1_withRetroactions"
DATA_LINEAR="test.csv"
DATA_NONLINEAR="testNL.csv"
VERBOSE=0
SEED=303
TRACE_FLAG="--traceplot"

# =========================
# Simulation et filtrage des données linéaires
# =========================
python prg/simulateLinearData.py \
    --N "$N_LINEAR" \
    --linearModelName "$LINEAR_MODEL" \
    --dataFileName "$DATA_LINEAR" \
    --verbose "$VERBOSE" \
    --sKey "$SEED"

python prg/filterPKFdata_fromfile.py \
    --linearModelName "$LINEAR_MODEL" \
    --dataFileName "$DATA_LINEAR" \
    --verbose "$VERBOSE" \
    $TRACE_FLAG

python prg/filterPKFdata.py \
    --N "$N_LINEAR" \
    --linearModelName "$LINEAR_MODEL" \
    --verbose "$VERBOSE" \
    $TRACE_FLAG \
    --sKey "$SEED"

# =========================
# Simulation et filtrage des données non linéaires
# =========================
python prg/simulateNonLinearData.py \
    --N "$N_NONLINEAR" \
    --nonLinearModelName "$NONLINEAR_MODEL" \
    --dataFileName "$DATA_NONLINEAR" \
    --verbose "$VERBOSE" \
    --sKey "$SEED"

python prg/filterEPKFdata_fromfile.py \
    --nonLinearModelName "$NONLINEAR_MODEL" \
    --dataFileName "$DATA_NONLINEAR" \
    --verbose "$VERBOSE" \
    $TRACE_FLAG

python prg/filterUPKFdata_fromfile.py \
    --nonLinearModelName "$NONLINEAR_MODEL" \
    --dataFileName "$DATA_NONLINEAR" \
    --verbose "$VERBOSE" \
    $TRACE_FLAG

python prg/filterEPKFdata.py \
    --N "$N_NONLINEAR" \
    --nonLinearModelName "$NONLINEAR_MODEL" \
    --verbose "$VERBOSE" \
    $TRACE_FLAG \
    --sKey "$SEED"

python prg/filterUPKFdata.py \
    --N "$N_NONLINEAR" \
    --nonLinearModelName "$NONLINEAR_MODEL" \
    --verbose "$VERBOSE" \
    $TRACE_FLAG \
    --sKey "$SEED"
