#!/usr/bin/env bash
# -*- coding: utf-8 -*-

###################
# USAGE
# ./prg/test.sh
# Pour tester que les principaux programmes fonctionnent bien avant un commit


# Nettoyage des répertoires des fichiers générés
find ./data/datafile       -type f ! -name ".gitkeep" -delete
find ./data/historyTracker -type f ! -name ".gitkeep" -delete
find ./data/plot           -type f ! -name ".gitkeep" -delete

python prg/simulateLinearData.py      --N 1000 --linearModelName    A_mQ_x1_y1             --dataFileName test.csv   --verbose 0             --sKey 303
python prg/filterPKFdata_fromfile.py           --linearModelName    A_mQ_x1_y1             --dataFileName test.csv   --verbose 0 --traceplot

python prg/simulateNonLinearData.py   --N 1000 --nonLinearModelName x1_y1_withRetroactions --dataFileName testNL.csv --verbose 0             --sKey 303
python prg/filterEPKFdata_fromfile.py          --nonLinearModelName x1_y1_withRetroactions --dataFileName testNL.csv --verbose 0 --traceplot
python prg/filterUPKFdata_fromfile.py          --nonLinearModelName x1_y1_withRetroactions --dataFileName testNL.csv --verbose 0 --traceplot

python prg/filterPKFdata.py           --N 1000 --linearModelName    A_mQ_x1_y1                                       --verbose 0 --traceplot --sKey 303
python prg/filterEPKFdata.py          --N 1000 --nonLinearModelName x1_y1_withRetroactions                           --verbose 0 --traceplot --sKey 303
python prg/filterUPKFdata.py          --N 1000 --nonLinearModelName x1_y1_withRetroactions                           --verbose 0 --traceplot --sKey 303
