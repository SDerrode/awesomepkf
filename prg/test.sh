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

# python prg/filterPKFdata.py
# python prg/filterEPKFdata.py
# python prg/filterUPKFdata.py

python prg/simulateLinearData.py
python prg/filterPKFdata_fromfile.py

python prg/simulateNonLinearData.py
python prg/filterEPKFdata_fromfile.py
python prg/filterUPKFdata_fromfile.py

