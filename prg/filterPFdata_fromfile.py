#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np

import os
import numpy as np

# non linear models 
from models.nonLinear import ModelFactoryNonLinear
# Linear models
from models.linear import ModelFactoryLinear
# A few utils functions that are used several times
from others.utils import compute_errors, file_data_generator
# Manage algorithms for non linear PF
from classes.NonLinear_PF import NonLinear_PF
# Manage algorithms for the PKF
from classes.Linear_PKF import Linear_PKF
# Manage non linear and linear parameters
from classes.ParamNonLinear import ParamNonLinear
from classes.ParamLinear import ParamLinear
# Parser d'options
from others.parser    import *
# Parser d'options
from others.plot_settings import WINDOW

if __name__ == "__main__":
    """
    USAGES:
        python3 prg/filterPFdata_fromfile.py
        python3 prg/filterPFdata_fromfile.py --nonLinearModelName "x1_y1_withRetroactions" --dataFileName "testNL.csv" --nbParticles 300  --verbose 0 --plot --saveHistory
    """
    
    # ------------------------------------------------------------------
    # Constants (default value) - Parser
    # ------------------------------------------------------------------

    parser = argparse.ArgumentParser(description='Filter non linear data from file with PF')
    addParseToParser(parser, ['nonLinearModelName', 'dataFileName', 'nbParticles'])
    args   = parser.parse_args()

    plot               = args.plot
    saveHistory        = args.saveHistory
    verbose            = args.verbose
    resample_threshold = 0.5
    nbParticles        = args.nbParticles
    nonLinearModelName = args.nonLinearModelName
    dataFileName       = args.dataFileName
    if dataFileName is None:
        dataFileName = f"dataNonLinear_{nonLinearModelName}.csv"

    # ------------------------------------------------------------------
    # Output repo for data, traces and plots
    # ------------------------------------------------------------------
    base_dir     = os.path.join(".",      "data")
    tracker_dir  = os.path.join(base_dir, "historyTracker")
    datafile_dir = os.path.join(base_dir, "datafile")
    graph_dir    = os.path.join(base_dir, "plot")
    os.makedirs(tracker_dir,  exist_ok=True)
    os.makedirs(datafile_dir, exist_ok=True)
    os.makedirs(graph_dir,    exist_ok=True)

    # ------------------------------------------------------------------
    # Test parameters Chose a linear or a non linear model by commenting
    # ------------------------------------------------------------------

    model        = ModelFactoryNonLinear.create(nonLinearModelName)
    params       = model.get_params()
    dim_x, dim_y = params.pop('dim_x'), params.pop('dim_y')
    param        = ParamNonLinear(verbose, dim_x, dim_y, **params)
    # In test to integrate a linear model as a non linear model
    # model        = ModelFactoryLinear.create("A_mQ_x1_y1")
    # params       = model.get_params().copy()
    # dim_x, dim_y = params.pop('dim_x'), params.pop('dim_y')
    # param        = ParamLinear(verbose, dim_x, dim_y, **params)
    if verbose > 1:
        print(f'model={model}')
        param.summary()

    # ------------------------------------------------------------------
    # Let's go
    # ------------------------------------------------------------------

    if verbose > 1:
        print(f"\nPF filtering (nbParticles={nbParticles}, resample_threshold={resample_threshold}) with data generated from a file... ")

    pf_2     = NonLinear_PF(param, nbParticles=nbParticles, resample_threshold=resample_threshold, verbose=verbose)
    filename = os.path.join(datafile_dir, dataFileName)
    listePF  = pf_2.process_N_data(N=None, data_generator=file_data_generator(filename, dim_x, dim_y, verbose))

    if verbose > 1:
        print("\nExcerpt of the filtering with PF :")
        print(pf_2.history.as_dataframe().head())

    # print scoring
    if listePF[0][1] is not None:
        ListeA = ['xkp1']
        ListeB = ['Xkp1_update']
        ListeC = ['PXXkp1_update']
        ListeD = None
        ListeE = None
        pf_2.history.compute_errors(pf_2, ListeA, ListeB, ListeC, ListeD, ListeE)

    if saveHistory:
        pf_2.history.save_pickle(os.path.join(tracker_dir, f"history_run_pf_2.pkl"))
    
    if listePF[0][1] is not None and plot:
        title = f"'{nonLinearModelName}' model data filtered with PF"
        pf_2.history.plot(title, 
                        list_param= ["ykp1"], \
                        list_label= ["Observations y"], \
                        list_covar= [None], \
                        window    = WINDOW, \
                        basename  = f'pf_2_{nonLinearModelName}_observations', show=False, base_dir=graph_dir)
        pf_2.history.plot(title, 
                        list_param= ["xkp1"  , "Xkp1_update"], \
                        list_label= ["x true", "x estimated"], \
                        list_covar= [None,     "PXXkp1_update"], \
                        window    = WINDOW, \
                        basename  = f'pf_2_{nonLinearModelName}', show=False, base_dir=graph_dir)

