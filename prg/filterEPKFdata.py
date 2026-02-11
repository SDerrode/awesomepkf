#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np

# non linear models 
from models.nonLinear import ModelFactoryNonLinear
# A few utils functions that are used several times
from others.utils import compute_errors
# Manage algorithms for the EPKF
from classes.NonLinear_EPKF import NonLinear_EPKF
# Manage non linear parameters
from classes.ParamNonLinear import ParamNonLinear
# Parser d'options
from others.parser    import *
# Parser d'options
from others.plot_settings import WINDOW

if __name__ == "__main__":
    """
    USAGES:
        python3 prg/filterEPKFdata.py
        python3 prg/filterEPKFdata.py --N 1000 --nonLinearModelName "x1_y1_withRetroactions" --sKey 303 --verbose 0 --plot --saveHistory
    """
    
    # ------------------------------------------------------------------
    # Constants (default value) - Parser
    # ------------------------------------------------------------------

    parser = argparse.ArgumentParser(description='Simulate and filter non linear data with EPKF')
    addParseToParser(parser, ['nonLinearModelName', 'N', 'sKey'])
    args   = parser.parse_args()
    
    plot               = args.plot
    saveHistory        = args.saveHistory
    verbose            = args.verbose
    N                  = args.N
    sKey               = args.sKey 
    nonLinearModelName = args.nonLinearModelName
    if sKey is not None and sKey < 0:
        parser.error("sKey must be >= 0")
    if N < 200:
        parser.error("N must be >= 200")
    # exit(1)

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
    # Test parameters
    # ------------------------------------------------------------------

    model        = ModelFactoryNonLinear.create(nonLinearModelName)
    params       = model.get_params()
    dim_x, dim_y = params.pop('dim_x'), params.pop('dim_y')
    param        = ParamNonLinear(verbose, dim_x, dim_y, **params)
    if verbose > 1:
        print(f'model={model}')
        param.summary()

    # ------------------------------------------------------------------
    # Let's go
    # ------------------------------------------------------------------

    if verbose > 1:
        print("\nEPKF filtering with data generated from a non-linear model...")
        
    epkf_1    = NonLinear_EPKF(param, sKey=sKey, verbose=verbose)
    listeEPKF = epkf_1.process_N_data(N=N)  # Call with the default data simulator generator


    if verbose > 1:
        print("\nExcerpt of the filtering with EPKF :")
        print(epkf_1.history.as_dataframe().head())

    # print scoring
    ListeA = ['xkp1']
    ListeB = ['Xkp1_update']
    ListeC = ['PXXkp1_update']
    ListeD = ['ikp1']
    ListeE = ['Skp1']
    epkf_1.history.compute_errors(epkf_1, ListeA, ListeB, ListeC, ListeD, ListeE)
    
    if saveHistory:
        epkf_1.history.save_pickle(os.path.join(tracker_dir, f"history_run_epfk_1.pkl"))

    if plot:
        title = f"'{nonLinearModelName}' model data filtered with EPKF"
        epkf_1.history.plot(title, 
                            list_param = ["ykp1"], \
                            list_label = ["Observations y"], \
                            list_covar = [None], \
                            window     = WINDOW, \
                            basename   = f'epkf_1_{nonLinearModelName}_observations', show=False, base_dir=graph_dir)
        epkf_1.history.plot(title, 
                            list_param = ["xkp1"  , "Xkp1_update"], \
                            list_label = ["x true", "x estimated"], \
                            list_covar = [None,     "PXXkp1_update"], \
                            window     = WINDOW, \
                            basename   = f'epkf_1_{nonLinearModelName}', show=False, base_dir=graph_dir)