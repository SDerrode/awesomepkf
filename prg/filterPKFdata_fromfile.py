#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np

# Linear models
from models.linear import ModelFactoryLinear
# A few utils functions that are used several times
from others.utils import compute_errors, file_data_generator
# Manage algorithms for the PKF
from classes.Linear_PKF import Linear_PKF
# Manage Linear parameters
from classes.ParamLinear import ParamLinear
# Parser d'options
from others.parser    import *
# Parser d'options
from others.plot_settings import WINDOW

if __name__ == "__main__":
    """
    USAGES:
        python3 prg/filterPKFdata_fromfile.py
        python3 prg/filterPKFdata_fromfile.py --linearModelName "A_mQ_x1_y1" --dataFileName "test.csv" --verbose 0 --plot --saveHistory
    """
    
    # ------------------------------------------------------------------
    # Constants (default value) - Parser
    # ------------------------------------------------------------------

    parser = argparse.ArgumentParser(description='Filter linear data from file with PKF')
    addParseToParser(parser, ['linearModelName', 'dataFileName'])
    args   = parser.parse_args()

    plot               = args.plot
    saveHistory        = args.saveHistory
    verbose         = args.verbose
    linearModelName = args.linearModelName
    dataFileName    = args.dataFileName
    if dataFileName is None:
        dataFileName = f"dataLinear_{linearModelName}.csv"

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
    # Test parameters for the Two ((A, mQ) or Sigma) parametrizations
    # ------------------------------------------------------------------

    model        = ModelFactoryLinear.create(linearModelName)
    params       = model.get_params().copy()
    dim_x, dim_y = params.pop('dim_x'), params.pop('dim_y')
    param        = ParamLinear(verbose, dim_x, dim_y, **params)
    if verbose > 1:
        print(f'model={model}')
        param.summary()

    # ------------------------------------------------------------------
    # Let's go
    # ------------------------------------------------------------------

    if verbose > 1:
        print("\nPKF filtering with data generated from a file...")
    
    pkf_2    = Linear_PKF(param, verbose=verbose)
    filename = os.path.join(datafile_dir, dataFileName)
    listePKF = pkf_2.process_N_data(N=None, data_generator=file_data_generator(filename, dim_x, dim_y, verbose))

    if verbose > 1:
        print("\nExcerpt of the filtering with PKF :")
        print(pkf_2.history.as_dataframe().head())
            
    # print scoring
    if listePKF[0][1] is not None:
        ListeA = ['xkp1']
        ListeB = ['Xkp1_update']
        ListeC = ['PXXkp1_update']
        ListeD = ['ikp1']
        ListeE = ['Skp1']
        pkf_2.history.compute_errors(pkf_2, ListeA, ListeB, ListeC, ListeD, ListeE)

    if saveHistory:
        pkf_2.history.save_pickle(os.path.join(tracker_dir, f"history_run_pfk_2.pkl"))
        
    if listePKF[0][1] is not None and plot:
        title = f"'{linearModelName}' model data filtered with PKF"
        pkf_2.history.plot(title, 
                        list_param= ["ykp1"], \
                        list_label= ["Observations y"], \
                        list_covar = [None], \
                        window    = WINDOW, \
                        basename  = f'pkf_2_{linearModelName}_observations', show=False, base_dir=graph_dir)
        pkf_2.history.plot(title, 
                        list_param= ["xkp1"  , "Xkp1_update"], \
                        list_label= ["x true", "x estimated"], \
                        list_covar = [None,    "PXXkp1_update"], \
                        window    = WINDOW, \
                        basename  = f'pkf_2_{linearModelName}', show=False, base_dir=graph_dir)
