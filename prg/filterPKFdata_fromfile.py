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

if __name__ == "__main__":
    """
    USAGES:
        python3 prg/filterPKFdata_fromfile.py
        python3 prg/filterPKFdata_fromfile.py --verbose 0 --traceplot --linearModelName A_mQ_x1_y1 --dataFileName test.csv
    """
    
    # ------------------------------------------------------------------
    # Constants (default value) - Parser
    # ------------------------------------------------------------------

    parser = argparse.ArgumentParser(description='Filter linear data from file with PKF')
    addParseToParser(parser, ['linearModelName', 'dataFileName'])
    args   = parser.parse_args()

    traceplot       = args.traceplot
    verbose         = args.verbose
    linearModelName = args.linearModelName
    dataFileName    = args.dataFileName
    if dataFileName == None:
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
    if verbose > 0:
        print(f'model={model}')
        param.summary()

    # ------------------------------------------------------------------
    # Let's go
    # ------------------------------------------------------------------

    if verbose>0:
        print("\nPKF filtering with data generated from a file with data...")
    
    pkf_2    = Linear_PKF(param, save_pickle=traceplot, verbose=verbose)
    filename = os.path.join(datafile_dir, dataFileName)
    listePKF = pkf_2.process_N_data(N=None, data_generator=file_data_generator(filename, dim_x, dim_y, verbose))
    N        = listePKF[-1][0]+1

    if traceplot and pkf_2.history is not None:
        df = pkf_2.history.as_dataframe()
        if verbose > 0:
            print("\nExcerpt of the filtering with PKF :")
            print(df.head())
            
        # print scoring
        if listePKF[0][1] is not None:
            ListeA = ['xkp1']
            ListeB = ['Xkp1_update']
            ListeC = ['PXXkp1_update']
            ListeD = ['ikp1']
            ListeE = ['Skp1']
            pkf_2.history.compute_errors(ListeA, ListeB, ListeC, ListeD, ListeE)

        # pickle storing and plots
        pkf_2.history.save_pickle(os.path.join(tracker_dir, f"history_run_pfk_2.pkl"))
        if listePKF[0][1] is not None:
            title = f"'{linearModelName}' model data filtered with PKF"
            # window    = {'xmin': min(20, N), 'xmax': min(min(20, N)+100, N) }, \
            window    = {'xmin':0, 'xmax': 300 }
            pkf_2.history.plot(title, 
                            list_param= ["ykp1"], \
                            list_label= ["Observations y"], \
                            list_covar = [None], \
                            window    = window, \
                            basename  = f'pkf_2_{linearModelName}_observations', show=False, base_dir=graph_dir)
            pkf_2.history.plot(title, 
                            list_param= ["xkp1"  , "Xkp1_update"], \
                            list_label= ["x true", "x estimated"], \
                            list_covar = [None, "PXXkp1_update"], \
                            window    = window, \
                            basename  = f'pkf_2_{linearModelName}', show=False, base_dir=graph_dir)
