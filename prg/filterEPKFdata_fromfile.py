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
# Manage algorithms for non linear EPKF
from classes.NonLinear_EPKF import NonLinear_EPKF
# Manage algorithms for the PKF
from classes.Linear_PKF import Linear_PKF
# Manage non linear and linear parameters
from classes.ParamNonLinear import ParamNonLinear
from classes.ParamLinear import ParamLinear
# Parser d'options
from others.parser    import *

if __name__ == "__main__":
    """
    USAGES:
        python prg/filterEPKFdata_fromfile.py
        python prg/filterEPKFdata_fromfile.py --verbose 0 --traceplot --nonLinearModelName x1_y1_withRetroactions --dataFileName testNL.csv
    """
    
    # ------------------------------------------------------------------
    # Constants (default value) - Parser
    # ------------------------------------------------------------------

    parser = argparse.ArgumentParser(description='Filter non linear data from file with EPKF')
    addParseToParser(parser, ['nonLinearModelName', 'dataFileName'])
    args   = parser.parse_args()

    traceplot          = args.traceplot
    verbose            = args.verbose
    nonLinearModelName = args.nonLinearModelName
    dataFileName       = args.dataFileName
    if dataFileName == None:
        dataFileName = f"dataLinear_{nonLinearModelName}.csv"

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
    
    # model        = ModelFactoryLinear.create("A_mQ_x1_y1")
    # params       = model.get_params().copy()
    # dim_x, dim_y = params.pop('dim_x'), params.pop('dim_y')
    # param        = ParamLinear(verbose, dim_x, dim_y, **params)
    
    if verbose > 0:
        print(f'model={model}')
        param.summary()

    # ------------------------------------------------------------------
    # Let's go
    # ------------------------------------------------------------------

    if verbose > 0:
        print("\nEPKF filtering with data generated from a file with data... ")

    epkf_2    = NonLinear_EPKF(param, save_pickle=traceplot, verbose=verbose)
    filename  = os.path.join(datafile_dir, dataFileName)
    listeEPKF = epkf_2.process_N_data(N=None, data_generator=file_data_generator(filename, dim_x, dim_y, verbose))
    N = listeEPKF[-1][0] + 1

    if traceplot and epkf_2.history is not None:
        df = epkf_2.history.as_dataframe()
        if verbose > 0:
            print("\nExtract of the filtering with EPKF :")
            print(df.head())

        # print scoring
        if listeEPKF[0][1] is not None:
            # ListeA = ['xkp1',           'xkp1']
            # ListeB = ['Xkp1_predict',   'Xkp1_update']
            # ListeC = ['PXXkp1_predict', 'PXXkp1_update']
            ListeA = ['xkp1']
            ListeB = ['Xkp1_update']
            ListeC = ['PXXkp1_update']
            epkf_2.history.compute_errors(ListeA, ListeB, ListeC)

        # pickle storing and plots
        epkf_2.history.save_pickle(os.path.join(tracker_dir, f"history_run_epfk_2.pkl"))
        if listeEPKF[0][1] is not None:
            title = f"'{nonLinearModelName}' model data filtered with EPKF"
            epkf_2.history.plot(title, 
                            list_param= ["ykp1"], \
                            list_label= ["Observations y"], \
                            list_covar = [None], \
                            # window    = {'xmin': min(20, N), 'xmax': min(min(20, N)+100, N) }, \
                            window    = {'xmin': 2300, 'xmax': 2500 }, \
                            basename  = f'epkf_2_{nonLinearModelName}_observations', show=False, base_dir=graph_dir)
            epkf_2.history.plot(title, 
                            list_param= ["xkp1"  , "Xkp1_update"], \
                            list_label= ["x true", "x estimated"], \
                            list_covar = [None, "PXXkp1_update"], \
                            # window    = {'xmin': min(20, N), 'xmax': min(min(20, N)+100, N) }, \
                            window    = {'xmin': 2300, 'xmax': 2500 }, \
                            basename  = f'epkf_2_{nonLinearModelName}', show=False, base_dir=graph_dir)

