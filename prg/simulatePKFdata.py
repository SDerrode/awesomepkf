#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
from pathlib import Path

# A few utils functions that are used several times
from others.Utils import save_dataframe_to_csv
# PKF class
from PKF import PKF
# Parameters for PKF
from classes.ParamPKF import ParamPKF
# non linear models 
from models.linear import BaseModel, all_models


def data_to_dataframe(listData, dim_x, dim_y):
    """Convertit une liste de tuples PKF en DataFrame pandas."""
    
    data = []
    for idx, (x, y) in [(i, vals) for i, vals in listData]:
        # Validation des types
        if not hasattr(x, "flatten") or not hasattr(y, "flatten"):
            raise TypeError(f"Les éléments pour l'index {idx} ne sont pas des numpy.array valides.")
        x_values = x.flatten()
        y_values = y.flatten()
        if len(x_values) != 3 or len(y_values) != 1:
            raise ValueError(f"Taille inattendue des vecteurs à l'index {idx}: X={len(x_values)}, Y={len(y_values)}")
        data.append([*x_values, *y_values])

    # Création du DataFrame
    columns = []
    for c in range(dim_x):
        columns.append(f"X{c}")
    for c in range(dim_y):
        columns.append(f"Y{c}")
    df = pd.DataFrame(data, columns=columns)
    # df.set_index("index", inplace=True)

    return df


if __name__ == "__main__":
    """
        python prg/simulatePKFdata.py
    """
    # ------------------------------------------------------------------
    # Constants
    # ------------------------------------------------------------------
    save_pickle = False
    verbose     = 0
    N           = 10 # > 20
    
    # ------------------------------------------------------------------
    # Output repo for data
    # ------------------------------------------------------------------
    base_dir     = os.path.join(".",      "data")
    datafile_dir = os.path.join(base_dir, "datafile")

    # ------------------------------------------------------------------
    # Test parameters for the Two ((A, mQ) or Sigma) parametrizations
    # ------------------------------------------------------------------
    
    # Available : ['A_mQ_x1_y1', 'A_mQ_x3_y1', 'Sigma_x1_y1', 'Sigma_x3_y1', 'A_mQ_x2_y2', 'Sigma_x2_y2', 'A_mQ_x1_y1_VPgreaterThan1']
    model_module = all_models['Sigma_x3_y1']
    model = model_module.create_model()
    print(f'model={model_module.MODEL_NAME}')
    # print(f'model={model.get_params()}')
    
    params = model.get_params().copy()
    dim_x, dim_y = params.pop('dim_x'), params.pop('dim_y')
    param = ParamPKF(verbose, dim_x, dim_y, **params)
    if verbose > 0:
        param.summary()

    # ------------------------------------------------------------------
    # Let's go
    # ------------------------------------------------------------------

    print("\nPKF simulation")
    sKey  = 41
    pkf_1 = PKF(param, sKey=sKey, save_pickle=save_pickle, verbose=verbose)
    
    # Simulate data with the simulator generator
    listData = pkf_1.simulate_N_data(N=N)
    # print(f'listData={listData}')
    
    # Save data as a dataframe using pandas
    df       = data_to_dataframe(listData, dim_x, dim_y)
    filepath = os.path.join(datafile_dir, f"dataPKF_{model_module.MODEL_NAME}_dim{dim_x}x{dim_y}.csv")
    save_dataframe_to_csv(df, filepath)
    