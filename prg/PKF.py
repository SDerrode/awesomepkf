#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, path
directory = path.Path(__file__)
sys.path.append(directory.parent.parent)

import math

import numpy  as np
import scipy  as sc
import pandas as pd

# from prg.settings.parser     import *


class PKF:
    
    def __init__(self, ):
        pass
    
        self.generateNewSeed() # cle pour la génération de nombres aléatoires
    
    def getSeed(self) -> int:
        return self.__Seed
    def setSeed(self, sKey) -> None:
        self.__Seed = sKey
        self.__rng  = np.random.default_rng(self.__Seed)
    def generateNewSeed(self) -> int:
        self.setSeed(secrets.randbits(128))
        return self.__Seed


if __name__ == '__main__':
    
    """
    python prg/PKF.py
    """
    
    