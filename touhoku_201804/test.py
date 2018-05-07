# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 18:21:18 2018

@author: 6969p
"""
import pandas as pd
import numpy as np
import sys
import time

path = "Z:/Box Sync/Personal/Documents/20180420_touhoku_patch/csvtest.csv"

df = pd.read_csv(path, delimiter='\t', skiprows=[0, 1])
# df = pd.read_csv(path, delimiter=',', skiprows=[0, 1])
print(df)
df = df.replace(0, np.nan)
print(df)
