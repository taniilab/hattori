# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 08:57:37 2017
csvファイルを読み込んでプロット
@author: 6969p
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
"""
dataset= pd.read_csv('./2017_2_19_18_26_50_1oua1oud1hh_model.csv', index_col=0)
p = dataset.as_matrix()
print(p)

fig, ax = plt.subplots(nrows = 1, figsize=(20, 20))  
fig.tight_layout()
fig.subplots_adjust(left=0.05, bottom=0.03)
ax.set_ylim(-100, 50)
ax.plot(p[:, 1], p[:, 0]) 
"""
data = np.genfromtxt('./1.csv', delimiter=',', filling_values=0)
print(data)

print(data[2][3])