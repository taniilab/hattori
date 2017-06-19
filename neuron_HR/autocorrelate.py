# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 14:59:29 2017

@author: Hattori
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal as sig

dataset = pd.read_csv('./results/2017_6_19_12_38_7_5HR_model.csv',
                      index_col=0)
p = dataset.as_matrix()
print(p[:, 2])
cor = sig.correlate(p[:, 2], p[:, 2], mode="full")
print(cor)

print(len(cor))
print(len(p[:, 2]))

fig, ax = plt.subplots(nrows=1, figsize=(12, 12))
fig.tight_layout()
fig.subplots_adjust(left=0.05, bottom=0.03)
ax.set_ylim(-100, 50)
# ax.plot(p[:, 4], cor)
