#%matplotlib notebook
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 2 cells burst 13.csv
#single cell burst 14.csv
path_v = """C:/Users/Hattori/Documents/2017_11_12_touhoku/files_voltage/0.csv"""
path_i = """C:/Users/Hattori/Documents/2017_11_12_touhoku/files_current/0_current.csv"""

fig = plt.figure(figsize=(10, 7))
dfv = pd.read_csv(path_v, delimiter=',')
dfi = pd.read_csv(path_i, delimiter=',')
"""
axv = plt.subplot(3, 1, 1)
axv.plot(dfv['index'], dfv['V[mV]'], markevery=[0, -1])
axv.grid(which='major', color='thistle', linestyle='-')

axi = plt.subplot(3, 1, 2)
axi.plot(dfi['index'], dfi['V[mV]'], markevery=[0, -1])
axi.grid(which='major', color='thistle', linestyle='-')
"""
ax1 = plt.subplot(1, 1, 1)
ax2 = plt.subplot(1, 1, 1)
ax1 = ax2.twinx()
ax1.plot(dfv['index'], dfv['V[mV]'], markevery=[0, -1])
ax2.plot(dfi['index'], dfi['V[mV]'], color= "darkcyan", markevery=[0, -1])
ax1.grid(which='major', linestyle='-')
ax2.grid(which='major', linestyle='-')

fig.tight_layout()
plt.show()
