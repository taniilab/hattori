#%matplotlib notebook
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 2 cells burst 13.csv
#single cell burst 14.csv
# FI curve  0.csv
path_v = """C:/Users/Hattori/Documents/2017_11_12_touhoku/files_voltage/0_voltage.csv"""
path_v2 = """C:/Users/Hattori/Documents/2017_11_12_touhoku/files_voltage/14_voltage.csv"""
path_h = "C:/Users/Hattori/Documents/HR_results/2017_6_22_1_25_51_0_D_2_alpha_0.5_beta_0.0_tausyn_6_Pmax_0.0_HR.csv"

fsize = 24

fig = plt.figure(figsize=(10, 7))
plt.tick_params(labelsize=fsize)
dfv = pd.read_csv(path_v, delimiter=',')
dfv2 = pd.read_csv(path_v2, delimiter=',')
ax1 = plt.subplot(2, 1, 1)
ax2 = plt.subplot(2, 1, 2)
ax1.plot(dfv['index']/20000, dfv['V[mV]'], markevery=[0, -1])
ax2.plot(dfv2['index']/20000, dfv2['V[mV]'], markevery=[0, -1])
#ax1.plot(dfv['index'], dfv['x'], markevery=[0, -1])
ax1.grid(which='major', linestyle='-')
plt.tick_params(labelsize=fsize)

fig.tight_layout()
plt.show()
