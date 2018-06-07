#%matplotlib notebook
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 2 cells burst 13.csv
#single cell burst 14.csv
# FI curve  0.csv
path_v = """C:/Users/Hattori/Documents/2017_11_12_touhoku/files_voltage/0_voltage.csv"""
path_v2 = """C:/Users/Hattori/Documents/2017_11_12_touhoku/files_voltage/13_voltage.csv"""
#path_h = "C:/Users/Hattori/Documents/HR_results/2017_6_22_1_25_51_0_D_2_alpha_0.5_beta_0.0_tausyn_6_Pmax_0.0_HR.csv"
#path_v = """C:/Users/6969p/Downloads/2017_11_12_touhoku/files_voltage/0.csv"""
#path_v2 = """C:/Users/6969p/Downloads/2017_11_12_touhoku/files_voltage/14.csv"""
path_i = "C:/Users/6969p/Downloads/2017_11_12_touhoku/files_current/0.csv"
path_h = "C:/Users/Hattori/Documents/HR_results/2017_6_22_1_25_51_0_D_2_alpha_0.5_beta_0.0_tausyn_6_Pmax_0.0_HR.csv"


fsize = 24
fig = plt.figure(figsize=(10, 7))

dfv = pd.read_csv(path_v, delimiter=',')
dfv2 = pd.read_csv(path_v2, delimiter=',')
"""
ax1 = fig.add_subplot(1, 1, 1)
ax1.plot(dfv['index']/20000, dfv['V[mV]'], markevery=[0, -1])
ax1.tick_params(labelsize=fsize)
ax1.set_xlabel("time[s]", fontsize=fsize)
ax1.set_ylabel("membrane potential[mV]", fontsize=fsize)

"""
ax2 = fig.add_subplot(1, 1, 1)
ax2.plot(dfv2['index']/20000, dfv2['V[mV]'], markevery=[0, -1])
ax2.tick_params(labelsize=fsize)
ax2.set_xlabel("time[s]", fontsize=fsize)
ax2.set_ylabel("membrane potential[mV]", fontsize=fsize)


#ax1.plot(dfv['index'], dfv['x'], markevery=[0, -1])

fig.tight_layout()
plt.show()
