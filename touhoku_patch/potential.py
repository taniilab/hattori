#%matplotlib notebook
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

"""
discription of index

0:cell1 state-ficurve-ficurve-ficurve-state ?
1:cell1 ficurve
2:cell1 state
3:cell2 ficurve
4:cell2 state
5:cell2 ficurve
6:cell2 state
7:cell2 1pulse
8:cell3 ficurve
9:cell3 state
10:cell4 ficurve
11:cell4 state
12:cell4 1pulse
13:cell5 ficurve
14:cell5 state
15:cell6 ficurve ?
16:cell6 state ?
17:cell6 ficurve
18:cell6 1pulse
"""

index = "3"
path = "C:/Box Sync/Personal/Documents/touhoku_patch/20180501_cortex/"
path_h = path + "voltage/" + index + ".csv"
path_i = path + "current/" + index + ".csv"

fsize = 24
sample = 20000
fig = plt.figure(figsize=(10, 7))

dfv = pd.read_csv(path_h, delimiter=',')
dfc = pd.read_csv(path_i, delimiter=',')

ax2 = fig.add_subplot(1, 1, 1)
ax2.plot(dfc['index']/sample, dfc['I[pA]'], markevery=[0, -1], color="purple")
ax2.tick_params(labelsize=fsize)
ax2.set_xlabel("time[s]", fontsize=fsize)
ax2.set_ylabel("clamp current[pA]", fontsize=fsize)

ax1 = ax2.twinx()
ax1.plot(dfv['index']/sample, dfv['V[mV]'], markevery=[0, -1])
ax1.tick_params(labelsize=fsize)
ax1.set_xlabel("time[s]", fontsize=fsize)
ax1.set_ylabel("membrane potential[mV]", fontsize=fsize)

fig.tight_layout()
plt.show()
