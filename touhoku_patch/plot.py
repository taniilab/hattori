#%matplotlib notebook
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore

# for overflow error
mpl.rcParams['agg.path.chunksize'] = 100000

index = "15"
path = "C:/Box Sync/Personal/Documents/touhoku_patch/20180713_cortex/"
path_h = path + "voltage/voltage" + index + ".csv"
path_i = path + "current/current" + index + ".csv"

fsize = 24
sample = 20000
fig = plt.figure(figsize=(30, 15))

dfv = pd.read_csv(path_h, delimiter=',')
dfc = pd.read_csv(path_i, delimiter=',')
dfv.fillna(0)
dfc.fillna(0)

glaph_tab = pg.GraphicsWindow(title="four terminal voltage")
p1 = glaph_tab.addPlot(title="Vx1")
curve1 = p1.plot(dfv['index']/sample, dfv['voltage(mV)'])

ax2 = fig.add_subplot(1, 1, 1)

ax2.plot(dfc['index']/sample, dfc['current(pA)'], markevery=[0, -1], color="purple")
ax2.tick_params(labelsize=fsize)
ax2.set_xlabel("time[s]", fontsize=fsize)
ax2.set_ylabel("clamp current[pA]", fontsize=fsize)


ax1 = ax2.twinx()
ax1.plot(dfv['index']/sample, dfv['voltage(mV)'], markevery=[0, -1])
ax1.tick_params(labelsize=fsize)
ax1.set_xlabel("time[s]", fontsize=fsize)
ax1.set_ylabel("membrane potential[mV]", fontsize=fsize)

fig.tight_layout()
plt.show()