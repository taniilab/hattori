import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore

# for overflow error
mpl.rcParams['agg.path.chunksize'] = 100000

path = "E:/simulation/HH/tmp/2018_6_24_21_57_17/" + \
       "2018_6_24_21_29_8__T_ 60000_ dt_ 0.05_ Iext_amp_ 0_ eK_ -90_ syncp_ 5_ noise_ 2_ gK_ 3_ gpNa_ 0_ Pmax_AMPA_ 0.3_ Pmax_NMDA_ 0.5_ gtCa_ 0.4_ Mg_conc_ 1_ alpha_ 0.5_ beta_ 0.1_ D_ 0.05_ delay_ 10__N0_HH.csv"
fsize = 24
sample = 20000
fig = plt.figure(figsize=(30, 15))

df = pd.read_csv(path, delimiter=',', skiprows=1)
df.fillna(0)

glaph_tab = pg.GraphicsWindow(title="four terminal voltage")
p1 = glaph_tab.addPlot(title="Vx1")
curve1 = p1.plot(df['T [ms]'], df['V [mV]'])

ax2 = fig.add_subplot(1, 1, 1)
ax2.plot(df['T [ms]'], df['I_noise [uA]']+df['I_syn [uA]'], markevery=[0, -1], color="thistle")
ax2.tick_params(labelsize=fsize)
ax2.set_xlabel("time[ms]", fontsize=fsize)
ax2.set_ylabel("autaptic current[uA]", fontsize=fsize)

ax1 = ax2.twinx()
ax1.plot(df['T [ms]'], df['V [mV]'], markevery=[0, -1])
ax1.tick_params(labelsize=fsize)
ax1.set_xlabel("time[ms]", fontsize=fsize)
ax1.set_ylabel("membrane potential[mV]", fontsize=fsize)

fig.tight_layout()
plt.show()