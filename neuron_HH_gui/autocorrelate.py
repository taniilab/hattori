@ -1,36 +0,0 @@
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns
import pyqtgraph as pg


# for overflow error
mpl.rcParams['agg.path.chunksize'] = 100000

path = "E:/simulation/" + \
       "2018_6_25_17_1_12__T_ 100000_ dt_ 0.04_ Iext_amp_ 0_ eK_ -90_ syncp_ 5_ noise_ 2_ gK_ 3_ gpNa_ 0_ Pmax_AMPA_ 0.4_ Pmax_NMDA_ 0.5_ gtCa_ 0.4_ Mg_conc_ 1_ alpha_ 0.5_ beta_ 0.1_ D_ 0.05_ delay_ 10__N0_HH.csv"
fsize = 72
sample = 20000
fig = plt.figure(figsize=(21, 14))

df = pd.read_csv(path, delimiter=',', skiprows=1)
df.fillna(0)
"""
glaph_tab = pg.GraphicsWindow(title="four terminal voltage")
p1 = glaph_tab.addPlot(title="Vx1")
curve1 = p1.plot(df['T [ms]'], df['V [mV]'])
"""

ac = np.correlate(df['V [mV]']+70, df['V [mV]']+70)

ax1 = fig.add_subplot(1, 1, 1)
ax1.plot(ac)
#ax1.plot(df['T [ms]'], df['V [mV]'], markevery=[0, -1])
ax1.tick_params(labelsize=fsize)
ax1.set_xlabel("time[ms]", fontsize=fsize)
ax1.set_ylabel("membrane potential[mV]", fontsize=fsize)

fig.tight_layout()
plt.show()