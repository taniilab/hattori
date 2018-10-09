import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns
import pyqtgraph as pg


# for overflow error
mpl.rcParams['agg.path.chunksize'] = 100000

path = "//192.168.13.10/Public/hattori/simulation/HH/rawdata/2018_9_22_9_50_8/" + \
       "2018_9_20_22_59_16_N0_P_AMPA0.3_P_NMDA0.2_Mg_conc0.4_delay0HH.csv"

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

t = np.arange(0, 100, 0.01)
y = np.sin(t)

ac = np.correlate(y, y)
#ac = np.correlate(df['V [mV]']+70, df['V [mV]']+70)

print(y)
print(ac)

ax1 = fig.add_subplot(1, 1, 1)
ax1.plot(ac)
#ax1.plot(df['T [ms]'], df['V [mV]'], markevery=[0, -1])
ax1.tick_params(labelsize=fsize)
ax1.set_xlabel("time[ms]", fontsize=fsize)
ax1.set_ylabel("membrane potential[mV]", fontsize=fsize)

fig.tight_layout()
plt.show()