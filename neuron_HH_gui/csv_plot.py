import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib_scalebar.scalebar import ScaleBar
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore

def main():

       # for overflow error
       mpl.rcParams['agg.path.chunksize'] = 100000

       path = "G:/simulation/HH/tmp/2018_7_26_15_34_50/" + \
              "2018_7_25_8_37_26_N0P_AMPA0.3_P_NMDA0.2_Mg_conc0.4_HH.csv"
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
       ax1 = fig.add_subplot(1, 1, 1)
       ax1.plot(df['T [ms]'], df['I_syn [uA]'], color="black", markevery=[0, -1])
       ax1.tick_params(labelsize=fsize)
       ax1.tick_params(axis="x", colors="white")
       #ax1.set_xlabel("time[ms]", fontsize=fsize, color="gray")
       #ax1.set_ylabel("membrane potential[mV]", fontsize=fsize)

       ax1.spines["right"].set_color("none")
       ax1.spines["top"].set_color("none")
       ax1.spines["left"].set_linewidth(5)
       ax1.spines["bottom"].set_linewidth(5)

       """
       scalebar = ScaleBar(dx=1.0)
       #scalebar.border_pad(0.5)
       ax1.add_artist(scalebar)
       """
       """
       ax2 = ax1.twinx()
       ax2.plot(df['T [ms]'], df['I_noise [uA]']+df['I_syn [uA]'], markevery=[0, -1], color="thistle")
       ax2.tick_params(labelsize=fsize)
       ax2.set_xlabel("time[ms]", fontsize=fsize)
       ax2.set_ylabel("autaptic current[uA]", fontsize=fsize)
       """
       """
       fig.tight_layout()
       plt.show()

       fig2 = plt.figure(figsize=(21, 14))

       ax2 = fig2.add_subplot(1, 1, 1)
       ax2.plot(df['T [ms]'], df['I [mV]'], color="black", markevery=[0, -1])
       ax2.tick_params(labelsize=fsize)
       ax2.tick_params(axis="x", colors="white")
       """


if __name__ == '__main__':
     main()
