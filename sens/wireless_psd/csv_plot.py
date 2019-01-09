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

       """
       path = "Z:/simulation/HH/2018_9_19_14_23_37_N0_P_AMPA0.4_P_NMDA0.5_Mg_conc1.0_delay4.6HH.csv"
       """

       path = "//192.168.13.10/Public/hattori/sens/" + \
              "wirelsespsd_2018_9_26_15_14_31.csv"

       fsize = 72
       sample = 20000
       fig = plt.figure(figsize=(21, 14))

       df = pd.read_csv(path, delimiter=',')
       df.fillna(0)
       """
       glaph_tab = pg.GraphicsWindow(title="four terminal voltage")
       p1 = glaph_tab.addPlot(title="Vx1")
       curve1 = p1.plot(df['T [ms]'], df['V [mV]'])
       """
       print(df)
       ax1 = fig.add_subplot(3, 1, 1)
       ax1.plot(df['index'], df['vx1 [V]'], color="black", markevery=[0, -1])
       ax1.tick_params(labelsize=fsize)
       ax1.tick_params(axis="x", colors="white")
       #ax1.set_xlabel("time[ms]", fontsize=fsize, color="gray")
       #ax1.set_ylabel("membrane potential[mV]", fontsize=fsize)

       ax1.spines["right"].set_color("none")
       ax1.spines["top"].set_color("none")
       ax1.spines["left"].set_linewidth(5)
       ax1.spines["bottom"].set_linewidth(5)
       ax1.set_ylim(0, 5)


       ax2 = fig.add_subplot(3, 1, 2)
       ax2.plot(df['index'], df['vx2 [V]'], color="black", markevery=[0, -1])
       ax2.tick_params(labelsize=fsize)
       ax2.tick_params(axis="x")

       ax2.spines["right"].set_color("none")
       ax2.spines["top"].set_color("none")
       ax2.spines["left"].set_linewidth(5)
       ax2.spines["bottom"].set_linewidth(5)
       ax2.set_ylim(0, 5)

       ax3 = fig.add_subplot(3, 1, 3)
       ax3.plot(df['index'], df['vy1 [V]'], color="black", markevery=[0, -1])
       ax3.tick_params(labelsize=fsize)
       ax3.tick_params(axis="x")

       ax3.spines["right"].set_color("none")
       ax3.spines["top"].set_color("none")
       ax3.spines["left"].set_linewidth(5)
       ax3.spines["bottom"].set_linewidth(5)
       ax3.set_ylim(0, 5)

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
       fig.tight_layout()
       plt.show()
       """

       fig2 = plt.figure(figsize=(21, 14))

       ax2 = fig2.add_subplot(1, 1, 1)
       ax2.plot(df['T [ms]'], df['I [mV]'], color="black", markevery=[0, -1])
       ax2.tick_params(labelsize=fsize)
       ax2.tick_params(axis="x", colors="white")
       """


if __name__ == '__main__':
     main()