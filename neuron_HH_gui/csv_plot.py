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
       path = "G:/simulation/HH/tmp/2018_7_26_15_34_50/" + \
              "2018_7_25_8_37_26_N0P_AMPA0.3_P_NMDA0.2_Mg_conc0.4_HH.csv"
       """

       path = "//192.168.13.10/Public/ishida/simulation/dynamic_synapse_exp4/check/tmp/2018_9_13_10_35_17/" + \
              "2018_9_13_10_35_17__T_ 1000_ Iext_amp_ 5_ dt_ 0.04_ noise_ 3_ syncp_ 2_ U_SE_AMPA_ 0.5_ tau_rise_AMPA_ 1.0_ A_SE_AMPA_ 1_ A_SE_NMDA_ 0.49__N0_HH.csv"
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
       ax1 = fig.add_subplot(3, 1, 1)
       ax1.plot(df['T [ms]'], df['V [mV]'], color="black", markevery=[0, -1])
       ax1.tick_params(labelsize=fsize)
       ax1.tick_params(axis="x", colors="white")
       #ax1.set_xlabel("time[ms]", fontsize=fsize, color="gray")
       #ax1.set_ylabel("membrane potential[mV]", fontsize=fsize)

       ax1.spines["right"].set_color("none")
       ax1.spines["top"].set_color("none")
       ax1.spines["left"].set_linewidth(5)
       ax1.spines["bottom"].set_linewidth(5)


       ax0 = fig.add_subplot(3, 1, 2)
       ax0.plot(df['T [ms]'], df['I_NMDA [uA]'], color="black", markevery=[0, -1])
       ax0.tick_params(labelsize=fsize)
       ax0.tick_params(axis="x")

       ax0.spines["right"].set_color("none")
       ax0.spines["top"].set_color("none")
       ax0.spines["left"].set_linewidth(5)
       ax0.spines["bottom"].set_linewidth(5)


       ax0 = fig.add_subplot(3, 1, 3)
       ax0.plot(df['T [ms]'], df['I_AMPA [uA]'], color="black", markevery=[0, -1])
       ax0.tick_params(labelsize=fsize)
       ax0.tick_params(axis="x")

       ax0.spines["right"].set_color("none")
       ax0.spines["top"].set_color("none")
       ax0.spines["left"].set_linewidth(5)
       ax0.spines["bottom"].set_linewidth(5)

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
