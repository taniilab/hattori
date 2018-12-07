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
       """
       path = "//192.168.13.10/Public/ishida/simulation/dynamic_synapse_exp13__Mg=2~10/tmp/2018_9_22_1_8_21/" + \
              "2018_9_20_20_47_5__T_ 70000_ Iext_amp_ 10_ Mg_ 2_ noise_ 2_ syncp_ 2_ U_SE_AMPA_ 0.7_ A_SE_AMPA_ 0.9_ A_SE_NMDA_ 0.9__N0_HH.csv"
       """

       """
       path = "//192.168.13.10/Public/hattori/simulation/HH/raw_data/2018_10_10_9_46_37/Mg_0.4/" + \
              "2018_10_9_20_52_7_N0_P_AMPA0.4_P_NMDA0.2_Mg_conc0.4_delay0HH.csv"
       """
       path = "C:/Users/Tanii_Lab/Box Sync/Personal/Paper/first/Data/EPSC_NMDA_AMPA/" + \
              "fig1.csv"

       """
       path = "//192.168.13.10/Public/nakanishi/simulation/2018_9_21 depression synapse Mg_0.5~1.0/tmp/2018_9_23_12_25_20/" + \
              "2018_9_21_23_2_51__P_AMPA_0.6_P_NMDA_0.7_Mg_conc_0.5_delay_0_N0_HH.csv"
       """


       sample = 20000
       target_dpi = 600
       config_dpi = 600
       ratio = target_dpi/config_dpi
       """inch"""
       graph_width = 6.6929
       graph_hight = 1

       line_w = 1.0
       fsize = 8

       fig = plt.figure(figsize=(graph_width*ratio, graph_hight*ratio), dpi=config_dpi)

       df = pd.read_csv(path, delimiter=',', skiprows=1)
       df.fillna(0)
       """
       glaph_tab = pg.GraphicsWindow(title="four terminal voltage")
       p1 = glaph_tab.addPlot(title="Vx1")
       curve1 = p1.plot(df['T [ms]'], df['V [mV]'])
       """


       #pyqtgraph
       """
       pg.setConfigOption('background', (255,255,255))
       pg.setConfigOption('foreground', (0,0,0))
       glaph_tab = pg.GraphicsWindow(title="single autaptic neuron")
       p1 = glaph_tab.addPlot(title="Vx1")
       p1.showGrid(True, True, 0.2)
       curve1 = p1.plot(df['T [ms]'], df['V [mV]'], pen=(0,0,0))

       glaph_tab.nextRow()
       p2 = glaph_tab.addPlot(title="Vx1")
       p2.showGrid(True, True, 0.2)
       curve1 = p2.plot(df['T [ms]'], df['I_AMPA [uA]'], pen=(200,0,0))
       curve1 = p2.plot(df['T [ms]'], df['I_NMDA [uA]'], pen=(0,100,100))
       """

       #matplotlib
       ax0 = fig.add_subplot(2, 1, 1)
       ax0.plot(df['T [ms]'], df['Iext [uA]']/0.46499,
                color="black",
                linewidth=line_w, markevery=[0, -1])
       ax0.tick_params(labelsize=fsize)
       ax0.tick_params(axis="x", colors="white")
       #ax1.set_xlabel("time[ms]", fontsize=fsize, color="gray")
       #ax1.set_ylabel("membrane potential[mV]", fontsize=fsize)

       ax0.spines["right"].set_color("none")
       ax0.spines["top"].set_color("none")
       ax0.spines["left"].set_linewidth(line_w)
       ax0.spines["bottom"].set_linewidth(line_w)


       ax1 = fig.add_subplot(2, 1, 2)
       ax1.plot(df['T [ms]'], df['Iext [uA]_s']/0.46499,
                color="black",
                linewidth=line_w, markevery=[0, -1])
       ax1.tick_params(labelsize=fsize)
       ax1.tick_params(axis="x")

       ax1.spines["right"].set_color("none")
       ax1.spines["top"].set_color("none")
       ax1.spines["left"].set_linewidth(line_w)
       ax1.spines["bottom"].set_linewidth(line_w)

       plt.savefig("C:/Users/Tanii_Lab/fig1_3.png")

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

       fig2 = plt.figure(figsize=(21, 14))

       ax2 = fig2.add_subplot(1, 1, 1)
       ax2.plot(df['T [ms]'], df['I [mV]'], color="black", markevery=[0, -1])
       ax2.tick_params(labelsize=fsize)
       ax2.tick_params(axis="x", colors="white")
       """


       fig.tight_layout()
       # plt.show()


if __name__ == '__main__':
     main()