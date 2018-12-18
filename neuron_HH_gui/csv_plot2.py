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

       path  = "//192.168.13.10/Public/hattori/simulation/HH/raw_data/2018_10_10_9_46_37(maindata)/Mg_1.0/raw_data/2018_12_7_14_13_43/"
       csv_name = "2018_10_9_21_18_47_N0_P_AMPA0.6_P_NMDA0.4_Mg_conc1.0_delay0HH.csv"


       sample = 20000
       target_dpi = 600
       config_dpi = 600
       ratio = target_dpi/config_dpi
       """inch"""
       graph_width = 6.6929
       graph_hight = 4

       line_w = 0.5
       fsize = 8

       fig = plt.figure(figsize=(graph_width*ratio, graph_hight*ratio), dpi=config_dpi)

       df = pd.read_csv(path+csv_name, delimiter=',', skiprows=1)
       df.fillna(0)
       #pyqtgraph
       """
       glaph_tab = pg.GraphicsWindow(title="four terminal voltage")
       p1 = glaph_tab.addPlot(title="Vx1")
       curve1 = p1.plot(df['T [ms]'], df['V [mV]'])

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
       ax0 = fig.add_subplot(1, 1, 1)
       ax0.plot(df['T [ms]'], df['I_AMPA [uA]'],
                color="black",
                linewidth=line_w, markevery=[0, -1])
       ax0.plot(df['T [ms]'], df['I_NMDA [uA]'],
                color="black",linestyle="dashed",
                linewidth=line_w, markevery=[0, -1])

       ax0.tick_params(labelsize=fsize)
       ax0.tick_params(axis="x", colors="white")
       #ax1.set_xlabel("time[ms]", fontsize=fsize, color="gray")
       #ax1.set_ylabel("membrane potential[mV]", fontsize=fsize)

       ax0.spines["right"].set_color("none")
       ax0.spines["top"].set_color("none")
       ax0.spines["left"].set_linewidth(line_w)
       ax0.spines["bottom"].set_linewidth(line_w)


       """
       ax1 = fig.add_subplot(2, 1, 2)
       ax1.plot(df['T [ms]'], df['I_NMDA [uA]']/0.46499,
                color="black",
                linewidth=line_w, markevery=[0, -1])
       ax1.tick_params(labelsize=fsize)
       ax1.tick_params(axis="x")

       ax1.spines["right"].set_color("none")
       ax1.spines["top"].set_color("none")
       ax1.spines["left"].set_linewidth(line_w)
       ax1.spines["bottom"].set_linewidth(line_w)
       """

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

       plt.savefig(path + "fig1_3.png")
       fig.tight_layout()
       plt.show()


if __name__ == '__main__':
     main()