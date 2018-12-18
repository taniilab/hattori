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
       path = "C:/Users/Tanii_Lab/Box Sync/Personal/Paper/first/Data/" + \
              "test.csv"

       path = "//192.168.13.10/Public/experimental data/touhoku_patch/20181018_cortex/voltage/voltage6.csv"

       sample = 20000
       target_dpi = 600
       config_dpi = 600
       ratio = target_dpi/config_dpi
       """inch"""
       graph_width = 6.6929
       graph_hight = 10
       skip_rows = 0
       line_w = 0.5
       fsize = 8


       fig = plt.figure(figsize=(graph_width*ratio, graph_hight*ratio), dpi=config_dpi)

       df = pd.read_csv(path, delimiter=',', skiprows=skip_rows)
       df.fillna(0)

       #matplotlib
       ax0 = fig.add_subplot(1, 1, 1)

       """
       ax0.plot(df['T [ms]'], df['I_NMDA [uA]'],
                color="darkturquoise", linewidth=line_w, markevery=[0, -1], alpha=0.8)
       ax0.plot(df['T [ms]'], df['I_AMPA [uA]'],
                color="purple", linewidth=line_w, markevery=[0, -1], alpha=0.8)
       """
       index = df['index']
       voltage = df['voltage(mV)']
       init = 0
       last = 10000
       ax0.plot(index[init:last], voltage[init:last],
                color="black", linewidth=line_w, markevery=[0, -1], alpha=0.8)

       ax0.tick_params(labelsize=fsize)
       ax0.tick_params(axis="x", colors="white")
       #ax1.set_xlabel("time[ms]", fontsize=fsize, color="gray")
       #ax1.set_ylabel("membrane potential[mV]", fontsize=fsize)

       ax0.spines["right"].set_color("none")
       ax0.spines["top"].set_color("none")
       ax0.spines["left"].set_linewidth(line_w)
       ax0.spines["bottom"].set_linewidth(line_w)

       plt.savefig("C:/Users/Tanii_Lab/fig1.png")

       fig.tight_layout()
       plt.show()

if __name__ == '__main__':
     main()