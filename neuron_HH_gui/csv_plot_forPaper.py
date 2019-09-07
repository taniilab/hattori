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

       path = "Z:/simulation/HH/2018_9_19_14_23_37_N0_P_AMPA0.4_P_NMDA0.5_Mg_conc1.0_delay4.6HH.csv"
       path = "//192.168.13.10/Public/ishida/simulation/dynamic_synapse_exp13__Mg=2~10/tmp/2018_9_22_1_8_21/" + \
              "2018_9_20_20_47_5__T_ 70000_ Iext_amp_ 10_ Mg_ 2_ noise_ 2_ syncp_ 2_ U_SE_AMPA_ 0.7_ A_SE_AMPA_ 0.9_ A_SE_NMDA_ 0.9__N0_HH.csv"
       path = "//192.168.13.10/Public/hattori/simulation/HH/raw_data/2018_10_10_9_46_37/Mg_0.4/" + \
              "2018_10_9_20_52_7_N0_P_AMPA0.4_P_NMDA0.2_Mg_conc0.4_delay0HH.csv"
       path = "//192.168.13.10/Public/experimental data/touhoku_patch/20181018_cortex/voltage/voltage6.csv"

       path = "C:/Users/Tanii_Lab/Box Sync/Personal/Paper/first/Data/gAMPA_vs_gNMDA/"
       path = "//192.168.13.10/Public/experimental data/touhoku_patch/20181018_cortex/voltage/"
       path = "C:/Users/Tanii_Lab/Box Sync/Personal/Paper/first/Data/gAMPA_vs_gNMDA/"
       path = "Z:/Box Sync/Personal/Paper/first/Data/gAMPA_vs_gNMDA/"
       csv_name = "2018_12_18_19_51_31_N0_P_AMPA0.5_P_NMDA0.7_Mg_conc1.6_delay0HH.csv"
       fig_name = "fig3_a05n07_isyn.png"

       # parameters
       sample = 20000
       target_dpi = 600
       config_dpi = 600
       ratio = target_dpi/config_dpi
       """inch"""
       graph_width = 15
       graph_hight = 5
       skip_rows = 1
       plot_line_w = 2
       ax_line_w = 6
       fsize = 8
       label_color = "white"

       fig = plt.figure(figsize=(graph_width*ratio, graph_hight*ratio), dpi=config_dpi)

       df = pd.read_csv(path+csv_name, delimiter=',', skiprows=skip_rows)
       df.fillna(0)

       # matplotlib
       ax0 = fig.add_subplot(1, 1, 1)
       #ax0.set_ylim(-2, 8)
       #19-21.5 2.5s
       init = int(1)
       last = int(-1)

       #simulation

       index = df['T [ms]']
       voltage = df['V [mV]']
       nmda = df['I_NMDA [uA]']
       ampa = df['I_AMPA [uA]']
       ax0.plot(index[int(2000/0.02):int(2500/0.02)], ampa[int(2000/0.02):int(2500/0.02)],
                color="red", linewidth=plot_line_w, markevery=[0, -1], alpha=1)
       ax0.plot(index[int(2000 / 0.02):int(2500 / 0.02)], nmda[int(2000 / 0.02):int(2500 / 0.02)],
                color="blue", linewidth=plot_line_w, markevery=[0, -1], alpha=1)
       ax0.set_ylim(-2, 8)
       """

       #experiment
       index = df['index']
       voltage = df['voltage(mV)']
       ax0.plot(index[init:last]/200000, voltage[init:last],
                color="black", linewidth=plot_line_w, markevery=[0, -1], alpha=1)
       """

       ax0.tick_params(labelsize=fsize, axis="x", colors=label_color)
       ax0.tick_params(labelsize=fsize, axis="y", colors=label_color)
       #ax1.set_xlabel("time[ms]", fontsize=fsize, color="gray")
       #ax1.set_ylabel("membrane potential[mV]", fontsize=fsize)

       ax0.spines["right"].set_color("none")
       ax0.spines["top"].set_color("none")
       ax0.spines["left"].set_linewidth(ax_line_w)
       ax0.spines["bottom"].set_linewidth(ax_line_w)

       plt.savefig(path + fig_name)

       fig.tight_layout()
       plt.show()

if __name__ == '__main__':
     main()