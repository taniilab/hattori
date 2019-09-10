import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib_scalebar.scalebar import ScaleBar
import numpy as np
from scipy import signal


def main():

       # for overflow error
       mpl.rcParams['agg.path.chunksize'] = 100000



       path = "C:/Users/6969p/Downloads/experimental_data/20190715/"
       csv_name = "444.csv"
       fig_name = "output.png"

       # parameters
       sample = 20000
       target_dpi = 600
       config_dpi = 600
       ratio = target_dpi/config_dpi
       """inch"""
       graph_width = 10
       graph_hight = 5
       skip_rows = 0
       plot_line_w = 2
       ax_line_w = 4
       fsize = 20
       label_color = "black"

       fig = plt.figure(figsize=(graph_width*ratio, graph_hight*ratio), dpi=config_dpi)

       df = pd.read_csv(path+csv_name, delimiter=',', skiprows=skip_rows)
       df.fillna(0)

       # matplotlib
       ax0 = fig.add_subplot(1, 1, 1)
       init = int(1)
       last = int(-1)

       print(len(df['Area1']))
       Time = 3 * (np.arange(0, len(df['Area1']))) / len(df['Area1'])
       Fmax = 4095
       Mean1 = df['Mean1']/Fmax
       """
       Mean2 = df['Mean2']/Fmax
       Mean3 = df['Mean3']/Fmax
       
       Mean4 = df['Mean4']/Fmax
       Mean5 = df['Mean5']/Fmax
       Mean6 = df['Mean6']/Fmax
       Mean7 = df['Mean7']/Fmax
       Mean8 = df['Mean8']/Fmax
       """
       # drift removal
       Mean1 = signal.detrend(Mean1) - np.min(signal.detrend(Mean1))
       """
       Mean2 = signal.detrend(Mean2) - np.min(signal.detrend(Mean2))
       Mean3 = signal.detrend(Mean3) - np.min(signal.detrend(Mean3))
       
       Mean4 = signal.detrend(Mean4) - np.min(signal.detrend(Mean4))
       Mean5 = signal.detrend(Mean5) - np.min(signal.detrend(Mean5))
       Mean6 = signal.detrend(Mean6) - np.min(signal.detrend(Mean6))
       Mean7 = signal.detrend(Mean7) - np.min(signal.detrend(Mean7))
       Mean8 = signal.detrend(Mean8) - np.min(signal.detrend(Mean8))
       """
       ax0.plot(Time, Mean1,
                color="black", linewidth=plot_line_w, markevery=[0, -1], alpha=1)

       """
       ax0.plot(Time, Mean2,
                color="maroon", linewidth=plot_line_w, markevery=[0, -1], alpha=1)
       ax0.plot(Time, Mean3,
                color="dodgerblue", linewidth=plot_line_w, markevery=[0, -1], alpha=1)
       
       ax0.plot(Time, Mean8,
                color="black", linewidth=plot_line_w, markevery=[0, -1], alpha=1)
       """
       ax0.set_ylim([0, 0.2])

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
       #plt.show()

if __name__ == '__main__':
     main()