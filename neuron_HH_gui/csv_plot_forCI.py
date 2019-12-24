import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib_scalebar.scalebar import ScaleBar
import numpy as np
from scipy import signal


def main():

       # for overflow error
       mpl.rcParams['agg.path.chunksize'] = 100000



       path = "I:/Box Sync/Personal/Experimental Data/20191218/Data64/csv/"
       csv_name = "Data64_roi_data.csv"
       fig_name = "output.png"

       # parameters
       target_dpi = 600
       config_dpi = 600
       ratio = target_dpi/config_dpi
       """inch"""
       graph_width = 4
       graph_hight = 2.5
       skip_rows = 0
       plot_line_w = 0.2
       ax_line_w = 2
       fsize = 8
       label_color = "black"

       fig = plt.figure(figsize=(graph_width*ratio, graph_hight*ratio), dpi=config_dpi)

       df = pd.read_csv(path+csv_name, delimiter=',', skiprows=skip_rows)
       df.fillna(0)

       # matplotlib
       ax0 = fig.add_subplot(1, 1, 1)

       print(len(df['Area1']))
       Time = np.arange(0, len(df['Area1']))/ 10
       Mean1 = df['Mean1']
       # drift removal
       #Mean1 = signal.detrend(Mean1) - np.min(signal.detrend(Mean1))

       #10 pulse
       start_pulse = 300
       ax0.axvline(start_pulse, linewidth=plot_line_w, color="orange")
       start_pulse += 10
       ax0.axvline(start_pulse, linewidth=plot_line_w, color="orange")
       start_pulse += 10
       ax0.axvline(start_pulse, linewidth=plot_line_w, color="orange")
       start_pulse += 10
       ax0.axvline(start_pulse, linewidth=plot_line_w, color="orange")
       start_pulse += 10
       ax0.axvline(start_pulse, linewidth=plot_line_w, color="orange")
       start_pulse += 10
       ax0.axvline(start_pulse, linewidth=plot_line_w, color="orange")
       start_pulse += 10
       ax0.axvline(start_pulse, linewidth=plot_line_w, color="orange")
       start_pulse += 10
       ax0.axvline(start_pulse, linewidth=plot_line_w, color="orange")
       start_pulse += 10
       ax0.axvline(start_pulse, linewidth=plot_line_w, color="orange")
       start_pulse += 10
       ax0.axvline(start_pulse, linewidth=plot_line_w, color="orange")
       start_pulse += 10
       ax0.axvline(start_pulse, linewidth=plot_line_w, color="orange")
       start_pulse += 10
       ax0.axvline(start_pulse, linewidth=plot_line_w, color="orange")
       start_pulse += 10
       ax0.axvline(start_pulse, linewidth=plot_line_w, color="orange")
       start_pulse += 10
       ax0.axvline(start_pulse, linewidth=plot_line_w, color="orange")
       start_pulse += 10
       ax0.axvline(start_pulse, linewidth=plot_line_w, color="orange")
       start_pulse += 10
       ax0.axvline(start_pulse, linewidth=plot_line_w, color="orange")
       start_pulse += 10
       ax0.axvline(start_pulse, linewidth=plot_line_w, color="orange")
       start_pulse += 10
       ax0.axvline(start_pulse, linewidth=plot_line_w, color="orange")
       start_pulse += 10
       ax0.axvline(start_pulse, linewidth=plot_line_w, color="orange")
       start_pulse += 10
       ax0.axvline(start_pulse, linewidth=plot_line_w, color="orange")

       ax0.plot(Time, Mean1,
                color="black", linewidth=plot_line_w, markevery=[0, -1], alpha=1)

       ax0.tick_params(labelsize=fsize, axis="x", colors=label_color)
       ax0.tick_params(labelsize=fsize, axis="y", colors=label_color)
       ax0.set_xlabel("time[s]", fontsize=fsize, color="black")
       ax0.set_ylabel("Fluorescence intensity [a.u.]", fontsize=fsize)

       ax0.spines["right"].set_color("none")
       ax0.spines["top"].set_color("none")
       ax0.spines["left"].set_linewidth(ax_line_w)
       ax0.spines["bottom"].set_linewidth(ax_line_w)

       plt.savefig(path + fig_name)
       fig.tight_layout()
       plt.show()

if __name__ == '__main__':
     main()