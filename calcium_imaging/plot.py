import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib_scalebar.scalebar import ScaleBar
import numpy as np
import glob


def main():

       # for overflow error
       mpl.rcParams['agg.path.chunksize'] = 100000
       # parameters
       sample = 20000
       target_dpi = 300
       config_dpi = 300
       ratio = target_dpi/config_dpi
       """inch"""
       graph_width = 8
       graph_hight = 4.5
       skip_rows = 0
       plot_line_w = 0.5
       ax_line_w = 2
       fsize = 8
       label_color = "black"

       path = "C:/Users/Tanii_Lab/Desktop/tmp/"
       csv_name = "Results*.csv"
       tmp = glob.glob(path+csv_name)
       df = []

       for i in range(len(tmp)):

           df.append(pd.read_csv(tmp[i], index_col=0, skiprows=skip_rows))
           df[i].fillna(0)

       # matplotlib
       res =pd.concat([df[1], df[6]], ignore_index=True)
       fig = plt.figure(figsize=(graph_width*ratio, graph_hight*ratio), dpi=config_dpi)
       ax0 = fig.add_subplot(2, 1, 1)
       ax1 = fig.add_subplot(2, 1, 2)
       print(res)

       ax0.plot(res["Mean1"],
                color="darkviolet", linewidth=plot_line_w, markevery=[0, -1], alpha=1)
       ax0.plot(res["Mean2"],
                color="blue", linewidth=plot_line_w, markevery=[0, -1], alpha=1)

       ax0.plot(res["Mean3"],
                color="black", linewidth=plot_line_w, markevery=[0, -1], alpha=1)

       ax0.tick_params(labelsize=fsize, axis="x", colors=label_color)
       ax0.tick_params(labelsize=fsize, axis="y", colors=label_color)
       #ax1.set_xlabel("time[ms]", fontsize=fsize, color="gray")
       #ax1.set_ylabel("membrane potential[mV]", fontsize=fsize)

       ax0.spines["right"].set_color("none")
       ax0.spines["top"].set_color("none")
       ax0.spines["left"].set_linewidth(ax_line_w)
       ax0.spines["bottom"].set_linewidth(ax_line_w)
       ax1.spines["right"].set_color("none")
       ax1.spines["top"].set_color("none")
       ax1.spines["left"].set_linewidth(ax_line_w)
       ax1.spines["bottom"].set_linewidth(ax_line_w)
       fig.tight_layout()
       plt.show()

if __name__ == '__main__':
     main()