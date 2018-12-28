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

       path = "C:/Users/Tanii_Lab/Box Sync/Personal/Paper/first/Data/gAMPA_vs_gNMDA/"
       path = "//192.168.13.10/Public/experimental data/touhoku_patch/20181018_cortex/voltage/"
       csv_name_cont = "voltage2.csv"
       csv_name_ap5 = "voltage5.csv"

       fig_name = "histw.png"

       # parameters
       sample = 20000
       target_dpi = 600
       config_dpi = 600
       ratio = target_dpi/config_dpi
       """inch"""
       graph_width = 15
       graph_hight = 15
       skip_rows = 0

       edges = range(-70, 0, 2)

       fig = plt.figure(figsize=(graph_width*ratio, graph_hight*ratio), dpi=config_dpi)
       plt.xticks(color="None")
       plt.yticks(color="None")

       df = pd.read_csv(path+csv_name_cont, delimiter=',', skiprows=skip_rows)
       df.fillna(0)
       df2 = pd.read_csv(path + csv_name_ap5, delimiter=',', skiprows=skip_rows)
       df2.fillna(0)

       voltage = df['voltage(mV)']
       voltage2 = df2['voltage(mV)']
       plt.hist([voltage.dropna(), voltage2.dropna()],
                color=["gray", "black"],
                bins=edges, normed=True)
       plt.savefig(path + fig_name)

       fig.tight_layout()
       #plt.show()

if __name__ == '__main__':
     main()