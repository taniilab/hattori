import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib_scalebar.scalebar import ScaleBar
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
from scipy.optimize import curve_fit

def bimodal(x, mean1, sigma1, mean2, sigma2):
    return np.exp(-(x-mean1)**2/(2*sigma1))/np.sqrt(2*np.pi*sigma1) + \
           np.exp(-(x-mean2)**2/(2*sigma2))/np.sqrt(2*np.pi*sigma2)
def main():

       # for overflow error
       mpl.rcParams['agg.path.chunksize'] = 100000

       path = "C:/Users/Tanii_Lab/Box Sync/Personal/Paper/first/Data/gAMPA_vs_gNMDA/"
       path = "//192.168.13.10/Public/experimental data/tohoku_univ/tohoku_patch/20181018_cortex/voltage/"
       csv_name_cont = "voltage2.csv"
       csv_name_ap5 = "voltage5.csv"

       fig_name = "histw_zikuari.png"

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
       plt.xticks(color="black")
       plt.yticks(color="black")

       df = pd.read_csv(path+csv_name_cont, delimiter=',', skiprows=skip_rows)
       df.fillna(0)
       df2 = pd.read_csv(path + csv_name_ap5, delimiter=',', skiprows=skip_rows)
       df2.fillna(0)

       voltage = df['voltage(mV)']
       voltage2 = df2['voltage(mV)']
       kanopero = plt.hist([voltage.dropna(), voltage2.dropna()],
                color=["gray", "black"],
                bins=edges, normed=True)
       x_axis = kanopero[1]
       x_axis = np.delete(x_axis, -1, 0)
       data1 = kanopero[0][0]
       data2 = kanopero[0][1]
       plt.savefig(path + fig_name)


       print(kanopero[0][0])
       print("\n")
       print(kanopero[0][1])
       print("\n")
       print(kanopero[1])
       print("\n")
       fig = plt.figure()
       ax = fig.add_subplot(1,1,1)

       param, cov = curve_fit(bimodal, x_axis, data1)
       ax.plot(x_axis, data1)
       ax.plot(x_axis, bimodal(x_axis, param[0], param[1], param[2], param[3]))

       fig.tight_layout()
       plt.show()

if __name__ == '__main__':
     main()