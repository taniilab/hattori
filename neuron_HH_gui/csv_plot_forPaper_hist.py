import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.optimize import curve_fit
import scipy.interpolate



def Bimodal_gaussian(x, A, sigma1, mean1, sigma2, mean2):
    return (A*np.exp(-(x-mean1)**2/(2*sigma1))/np.sqrt(2*np.pi*sigma1) + \
            (1-A)*np.exp(-(x-mean2)**2/(2*sigma2))/np.sqrt(2*np.pi*sigma2))


def Gaussian(x, A, sigma1, mean1):
    return A*np.exp(-(x-mean1)**2/(2*sigma1))/np.sqrt(2*np.pi*sigma1)


def main():

       # for overflow error
       mpl.rcParams['agg.path.chunksize'] = 100000

       path = "//192.168.13.10/Public/experimental data/tohoku_univ/tohoku_patch/20181018_cortex/voltage/"
       csv_name = "ap5.csv"
       fig_name = "hist_fit.png"

       # parameters
       sample = 20000
       target_dpi = 600
       config_dpi = 600
       ratio = target_dpi/config_dpi
       """inch"""
       graph_width = 15
       graph_hight = 15
       skip_rows = 0
       plot_line_w = 5
       ax_line_w = 6
       fsize = 8
       label_color = "white"
       line_color_b = "green"
       line_color_a = "purple"

       df = pd.read_csv(path+csv_name, delimiter=',', skiprows=skip_rows)
       df.fillna(0)

       # matplotlib
       fig = plt.figure(figsize=(graph_width*ratio, graph_hight*ratio), dpi=config_dpi)
       ax0 = fig.add_subplot(1, 1, 1)
       voltage = df['V [mV]']
       before = df['before']
       after = df['after']
       x = np.arange(-75, 0, 0.1)
       y = Bimodal_gaussian(x, 0.5, 5, -60, 10, -30) + 0.03 * np.random.randn(len(x))

       # fitting : befor ap5
       param_b, cov_b = curve_fit(Bimodal_gaussian, voltage, before , p0=[0.5, 1, -65, 1, -20])
       # interpolation
       start = -70
       end = 0
       # bimodal
       sp = scipy.interpolate.InterpolatedUnivariateSpline(voltage, Bimodal_gaussian(voltage, param_b[0], param_b[1], param_b[2], param_b[3], param_b[4]))
       sx = np.linspace(start, end, 1000)
       sy = sp(sx)
       ax0.plot(sx, sy, linestyle="dashed", color= line_color_b, lw=plot_line_w)
       # gauss 1
       sp = scipy.interpolate.InterpolatedUnivariateSpline(voltage, Gaussian(voltage, 1-param_b[0], param_b[3], param_b[4]))
       sx = np.linspace(start, end, 1000)
       sy = sp(sx)
       ax0.plot(sx, sy, color= line_color_b, lw=plot_line_w)

       # gauss 2
       sp = scipy.interpolate.InterpolatedUnivariateSpline(voltage, Gaussian(voltage, param_b[0], param_b[1], param_b[2]))
       sx = np.linspace(start, end, 1000)
       sy = sp(sx)
       ax0.plot(sx, sy, color= line_color_b, lw=plot_line_w)

       # fitting : after ap5
       param_a, cov_a = curve_fit(Bimodal_gaussian, voltage, after , p0=[0.5, 1, -65, 1, -20])
       # bimodal
       sp = scipy.interpolate.InterpolatedUnivariateSpline(voltage, Bimodal_gaussian(voltage, param_a[0], param_a[1], param_a[2], param_a[3], param_a[4]))
       sx = np.linspace(start, end, 1000)
       sy = sp(sx)
       ax0.plot(sx, sy, linestyle="dashed", color= line_color_a, lw=plot_line_w)
       # gauss 1
       sp = scipy.interpolate.InterpolatedUnivariateSpline(voltage, Gaussian(voltage, 1-param_a[0], param_a[3], param_a[4]))
       sx = np.linspace(start, end, 1000)
       sy = sp(sx)
       ax0.plot(sx, sy, color= line_color_a, lw=plot_line_w)

       # gauss 2
       sp = scipy.interpolate.InterpolatedUnivariateSpline(voltage, Gaussian(voltage, param_a[0], param_a[1], param_a[2]))
       sx = np.linspace(start, end, 1000)
       sy = sp(sx)
       ax0.plot(sx, sy, color= line_color_a, lw=plot_line_w)

       # parameter_setting
       ax0.tick_params(labelsize=fsize, axis="x", colors=label_color)
       ax0.tick_params(labelsize=fsize, axis="y", colors=label_color)

       #ax0.spines["right"].set_color("none")
       #ax0.spines["top"].set_color("none")
       ax0.spines["left"].set_linewidth(ax_line_w)
       ax0.spines["bottom"].set_linewidth(ax_line_w)
       ax0.spines["top"].set_linewidth(ax_line_w)
       ax0.spines["right"].set_linewidth(ax_line_w)
       fig.tight_layout()

       plt.savefig(path + fig_name)

if __name__ == '__main__':
     main()