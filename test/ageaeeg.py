from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def bimodal(x, A1, sigma1, mean1, A2, sigma2, mean2):
    return A1*np.exp(-(x-mean1)**2/(2*sigma1))/np.sqrt(2*np.pi*sigma1) + \
           A2*np.exp(-(x-mean2)**2/(2*sigma2))/np.sqrt(2*np.pi*sigma2)

path = "//192.168.13.10/Public/experimental data/tohoku_univ/tohoku_patch/20181018_cortex/voltage/ap5.csv"
x = np.arange(-10, 10, 0.1)
df = pd.read_csv(path)

x1 = df["V [mV]"]
y1 = df["before"]
y2 = df["after"]
param, cov = curve_fit(bimodal, x1, y2)
  
print(df)
print(param)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(x1, y2)
ax.plot(x1, bimodal(x1, param[0], param[1], param[2], param[3], param[4], param[5]))
plt.show()
