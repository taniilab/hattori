from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def Bimodal_gaussian(x, A, sigma1, mean1, sigma2, mean2):
    return (A*np.exp(-(x-mean1)**2/(2*sigma1))/np.sqrt(2*np.pi*sigma1) + \
            (1-A)*np.exp(-(x-mean2)**2/(2*sigma2))/np.sqrt(2*np.pi*sigma2))

def Gaussian(x, A, sigma1, mean1):
    return A*np.exp(-(x-mean1)**2/(2*sigma1))/np.sqrt(2*np.pi*sigma1)


path = "//192.168.13.10/Public/experimental data/tohoku_univ/tohoku_patch/20181018_cortex/voltage/ap5.csv"
x = np.arange(-70, 0, 0.1)
y = Bimodal_gaussian(x, 0.5, 5, -60, 10, -30) + 0.03*np.random.randn(len(x))
df = pd.read_csv(path)

x1 = df["V [mV]"]
y1 = df["before"]
y2 = df["after"]
param, cov = curve_fit(Bimodal_gaussian, x1, y2, p0=[0.5, 1, -65, 1, -20])

print(df)
print(param)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(x1, y2)
ax.plot(x1, Bimodal_gaussian(x1, param[0], param[1], param[2], param[3], param[4]))
ax.plot(x1, Gaussian(x1, param[0], param[1], param[2]), '.')
ax.plot(x1, Gaussian(x1, 1-param[0], param[3], param[4]), '.')
plt.show()
