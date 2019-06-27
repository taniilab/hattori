from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-10, 10, 0.1)

def bimodal(x, sigma1, mean1, sigma2, mean2):
    return np.exp(-(x-mean1)**2/(2*sigma1))/np.sqrt(2*np.pi*sigma1) + \
           np.exp(-(x-mean2)**2/(2*sigma2))/np.sqrt(2*np.pi*sigma2)

y = bimodal(x, 2, -3, 0.5, 3) + 0.01*np.random.randn(len(x))

param, cov = curve_fit(bimodal, x, y)

print(param)
print("\n")
print(cov)
print(param[0])

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(x, y)
ax.plot(x,bimodal(x, param[0], param[1], param[2], param[3]))
plt.show()
