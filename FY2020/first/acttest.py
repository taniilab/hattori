import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot

x = np.array([0,0,1,1,0,0,1,1,0,0])
acf = np.correlate(x, x, mode='full')
acf = acf / np.max(acf)
print(acf)
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111)
ax.plot(acf)
plt.show()