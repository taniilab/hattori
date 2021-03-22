import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot

spike_train = np.array([0,0,1,1,0,0,1,1,1,0,1,1,0,0,1,1,0,0,1,0,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1])

#sin wave example
x = np.arange(0,1000,0.1)
x += 0.3*np.random.randn(len(x))
data = np.sin(x)
spike_train = []
for i in range(len(data)):
    if data[i] > 0.5:
        spike_train.append(1)
    else:
        spike_train.append(0)

window_size = 1000
x = np.arange(0, window_size)
correlograms = np.zeros(window_size)

for i in range(len(spike_train)-window_size):
    for j in range(window_size):
        correlograms[j] += spike_train[i] * spike_train[i+j]

print(correlograms)
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111)
ax.bar(x,correlograms, width=1, color="black")
plt.tight_layout()
plt.show()