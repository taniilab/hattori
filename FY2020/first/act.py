import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
#20khz
df = pd.read_csv("C:/Users/Kouhei/Downloads/voltage2.csv")
df2 = pd.read_csv("C:/Users/Kouhei/Downloads/voltage5.csv")
compress = 100

print(df)
df = df.values
mem = df[::compress]
print(mem)
df2 = df2.values
mem2 = df2[::compress]

spike_train = []
spike_train2 = []

for i in range(len(mem)):
    if mem[i, 2] > -35:
        spike_train.append(1)
    else:
        spike_train.append(0)

for i in range(len(mem2)):
    if mem2[i, 2] > -35:
        spike_train2.append(1)
    else:
        spike_train2.append(0)

acf = np.correlate(spike_train, spike_train, mode='full')
acf2 = np.correlate(spike_train2, spike_train2, mode='full')
acf = acf / np.max(acf)
acf2 = acf2 / np.max(acf2)
lag = np.arange(len(acf))
lag = (lag - len(lag)/2) * 0.005 * compress
print(lag)
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(211)
ax.plot(lag, acf, label="bef", color="red")
ax.plot(lag, acf2, label="aft", color="blue")
ax2 = fig.add_subplot(212)
ax2.plot(spike_train)
plt.tight_layout()
plt.legend()
plt.show()