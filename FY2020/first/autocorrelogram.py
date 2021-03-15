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

t = 1000 * np.arange(len(mem))/ 200 #ms

for i in range(len(mem)-1):
    if mem[i, 2] <= -35 and mem[i+1, 2] > -35:
        spike_train.append(1)
    else:
        spike_train.append(0)

for i in range(len(mem2)-1):
    if mem2[i, 2] <= -35 and mem2[i+1, 2] > -35:
        spike_train2.append(1)
    else:
        spike_train2.append(0)

print(len(spike_train))
window_size = 60
x = np.arange(0, window_size)
correlograms = np.zeros(window_size)

lw = 5
ls = 20

for i in range(len(spike_train)-window_size):
    for j in range(window_size):
        correlograms[j] += spike_train[i] * spike_train[i+j]

fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(231, xlim=(0, window_size), ylim=(0, 30))
ax_sptr = fig.add_subplot(232)
ax_raw = fig.add_subplot(233)
ax.bar(x, correlograms, width=1, color="black", label="before")
ax_sptr.plot(spike_train)
ax_raw.plot(t, mem[:, 2], marker='.')
ax.spines["top"].set_linewidth(0)
ax.spines["right"].set_linewidth(0)
ax.spines["bottom"].set_linewidth(lw)
ax.spines["left"].set_linewidth(lw)
ax.tick_params(labelsize=ls)

correlograms = np.zeros(window_size)

for i in range(len(spike_train2)-window_size):
    for j in range(window_size):
        correlograms[j] += spike_train2[i] * spike_train2[i+j]

ax2 = fig.add_subplot(234, xlim=(0, window_size), ylim=(0, 30))
ax2_sptr = fig.add_subplot(235)
ax2_raw = fig.add_subplot(236)
ax2.bar(x, correlograms, width=1, color="black", label="after")
ax2_sptr.plot(spike_train2)
ax2_raw.plot(t, mem2[:, 2], marker='.')
ax2.spines["top"].set_linewidth(0)
ax2.spines["right"].set_linewidth(0)
ax2.spines["bottom"].set_linewidth(lw)
ax2.spines["left"].set_linewidth(lw)
ax2.tick_params(labelsize=ls)

plt.legend()
plt.tight_layout()
plt.show()
