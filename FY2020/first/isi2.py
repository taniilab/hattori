import matplotlib.pyplot as plt
import glob
import os
import shutil
import pandas as pd
import numpy as np
import datetime
import matplotlib as mpl
from multiprocessing import Pool

path = "C:/Users/Kouhei/Desktop/test/"
title = path + "*.csv"
conc_data = 0
now_spike = 0
pre_spike = 0
dt = 0.02
counter = 0
isi_list = []
csv = glob.glob(title)
print(csv)

for i in range(len(csv)):
    df = pd.read_csv(csv[i], skiprows=1)
    df = df.values
    fire = df[:, 3]
    for j in range(len(fire)):
        if fire[j] > 0:
            pre_spike = now_spike
            now_spike = j + counter*len(fire)
            isi_list.append((now_spike - pre_spike) * dt)
    counter += 1
    print("kanopero: {}".format(i))

fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

isi_array = np.array(isi_list)
isi_array = isi_array[isi_array < 500]
#isi_array = isi_array[10 < isi_array]
print(isi_array)

ax.plot(isi_array)
ax2.hist(isi_array, bins = 30)


df = pd.DataFrame()
df['isi_arrays'] = isi_array
df.to_csv(path + 'for_hist.csv', mode='a')

plt.show()
