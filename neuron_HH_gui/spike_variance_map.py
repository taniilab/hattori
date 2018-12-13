import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import glob
import itertools


def index_initialize():
    i = 18
    j = 16

    return i, j

png_path = "//192.168.13.10/Public/hattori/" + \
       "seaborn_heatmap_list2.png"

#read_path = "Z:/Box Sync/Personal/tmp_data/"

"""
fig = plt.figure(figsize= (20,15))
ax = fig.add_subplot(1,1,1)
#sns.heatmap(list_duration_time, vmax=6, vmin=0, cmap="BuPu_r", ax=ax)
sns.heatmap(dummy_data, cmap="BuPu_r", ax=ax)
plt.show()
"""
read_path = "//192.168.13.10/Public/hattori/simulation/HH/raw_data/2018_10_10_9_46_37(maindata)/Mg_1.0/raw_data/2018_12_7_14_13_43/"


nowdir = read_path
i, j = index_initialize()
list_duration_time = np.zeros((i,j))
list_columns = []
list_rows = []
freq = []
list_freq_variability = np.zeros((i,j))
dt = 0.04
buf = 0

#2018_10_10_15_50_24_N0_P_AMPA0.0_P_NMDA0.0_Mg_conc0.1_delay0HH
for i, j in itertools.product(range(i), range(j)):
    tmp = read_path + "*_P_AMPA" + str(round(i*0.1, 1)) + "_P_NMDA" + str(round(j*0.1, 1)) + "*.csv"
    csv = glob.glob(tmp)
    print(tmp)
    print(csv)
    print(len(csv))

    # averaging
    for k in range(len(csv)):
        df = pd.read_csv(csv[k], index_col=0, skiprows=1)
        t_ap = []
        voltage = df['V [mV]']
        time = df['T [ms]']
        dt2 = 0.04

        for l in range(0, len(df['T [ms]'])):
            if voltage[l] > 0 and buf == 0:
                t_ap.append(time[l])
                buf = int(2 / dt)
            if buf > 0:
                buf -= 1

        # duration time of spontaneouos activity
        list_duration_time[i,j] += t_ap[-1]-t_ap[0]

    for k in range(0, len(t_ap) - 1):
        freq.append(1000 / (t_ap[k + 1] - t_ap[k]))

    if len(freq) != 0 and len(freq) != 1:
        print(np.var(freq))
        list_freq_variability[i, j] = np.var(freq)
    else:
        list_freq_variability[i, j] = 0

i, j = index_initialize()
for i in range(i):
    list_rows.append("AMPA: "+str(round(i*0.1, 1)))
for j in range(j):
    list_columns.append("NMDA: "+str(round(j*0.1, 1)))

fig = plt.figure(figsize=(20, 15))
ax = fig.add_subplot(1, 1, 1)
sns.heatmap(list_freq_variability, cmap="BuPu_r", ax=ax)
plt.show()
