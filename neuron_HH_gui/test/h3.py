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
#read_path = "//192.168.13.10/Public/hattori/simulation/HH/raw_data/2018_10_11_14_0_47/Mg_conc0.1"
read_path = "//192.168.13.10/Public/ishida/simulation/HH_hattori/raw_data/2018_10_11_14_0_47/Mg_conc1.0/"

nowdir = read_path
i, j = index_initialize()
list_duration_time = np.zeros((i,j))
list_columns = []
list_rows = []

for i, j in itertools.product(range(i), range(j)):
    tmp = read_path + "*_P_AMPA" + str(round(i*0.1, 1)) + "_P_NMDA" + str(round(j*0.1, 1)) + "*.csv"
    csv = glob.glob(tmp)
    print(csv)
    print(len(csv))

    for k in range(len(csv)):
        df = pd.read_csv(csv[k], index_col=0, skiprows=1)
        t_ap = []
        voltage = df['V [mV]']
        time = df['T [ms]']
        dt2 = 0.04

        for l in range(0, len(df['T [ms]'])):
            if voltage[l] > 10:
                t_ap.append(time[l])

        # duration time of spontaneouos activity
        list_duration_time[i,j] += t_ap[-1]-t_ap[0]

    list_duration_time[i, j] /= len(csv)


i, j = index_initialize()
for i in range(i):
    list_rows.append("AMPA: "+str(round(i*0.1, 1)))
for j in range(j):
    list_columns.append("NMDA: "+str(round(j*0.1, 1)))
print(list_rows)
print(list_columns)
df = pd.DataFrame(list_duration_time, columns=list_columns, index=list_rows)
df.to_csv(read_path+"heatmap.csv")
fig = plt.figure(figsize=(20, 15))
ax = fig.add_subplot(1, 1, 1)
sns.heatmap(list_duration_time, cmap="BuPu_r", ax=ax)
plt.show()
