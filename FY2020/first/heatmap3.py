import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import glob
import itertools


num_ampa = 16
num_nmda = 16

png_path = "//192.168.13.10/Public/hattori/" + \
       "seaborn_heatmap_list2.png"

"""
fig = plt.figure(figsize= (20,15))
ax = fig.add_subplot(1,1,1)
#sns.heatmap(list_duration_time, vmax=6, vmin=0, cmap="BuPu_r", ax=ax)
sns.heatmap(dummy_data, cmap="BuPu_r", ax=ax)
plt.show()
"""
#read_path = "//192.168.13.10/Public/hattori/simulation/HH/raw_data/2018_10_11_14_0_47/Mg_conc0.1"
read_path = "C:/sim/Mg16/"

nowdir = read_path
list_duration_time = np.zeros((num_ampa, num_nmda))
list_columns = []
list_rows = []

for i, j in itertools.product(range(num_ampa), range(num_nmda)):
    tmp = read_path + "*_P_AMPA" + str(round(i*0.1, 1)) + "_P_NMDA" + str(round(j*0.1, 1)) + "*.csv"
    csv = glob.glob(tmp)
    print(tmp)
    print(csv)
    print(len(csv))

    # averaging
    for k in range(len(csv)):
        df = pd.read_csv(csv[k], index_col=0, skiprows=1)
        t_ap = []
        voltage = df['V_0 [mV]']
        time = df['T_0 [ms]']
        dt2 = 0.02

        for l in range(0, len(time)):
            if voltage[l] > 10:
                t_ap.append(time[l])

        # duration time of spontaneouos activity
        list_duration_time[i, j] += t_ap[-1]-t_ap[0]

    list_duration_time[i, j] /= len(csv)


for i in range(num_ampa):
    list_rows.append("AMPA: "+str(round(i*0.1, 1)))
for j in range(num_nmda):
    list_columns.append("NMDA: "+str(round(j*0.1, 1)))
print(list_rows)
print(list_columns)
df = pd.DataFrame(list_duration_time, columns=list_columns, index=list_rows)
df.to_csv(read_path+"heatmap.csv")
fig = plt.figure(figsize=(20, 15))
ax = fig.add_subplot(1, 1, 1)
sns.heatmap(list_duration_time, cmap="inferno", ax=ax)
plt.show()
