import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import itertools

#nest
read_path = "//192.168.13.10/Public/hattori/simulation/HH/raw_data/2018_10_10_9_46_37(maindata)/Mg_1.0/raw_data/2018_12_7_14_13_43/" + \
            "2018_10_9_21_18_47_N0_P_AMPA0.6_P_NMDA0.4_Mg_conc1.0_delay0HH.csv"
"""
#AMPA
read_path = "//192.168.13.10/Public/hattori/simulation/HH/raw_data/2018_10_10_9_46_37(maindata)/Mg_1.0/raw_data/2018_12_7_14_13_43/" + \
            "2018_10_9_21_15_46_N0_P_AMPA0.6_P_NMDA0.0_Mg_conc1.0_delay0HH.csv"
"""

#NMDA
read_path = "//192.168.13.10/Public/hattori/simulation/HH/raw_data/2018_10_10_9_46_37(maindata)/Mg_1.0/raw_data/2018_12_7_14_13_43/" + \
"2018_10_9_20_2_45_N0_P_AMPA0.0_P_NMDA0.4_Mg_conc1.0_delay0HH.csv"

save_path = "//192.168.13.10/Public/hattori/"

df = pd.read_csv(read_path, index_col=0, skiprows=1)

voltage = df['V [mV]']
time = df['T [ms]']
tmp = 0 * voltage
dt = 0.04
buf = 0
t_ap = []
period = []
freq = []
line_w = 1

for l in range(0, len(df['T [ms]'])):
    if voltage[l] > 0 and buf == 0:
        t_ap.append(time[l])
        tmp[l] = 1
        buf = int(2/dt)
    if buf > 0:
        buf -= 1

for i in range(0, len(t_ap)-1):
    period.append(t_ap[i+1]-t_ap[i])
    freq.append(1000/(t_ap[i+1]-t_ap[i]))

print(t_ap)
print(period)
print(freq)

"""
print(np.var(freq))
freq = np.delete(freq, range(0,100))
print(freq)
print(np.var(freq))
"""
fig = plt.figure(figsize=(10,10))
ax0 = fig.add_subplot(2, 2, 1)
ax0.plot(df['T [ms]'], df['V [mV]'],
         color="black",
         linewidth=line_w, markevery=[0, -1])
ax1 = fig.add_subplot(2, 2, 2)
ax1.plot(df['T [ms]'], tmp,
         color="black",
         linewidth=line_w, markevery=[0, -1])
ax2 = fig.add_subplot(2, 2, 3)
ax2.plot(period,
         color="black",
         linewidth=line_w, markevery=[0, -1])
ax3 = fig.add_subplot(2, 2, 4)
ax3.plot(freq,
         color="black",
         linewidth=line_w, markevery=[0, -1])

df = pd.DataFrame({'freq': freq})
#df.to_csv(save_path + "nmda.csv", mode='a')

plt.show()
