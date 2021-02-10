import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

path = "C:/Users/Kouhei/Documents/GitHub/hattori/Stim_simulation/LTspice/20210210/"
filename = "re電圧刺激等価回路_20210210_2ニューロン位置依存性.csv"
#filename = "re電圧刺激等価回路_20210128_paramstep_r5_r6_50_3000_118_cathode.csv"
num_param = 10
df = pd.read_csv(path+filename, skiprows=1, names=["time", " V(N012", "N005)", " V(N013", "N007)", " V(n001)", " V(n002)", " V(n003)", " V(n004)", " V(n005)", " V(n006)", " V(n007)", " V(n008)", " V(n009)", " V(n010)", " V(n011)", " V(n012)", " V(n013)", " V(n014)", " V(n015)", " V(nc_01)", " V(nc_02)", " V(nc_03)", " V(nc_04)", " V(nc_05)", " V(nc_06)", " V(nc_07)", " V(nc_08)", " V(nc_09)", " V(nc_10)", " V(nc_11)", " V(nc_12)", " V(nc_13)", " V(nc_14)", " I(C1)", " I(C2)", " I(C3)", " I(C4)", " I(C5)", " I(C6)", " I(R1)", " I(R2)", " I(R3)", " I(R4)", " I(R5)", " I(R6)", " I(R7)", " I(R8)", " I(R9)", " I(R10)", " I(R11)", " I(R12)", " I(R13)", " I(R14)", " I(V1)", " I(V2)", " I(V3)", " I(V4)", " I(V5)", " I(V6)", " I(V7)", " I(V8)", " I(V9)", " I(V10)", " I(V11)", " I(V12)"
])
print(df)
sp = df.query('time.str.startswith("Step")', engine='python').reset_index()
print(sp)
index = sp['index']

peak_list = np.zeros(num_param)

for i in range(num_param):
    if i == num_param-1:
        value = 1000 * float(df.iloc[index[i]:, 1:2].max().values)
        peak_list[i] = value
    else:
        value = 1000 * float(df.iloc[index[i]:index[i + 1], 1:2].max().values)
        peak_list[i] = value
print(peak_list)
plt.figure(figsize=(25, 20))
plt.plot(peak_list)
plt.show()

plt.figure()
print()