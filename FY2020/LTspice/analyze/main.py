import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

path = "C:/Users/Kouhei/Documents/GitHub/hattori/Stim_simulation/LTspice/"
filename = "re電圧刺激等価回路_20210128_paramstep_r5_r6_50_3000_118_cathode.csv"
range_param = 26
num_param = range_param * range_param
df = pd.read_csv(path+filename, skiprows=1, names=["A", "B"])
print(df)
sp = df.query('A.str.startswith("Step")', engine='python').reset_index()
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
print(peak_list.shape)
peak_list = peak_list.reshape([range_param, range_param])
print(peak_list)
print(peak_list.shape)
plt.figure(figsize=(25, 20))
sb.heatmap(peak_list, square=True, cmap='gray')
plt.show()