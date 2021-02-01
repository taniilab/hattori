import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

path = "C:/Users/Kouhei/Documents/GitHub/hattori/Stim_simulation/LTspice/"
filename = "re電圧刺激等価回路_20210128_selectivity_2needle.csv"
num_param = 11
df = pd.read_csv(path+filename, skiprows=1, names=["A", "B"])
sp = df.query('A.str.startswith("Step")', engine='python').reset_index()
print(sp)
index = sp['index']
data = []
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(1,1,1)
for i in range(num_param):
    if i == num_param-1:
        df_tmp = pd.DataFrame(df.iloc[index[i]:, 0:2])
        df_tmp.to_csv(path + str(i) + ".csv")
    else:
        df_tmp = pd.DataFrame(df.iloc[index[i]:index[i + 1], 0:2])
        df_tmp.to_csv(path + str(i) + ".csv")

for i in range(num_param):
    df_r = pd.read_csv(path+str(i)+".csv",skiprows=1)
    print(df_r)
    ax.plot(1000*df_r.iloc[:, 1], 1000*df_r.iloc[:, 2], color='black', lw=4)

    ax_lw = 3
    tick_fsize = 50
    ax.spines["right"].set_linewidth(0)
    ax.spines["top"].set_linewidth(0)
    ax.spines["bottom"].set_linewidth(ax_lw)
    ax.spines["left"].set_linewidth(ax_lw)
    ax.tick_params(labelsize=tick_fsize, colors="black")

plt.show()