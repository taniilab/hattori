import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

path = "C:/Users/Kouhei/Documents/GitHub/hattori/Stim_simulation/LTspice/20210210/"
name = "2細胞距離依存性.csv"

fig = plt.figure(figsize=(12,15))
ax1 = fig.add_subplot(111)
df = pd.read_csv(path+name)
print(df)
lw = 10
fontsize = 72
ax1.plot(df.iloc[:, 0:1], df.iloc[:, 1:2], color="black", linestyle="dashed",
         linewidth=lw)
ax1.plot(df.iloc[:, 0:1], df.iloc[:, 2:3], color="black", linestyle="dotted",
         linewidth=lw)
plt.tight_layout()

ax2 = ax1.twinx()
ax2.plot(df.iloc[:, 0:1], df.iloc[:, 3:4], color="black", linestyle="solid",linewidth=lw)
plt.setp(ax1.get_xticklabels(), fontsize=fontsize)
plt.setp(ax1.get_yticklabels(), fontsize=fontsize)
plt.setp(ax2.get_yticklabels(), fontsize=fontsize)
plt.tight_layout()

splw = 5
ax1.spines["top"].set_linewidth(0)
ax2.spines["top"].set_linewidth(0)
ax1.spines["bottom"].set_linewidth(splw)
ax2.spines["bottom"].set_linewidth(splw)
ax1.spines["left"].set_linewidth(splw)
ax2.spines["left"].set_linewidth(splw)
ax1.spines["right"].set_linewidth(splw)
ax2.spines["right"].set_linewidth(splw)
plt.tight_layout()

plt.show()