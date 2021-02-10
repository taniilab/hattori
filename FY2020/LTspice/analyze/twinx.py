import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

path = "C:/Users/Kouhei/Documents/GitHub/hattori/Stim_simulation/LTspice/20210210/"
name = "2細胞距離依存性.csv"

fig = plt.figure(figsize=(20,15))
ax1 = fig.add_subplot(111)
df = pd.read_csv(path+name)
print(df)
lw = 10
ax1.plot(df.iloc[:, 0:1], df.iloc[:, 1:2], color="black", linestyle="dashed",
         linewidth=lw)
ax1.plot(df.iloc[:, 0:1], df.iloc[:, 2:3], color="black", linestyle="dotted",
         linewidth=lw)

ax2 = ax1.twinx()
ax2.plot(df.iloc[:, 0:1], df.iloc[:, 3:4], color="black", linestyle="solid",
         linewidth=lw)

plt.show()