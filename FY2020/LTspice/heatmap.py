import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import glob
import itertools


read_path = "C:/Users/Kouhei/Desktop/test.csv"
save_path = "C:/Users/Kouhei/Desktop/test.jpg"

target_dpi = 600
config_dpi = 600
ratio = target_dpi / config_dpi
"""inch"""
graph_width = 15
graph_hight = 15
fsize = 40

df = pd.read_csv(read_path)
list = df.iloc[:, 1:].values
print(list)

fig = plt.figure(figsize=(graph_width * ratio, graph_hight * ratio), dpi=config_dpi)
ax = fig.add_subplot(1, 1, 1)

sns.set(font_scale=6)
ax.tick_params(labelsize=fsize, color="white")

"""
sns.heatmap(list, xticklabels=5,
            yticklabels=5,
            cmap="inferno", ax=ax, square=True)
"""
sns.heatmap(list,cmap="inferno", square=True)

#ax.xaxis.tick_top()
ax.invert_yaxis()
ax.tick_params(labelbottom="off",bottom="off") # x軸の削除
ax.tick_params(labelleft="off",left="off") # y軸の削除
plt.tight_layout()
plt.savefig(save_path)
