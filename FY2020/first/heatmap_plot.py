import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import glob
import itertools

#read_path = "//192.168.13.10/Public/hattori/simulation/heatmap/Mg_conc2.2/heat_ave.csv"
#read_path = "//192.168.13.10/Public/ishida/simulation/HH_hattori/raw_data/2018_10_11_14_0_47/Mg_conc2.2/heatmap.csv"
Mg = "04"
work_dir = "C:/sim/Mg" + str(Mg) + "/"
read_path = work_dir + "heatmap.csv"
save_path = work_dir + "/Mg" + str(Mg) +  ".jpg"

target_dpi = 600
config_dpi = 600
ratio = target_dpi / config_dpi
"""inch"""
graph_width = 15
graph_hight = 15
fsize = 8

df = pd.read_csv(read_path)
df = df.round(2)
list = df.values
print(list)
list = np.delete(list, 0, 1)
print(list)
#list = np.delete(list, [16,17,18,19], 0)
#list = np.delete(list, [16,17,18,19], 1)
list_f = np.array(list, dtype=float)
print(list_f)

fig = plt.figure(figsize=(graph_width * ratio, graph_hight * ratio), dpi=config_dpi)
ax = fig.add_subplot(1, 1, 1)

sns.set(font_scale=1)
ax.tick_params(labelsize=fsize)
sns.heatmap(list_f, xticklabels=5,
            yticklabels=5,
            cmap="inferno", ax=ax, square=True)

#ax.xaxis.tick_top()
ax.invert_yaxis()
ax.tick_params(labelbottom="off",bottom="off") # x軸の削除
ax.tick_params(labelleft="off",left="off") # y軸の削除

plt.savefig(save_path)
