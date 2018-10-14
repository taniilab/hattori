import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import glob
import itertools

read_path = "//192.168.13.10/Public/hattori/simulation/heatmap/Mg_conc0.1/heat_ave.csv"
#read_path = "//192.168.13.10/Public/ishida/simulation/HH_hattori/raw_data/2018_10_11_14_0_47/Mg_conc2.2/heatmap.csv"

fsize = 72

df = pd.read_csv(read_path)
df = df.round(2)
list = df.as_matrix()
list = np.delete(list, 0, 1)
list_f = np.array(list, dtype=float)

fig = plt.figure(figsize=(30, 20))
ax = fig.add_subplot(1, 1, 1)

sns.set(font_scale=6)
ax.tick_params(labelsize=fsize)
yticks = np.linspace(0, 17,3)
sns.heatmap(list_f, xticklabels=[0, 5, 10, 15],
            yticklabels=[0,9,17],
            cmap="GnBu", ax=ax)
ax.set_xticks([0.5,5.5,10.5,15.5])
ax.set_yticks([0.5,9.5,17.5])
plt.show()
