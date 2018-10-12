import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import glob
import itertools

read_path = "//192.168.13.10/Public/hattori/simulation/HH/raw_data/2018_10_10_9_46_37/Mg_2.2/heatmap.csv"

df = pd.read_csv(read_path)
df = df.round(2)
list = df.as_matrix()
list = np.delete(list, 0, 1)
list_f = np.array(list, dtype=float)



fig = plt.figure(figsize=(20, 15))
ax = fig.add_subplot(1, 1, 1)
sns.heatmap(list_f, cmap="BuPu_r", ax=ax)
plt.show()