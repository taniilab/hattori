import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

path = "//192.168.13.10/Public/hattori/" + \
       "seaborn_heatmap_list2.png"


list_2d = [[0, 1, 2, 3, 4, 5], [2, 3, 4, 5, 5, 5]]


fig = plt.figure(figsize= (20,15))
ax = fig.add_subplot(1,1,1)
sns.heatmap(list_2d, vmax=6, vmin=0, cmap="BuPu_r", ax=ax)
plt.show()
