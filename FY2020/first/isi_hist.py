import matplotlib.pyplot as plt
import glob
import os
import shutil
import pandas as pd
import numpy as np
import datetime
import matplotlib as mpl
from multiprocessing import Pool
from scipy.stats import norm
import math

path = "C:/Users/Kouhei/Desktop/isi/"
title = path + "for_hist.csv"

df = pd.read_csv(title, skiprows=0)
isi_array = df.values

fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(111)

isi_array = isi_array[isi_array < 200]
isi_array = isi_array[50 < isi_array]
print(isi_array)

x = np.linspace(50,500,10000)
p = []
param = norm.fit(isi_array)
print(param)
x = np.linspace(50,200,150)
pdf_fitted = norm.pdf(x, loc=param[0], scale=param[1])
pdf = norm.pdf(x)
ax.plot(x, pdf_fitted, 'r-', x,pdf, 'b-')
ax.hist(isi_array, bins=100, normed=True)
plt.show()
