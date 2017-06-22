# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 14:54:07 2017

@author: Hattori
"""
import matplotlib.pyplot as plt
import glob
import pandas as pd
import shutil

allfiles = glob.glob('C:/Users/Hattori/Documents/HR_results/*.csv')
path = 'C:/Users/Hattori/Documents/HR_results/photo/'

list = []
for file_ in allfiles:
    df = pd.read_csv(file_, index_col=0)
    list.append(df.as_matrix())
    print(str(len(list)) + '個目のファイル読み込むぴっぴ！')

for i in range(len(list)):
    hr = list[i]
    title = ('alpha_' + str(hr[1, 3]) + '_beta_' + str(hr[1, 4]) +
             '_tausyn_' + str(hr[1, 6]) + '_Pmax_' + str(hr[1, 2]))
    plt.title(title)
    plt.plot(hr[:, 5], hr[:, 7])
    plt.savefig(filename=path + title + '.png', dpi=350)
    plt.clf()
    print(str(i) + '個目のファイルをちょう絶クールにpng化')

plt.close()
print('ちょう処理終わりました♪')
