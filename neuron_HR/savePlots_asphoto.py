# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 14:54:07 2017

@author: Hattori
"""
import matplotlib.pyplot as plt
import glob
import os
import pandas as pd
import itertools
import numpy as np
from PIL import Image

folder = 'tmp2'
files = glob.glob('C:/Users/Hattori/Documents/HR_results/' + folder + '/*.csv')
path = 'C:/Users/Hattori/Documents/HR_results/' + folder + '/photo/'
# files = glob.glob('C:/Users/Hattori/Documents/xx/*.csv')
# path = 'C:/Users/Hattori/Documents/xx/photo/'

im_ju = Image.open('junon.jpg', 'r')
im_pi = Image.open('pinon.jpg', 'r')
im_ka = Image.open('kanon.jpg', 'r')

line = 1
column = len(files)
counter = 0
file_sets = [[0 for i in range(line)] for j in range(column)]

# in order to avoid out of mempry,  files are processed every 'line' files.
for i, j in itertools.product(range(column), range(line)):
    file_sets[i][j] = files[counter]
    if counter != line*column:
        counter += 1

# main process
counter = 0
for i in range(column):
    list = []
    filenames = []
    # plt.imshow(np.array(im_pi))
    for file_ in file_sets[i]:
        df = pd.read_csv(file_, index_col=0)
        list.append(df.as_matrix())
        filenames.append(os.path.basename(file_).replace('.csv', ''))

        print(str(len(list) + counter) + '個目のファイル読み込むぴっぴ！')

    # plt.imshow(np.array(im_ju))
    for i in range(len(list)):
        hr = list[i]
        plt.title(filenames[i])
        plt.plot(hr[:, 6], hr[:, 8])
        plt.savefig(filename=path + filenames[i] + '.jpg', dpi=350)
        plt.clf()
        print(str(i + counter) + '個目のファイルを処理')
    counter += line

plt.figure(figsize=(3, 2))
plt.imshow(np.array(im_ka))
plt.close()
print('ちょう処理終わりました♪')
