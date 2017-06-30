# -*- coding: utf-8 -*-
"""
Created on ???

arrange images

@author: Hattori
"""
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import glob
import itertools

path = 'C:/Users/Hattori/Documents/HR_results/photo/'

x_axis = 'beta'
num_x = 10
y_axis = 'tausyn'
num_y = 10
z_axis = 'Pmax'
num_z = 10
horpx = 2100
verpx = 1400


def tiling(files, k):
    canvas = Image.new('RGB', (horpx*num_x, verpx*num_y), (255, 255, 255))

    filename = 'beta_vs_tausyn_alpha_0.9_Pmax_' + str(round(0.4*k, 1))

    for i, j in itertools.product(range(0, 10), range(0, 10)):
        pic_list[i].append(Image.open(files[j+i*10], 'r'))
        canvas.paste(pic_list[i][j], (horpx*i, verpx*j))

    canvas.save(path + '/tile/' + str(filename) + '.jpg', 'JPEG', quality=90,
                optimize=True)


for i in range(0, 6):
    # initialize
    files = []
    pic_list = []
    for j in range(0, num_y):
        pic_list.append([])

    # open
    files = glob.glob(path + 'alpha_0.9_beta_0.?_tausyn_?.?_Pmax_' +
                      str(round(0.4*i, 1)) + '.png')
    tiling(files, i)
    print(str(i) + 'steps')

im = Image.open("kanon.jpg", "r")
plt.imshow(np.array(im))
print("ちょう終わりました～♪")
