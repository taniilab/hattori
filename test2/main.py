# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 09:30:11 2017

@author: 6969p
"""
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0,1000,0.01)

y = (np.exp(1-(x/2))*x)/2

fig, ax = plt.subplots(nrows = 3, figsize = (12, 18))
ax[0].plot(x, y)


ndarr1 = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
ndarr2 = np.array([[11., 22., 33.], [44., 55., 66.], [77., 88., 99.]])
ndarr3 = np.array([[111., 222., 333.], [444., 555., 666.], [777., 888., 999.]])


print(ndarr1)
# 出力結果
# [[ 1.  2.  3.]
#  [ 4.  5.  6.]
#  [ 7.  8.  9.]]

p = ndarr1[:, 1]
q = ndarr2[:, 1]
r = ndarr3[:, 1]
print(ndarr1[:, 1])
# 出力結果
# [[ 4.  5.  6.]
#  [ 7.  8.  9.]]
p = ndarr1[:, 1]
p[:] = q * r
print(p)
print(q)
print(r)
print(ndarr1)
# 出力結果
# [[ 4.  5.]
#  [ 7.  8.]]