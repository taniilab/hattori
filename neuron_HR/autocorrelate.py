# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 14:59:29 2017

@author: Hattori
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal as sig
import scipy.fftpack as fft
import glob

"""
dataset = pd.read_csv('./results/2017_6_19_12_38_7_5HR_model.csv',
                      index_col=0)
"""
"""
memo
2017_6_20_12_20_48_5HR_model.csv
"""

allfiles = glob.glob('C:/Users/Hattori/Documents/HR_outputs/results/pippi14/*.csv')
list = []
list3 = []
cor2 = []
fft_res = []
amp_spec = []
frame = pd.DataFrame()
for file_ in allfiles:
    df = pd.read_csv(file_, index_col=0)
    list.append(df.as_matrix())

"""
for i in range(0, len(list)):
    # autocorrelate
    dtt = list[i]
    dt = dtt[:, 5]
    list3.append(dt)
    cor = sig.correlate(list3[i], list3[i], mode="full")
    cor2.append(cor[int(cor.size/2):])
    # fft
    start = 0
    N = 2500
    fs = 100000000

    fft_res.append(fft.fft(list[i][start:start+N, 5]))
    freqlist = fft.fftfreq(N, d=1.0/fs)
    amp_spec.append([np.sqrt(c.real ** 2 + c.imag ** 2) for c in fft_res[i]])


fig, ax = plt.subplots(nrows=len(list)*2, figsize=(5, 50))
fig.tight_layout()
fig.subplots_adjust(left=0.05, bottom=0.03)
for i in range(0, len(list)):
    p = list[i]
    ax[i*2].plot(p[:, 4], cor2[i])
    ax[i*2+1].plot(freqlist, amp_spec[i])
"""

# wave plot
fig, ax = plt.subplots(nrows=len(list), figsize=(5, 50))
fig.tight_layout()
for i in range(0, len(list)):
    p = list[i]
    ax[i].plot(p[:, 4], p[:, 5])
    ax[i].set_xlabel('D = ' + str(p[1, 0]) + 'alpha = '+str(p[1, 2]) +
                     'beta = ' + str(p[1, 3]))
