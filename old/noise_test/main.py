# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 08:36:53 2017
ホワイトノイズfft→スペクトルをカラードノイズの形状に変換→逆fft

@author: 6969p
"""

import numpy as np
from numpy.random import *
import matplotlib.pyplot as plt
import math
import scipy.fftpack as scft
from scipy import hamming
 
def white_noize(length, sample_rate=44100):
    sample = length * sample_rate
    time_step = 1. / sample_rate
    time_arr = np.arange(sample) * time_step
    noize = np.random.randn(time_arr.size)
    return noize

if __name__ == "__main__":
    sample_rate = 44100
    length = 10
    noize = white_noize(length, sample_rate)
    print(noize)    

    win = hamming(length * sample_rate)

    fft_noize = scft.fft(noize * win)
    spectrum_noize = [np.sqrt(data.real ** 2 + data.imag ** 2) for data in fft_noize]
  
    fig, ax = plt.subplots(nrows = 3, figsize = (12, 18))
    ax[0].plot(range(0, sample_rate//2), spectrum_noize[0: sample_rate//2])
    ax[0].set_ylim(min(spectrum_noize), max(spectrum_noize))
    ax[0].set_title("Power Spectrum of White Noize")

    for i in range(0, len(fft_noize)):
        fft_noize[i] *= np.exp(-i/22050)    
    spectrum_noize = [np.sqrt(data.real ** 2 + data.imag ** 2) for data in fft_noize]
    print(fft_noize)

    ax[1].plot(range(0, sample_rate//2), spectrum_noize[0: sample_rate//2])
    ax[1].set_ylim(min(spectrum_noize), max(spectrum_noize))
    ax[1].set_title("Power Spectrum of Pink Noize")
        
    
    ifft = scft.ifft(fft_noize)
    ifft /= win

    fft_noize = scft.fft(ifft * win)
    spectrum_noize = [np.sqrt(data.real ** 2 + data.imag ** 2) for data in fft_noize]
    
    ax[2].plot(range(0, sample_rate//2), spectrum_noize[0: sample_rate//2])
    ax[2].set_ylim(min(spectrum_noize), max(spectrum_noize))
    ax[2].set_title("Power Spectrum of Pink2 Noize")

