# -*- coding: utf-8 -*-
"""
Created on Sat May 27 10:49:16 2017

@author: Hattori
"""
from multiprocessing import Pool
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sb
from neuron import Neuron_HR as Neuron
import pandas as pd
import time
import datetime
import logging

    
def main():
    nr = Neuron()
    for i in range(0, nr.allsteps-1):
        nr.propagation()
    
    fig = plt.figure(figsize=(12,15))
    ax = fig.add_subplot(2, 1, 1)
    print(len(nr.tmhist))
    print(len(nr.x[0, :]))
    lines = ax.plot(nr.tmhist, nr.x[0, :])
    plt.grid(True)
    plt.show()
        
    
if __name__ == '__main__':
    main()
