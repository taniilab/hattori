# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 14:48:09 2018

@author: Hattori
"""

from multiprocessing import Pool
import os
import numpy as np
import matplotlib.pyplot as plt

class Main():
    def __init__(self, numproc):
        self.numproc = numproc

    def simulate(self, process):
        # parallel processing on each setting value
        self.pid = os.getpid()

        print("kanopero")
        return 0


def test(process):
    print("ju")
    x = np.arange(0, 5, 0.1)
    y = np.sin(x)
    plt.figure()
    plt.plot(x, y)
    plt.show()
    return 9


def main():
    process = 6
    main = Main(process)
    pool = Pool(process)
    cb = pool.map(test, range(process))
    print("kanopero")
    input()

if __name__ == '__main__':
    main()