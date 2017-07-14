# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 14:37:37 2017

@author: Hattori
"""
from multiprocessing import Pool
import os


def multi_process(num_process):
    pid = os.getpid()
    for i in range(0, 1000000):
        print("process id:" + str(pid))


def main():
    num_process = 6
    pool = Pool(num_process)
    pool.map(multi_process, range(num_process))

    pool.close()
    pool.join()

if __name__ == '__main__':
    main()
