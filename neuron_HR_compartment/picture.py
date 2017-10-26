# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 14:54:07 2017

@author: Hattori
"""
import matplotlib.pyplot as plt
import glob
import os
import pandas as pd
import numpy as np
from PIL import Image


class Picture():
    def __init__(self,
                 path='C:/Users/Hattori/Documents/HR_results'):

        self.nowdir = path
        if not os.path.isdir(self.nowdir + '/plots'):
            os.mkdir(self.nowdir + '/plots')
        self.csvs = path + '/' + '*.csv'
        self.files = {}
        self.counter = 0
        self.files = glob.glob(self.csvs)
        print(self.csvs)

    def run(self):

        for file_ in self.files:
            df = pd.read_csv(file_, index_col=0)
            matrix = df.as_matrix()
            filename = os.path.basename(file_).replace('.csv', '')

            plt.title(filename)
            linex, = plt.plot(matrix[:, 6], matrix[:, 8], lw=1)
            # plt.plot(matrix[:, 6], matrix[:, 9], lw=1)
            linez, = plt.plot(matrix[:, 6], matrix[:, 10], lw=1)
            plt.savefig(filename=self.nowdir + '/plots/' + filename + '.jpg',
                        dpi=350)
            linex.remove()
            linez.remove()

            lineisyn, = plt.plot(matrix[:, 6], matrix[:, 2], lw=1)
            plt.savefig(filename=self.nowdir + '/plots/' + filename +
                        'syn.jpg', dpi=350)
            plt.clf()

            print(str(self.counter) + '個目のファイルを処理します')
            self.counter += 1
