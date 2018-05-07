# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 14:54:07 2017

@author: Hattori
"""
import matplotlib.pyplot as plt
import glob
import os
import shutil
import pandas as pd
import numpy as np
from PIL import Image
import datetime


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
        self.tmp0 = []
        self.tmp1 = []
        self.tmp2 = []
        self.tmp3 = []
        self.tmp4 = []
        self.tmp5 = []

        self.gcounter = 0

        self.d = datetime.datetime.today()
        self.dirtmp = (self.nowdir + '/tmp/' +
                       str(self.d.year) + '_' + str(self.d.month) +
                       '_' + str(self.d.day) + '_' +
                       str(self.d.hour) + '_' + str(self.d.minute) +
                       '_' + str(self.d.second))
            
        if not os.path.isdir(self.dirtmp):
            os.mkdir(self.dirtmp)
            

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

            """
            lineiext, = plt.plot(matrix[:, 6], matrix[:, 1], lw=1)
            plt.savefig(filename=self.nowdir + '/plots/' + filename +
                        'iext.jpg', dpi=350)
            lineiext.remove()
            """
            if self.gcounter == 0:
                self.tmp0 = matrix[:, 6]
                self.tmp1 = matrix[:, 8]
                self.gcounter += 1
            elif self.gcounter == 1:
                self.tmp2 = matrix[:, 8]
                self.gcounter += 1
            elif self.gcounter == 2:
                self.tmp3 = matrix[:, 8]
                self.gcounter += 1
            elif self.gcounter == 3:
                self.tmp4 = matrix[:, 8]
                self.gcounter += 1
            elif self.gcounter == 4:
                self.tmp5 = matrix[:, 8]
                linetmp1, = plt.plot(self.tmp0, self.tmp1, lw=1)
                linetmp2, = plt.plot(self.tmp0, self.tmp2, lw=1)
                linetmp3, = plt.plot(self.tmp0, self.tmp3, lw=1)
                linetmp4, = plt.plot(self.tmp0, self.tmp4, lw=1)
                linetmp5, = plt.plot(self.tmp0, self.tmp5, lw=1)
                plt.ylim(-2, 3)
                plt.savefig(filename=self.nowdir + '/plots/' + filename +
                            'prop.jpg', dpi=350)
                self.gcounter = 0
            else:
                pass

            plt.clf()

            print(str(self.counter) + '個目のファイルを処理します')
            self.counter += 1

            # move csv file            
            shutil.move(file_, self.dirtmp)
