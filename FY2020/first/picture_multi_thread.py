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
import datetime
import matplotlib as mpl
from multiprocessing import Pool

# for overflow error
mpl.rcParams['agg.path.chunksize'] = 100000


class Picture():
    def __init__(self,
                 path='C:'):

        self.nowdir = path
        self.csvs = path + '/' + '*.csv'
        self.files = {}
        self.counter = 0
        self.files = glob.glob(self.csvs)
        print(self.files)
        self.tmp0 = []
        self.tmp1 = []
        self.tmp2 = []
        self.tmp3 = []
        self.tmp4 = []
        self.tmp5 = []

        self.gcounter = 0

        self.d = datetime.datetime.today()
        self.dirtmp1 = (self.nowdir + '/raw_data')
        self.dirtmp2 = (self.nowdir + '/raw_data/' +
                        str(self.d.year) + '_' + str(self.d.month) +
                        '_' + str(self.d.day) + '_' +
                        str(self.d.hour) + '_' + str(self.d.minute) +
                        '_' + str(self.d.second))

        if not os.path.isdir(self.nowdir + '/plots'):
            os.mkdir(self.nowdir + '/plots')
        if not os.path.isdir(self.nowdir + '/plots/voltage'):
            os.mkdir(self.nowdir + '/plots/voltage')
        if not os.path.isdir(self.nowdir + '/plots/syn'):
            os.mkdir(self.nowdir + '/plots/syn')
        if not os.path.isdir(self.dirtmp1):
            os.mkdir(self.dirtmp1)
        if not os.path.isdir(self.dirtmp2):
            os.mkdir(self.dirtmp2)

    def run2(self, value):
        self.value = value
        for file_ in self.csv_tmp_list[self.value]:
            df_title = pd.read_csv(file_, nrows=1)
            df = pd.read_csv(file_, index_col=0, skiprows=1)
            filename = os.path.basename(file_).replace('.csv', '')

            df.plot(x='T [ms]', y='V [mV]', figsize=(60, 20), title=str(filename), lw=0.5)
            plt.savefig(fname=self.nowdir + '/plots/voltage/' + filename + '.jpg',
                        dpi=350)
            # plt.show()
            plt.close()

            label1 = 'I_syn'
            df.plot(x='T [ms]', y='I_syn [uA]', figsize=(60, 20), title=str(filename) + label1, lw=0.5)
            plt.title(str(df_title));
            plt.savefig(fname=self.nowdir + '/plots/syn/' + filename + label1 + '.jpg',
                        dpi=350)
            # plt.show()
            plt.close()

            print(str(self.counter) + '個目のファイルを処理します')
            self.counter += 1

            # move csv file
            shutil.move(file_, self.dirtmp2)

        return 0


    def run(self):
        self. process = 16
        self.unit_files = int(len(self.files)/self.process)
        self.csv_tmp_list = list(zip(*[iter(self.files)] * int(self.unit_files)))
        pool = Pool(self.process)
        res = pool.map(self.run2, range(self.process))


def main():
    save_path = "H:/simulation/HH"
    #save_path = "//192.168.13.10/Public/hattori/simulation/HH"
    #save_path = "//192.168.13.10/Public/hattori/simulation/HH/raw_data/2018_10_10_9_46_37(maindata)/Mg_1.0/"
    pic = Picture(save_path)
    pic.run()


if __name__ == '__main__':
     main()
