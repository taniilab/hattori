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
                 path='C:', process=1, numneu=1):
        self.process = process
        self.numneu = numneu
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
        self.date = str(self.d.year) + '_' + str(self.d.month) + '_' + str(self.d.day) + '_' + \
                    str(self.d.hour) + '_' + str(self.d.minute) + '_' + str(self.d.second)
        self.dirtmp1 = self.nowdir + '/raw_data'
        self.dirtmp2 = self.nowdir + '/raw_data/' + self.date
        self.dirtmp3 = self.nowdir + '/plots'
        self.dirtmp4 = self.nowdir + '/plots/voltage'
        self.dirtmp5 = self.nowdir + '/plots/voltage/' + self.date
        self.dirtmp6 = self.nowdir + '/plots/syn'
        self.dirtmp7 = self.nowdir + '/plots/syn/' + self.date

        if not os.path.isdir(self.dirtmp1):
            os.mkdir(self.dirtmp1)
        if not os.path.isdir(self.dirtmp2):
            os.mkdir(self.dirtmp2)
        if not os.path.isdir(self.dirtmp3):
            os.mkdir(self.dirtmp3)
        if not os.path.isdir(self.dirtmp4):
            os.mkdir(self.dirtmp4)
        if not os.path.isdir(self.dirtmp5):
            os.mkdir(self.dirtmp5)
        if not os.path.isdir(self.dirtmp6):
            os.mkdir(self.dirtmp6)
        if not os.path.isdir(self.dirtmp7):
            os.mkdir(self.dirtmp7)

    def run2(self, value):
        self.value = value
        for file_ in self.csv_tmp_list[self.value]:
            df_title = pd.read_csv(file_, nrows=1)
            df = pd.read_csv(file_, index_col=0, skiprows=1)
            filename = os.path.basename(file_).replace('.csv', '')

            fig = plt.figure(figsize=(60, 20))
            ax = fig.add_subplot(111)
            time = df['T_0 [ms]']
            voltage = []
            syn = []
            for i in range(self.numneu):
               voltage.append(df['V_{} [mV]'.format(i)])
               ax.plot(time, voltage[i], lw=1)
            plt.title(str(df_title));
            plt.savefig(fname=self.dirtmp5 + '/' + filename + '.jpg', dpi=350)
            plt.close()

            fig = plt.figure(figsize=(60, 20))
            ax = fig.add_subplot(111)
            label1 = 'I_syn'
            for i in range(self.numneu):
                syn.append(df['I_syn_{} [uA]'.format(i)])
                ax.plot(time, syn[i], lw=1)
            plt.title(str(df_title)+label1);
            plt.savefig(fname=self.dirtmp7 + '/' + filename + label1 + '.jpg', dpi=350)
            plt.close()

            print(str(self.counter) + '個目のファイルを処理します')
            self.counter += 1
            # move csv file
            shutil.move(file_, self.dirtmp2)

        return 0


    def run(self):
        self.unit_files = int(len(self.files)/self.process)
        self.csv_tmp_list = list(zip(*[iter(self.files)] * int(self.unit_files)))
        pool = Pool(self.process)
        res = pool.map(self.run2, range(self.process))


def main():
    save_path = "H:/simulation/HH"
    process = 16
    numneu = 1
    pic = Picture(save_path, process, numneu)
    pic.run()


if __name__ == '__main__':
     main()
