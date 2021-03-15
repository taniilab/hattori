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
from statistics import mean, median,variance,stdev
import re
import itertools


# for overflow error
mpl.rcParams['agg.path.chunksize'] = 100000
process = 20

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
        self.d = datetime.datetime.today()
        self.date = str(self.d.year) + '_' + str(self.d.month) + '_' + str(self.d.day) + '_' + \
                    str(self.d.hour) + '_' + str(self.d.minute) + '_' + str(self.d.second)
        self.dirtmp1 = self.nowdir + '/cv'
        if not os.path.isdir(self.dirtmp1):
            os.mkdir(self.dirtmp1)

    def run2(self, value):
        self.value = value
        cv_list = []
        ampa_list = []
        nmda_list = []

        for file_ in self.csv_tmp_list[self.value]:
            print(self.csv_tmp_list[self.value])
            df_title = pd.read_csv(file_, nrows=1)
            df = pd.read_csv(file_, index_col=0, skiprows=1)
            print(file_)
            spike = df['fire_0']
            spike = spike.values
            spike_time_list = []
            isi = []

            filename = os.path.basename(file_).replace('.csv', '')
            filename = re.findall("P_AMPA[0-9]\.[0-9]_P_NMDA[0-9]\.[0-9]", filename)[0]
            print(filename)
            ampa_list.append(int(float(filename[6:9])*10))
            nmda_list.append(int(float(filename[16:19])*10))

            for i in range(len(spike)):
                if spike[i] != 0:
                    spike_time_list.append(spike[i])

            for i in range(len(spike_time_list)-1):
                isi.append(spike_time_list[i+1]-spike_time_list[i])
            try:
                #cv = variance(isi[5:]) / mean(isi[5:])
                cv = variance(isi) / mean(isi)
            except:
                cv = 0
            cv_list.append(cv)

            print(str(self.counter) + '個目のファイルを処理します')
            self.counter += 1

        return cv_list, ampa_list, nmda_list

    def run(self):
        num_ampa = 16
        num_nmda = 16
        cv_hist = np.zeros((num_ampa, num_nmda))
        list_columns = []
        list_rows = []
        self.unit_files = int(len(self.files)/self.process)
        self.csv_tmp_list = list(zip(*[iter(self.files)] * int(self.unit_files)))
        pool = Pool(self.process)
        res = pool.map(self.run2, range(self.process))
        print(res)

        for i, j in itertools.product(range(process), range(int(len(self.files)/process))):
            cv_hist[res[i][1][j]][res[i][2][j]] += res[i][0][j]

        cv_hist /= 5
        for i in range(num_ampa):
            list_rows.append("AMPA: "+str(round(i*0.1, 1)))
        for j in range(num_nmda):
            list_columns.append("NMDA: "+str(round(j*0.1, 1)))
        df = pd.DataFrame(cv_hist, columns=list_columns, index=list_rows)
        print("kashikoma")
        df.to_csv(self.nowdir+"/cv/heatmap_cv.csv")

def main():
    save_path = "C:/sim/raw_data/Mg16/raw_data/2020_10_9_10_50_53"
    #save_path = "C:/sim/raw_data/test"

    numneu = 1
    pic = Picture(save_path, process, numneu)
    pic.run()


if __name__ == '__main__':
     main()
