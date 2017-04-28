# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 17:01:05 2017

@author: 6969p

v1.xx 畳み込み型
v2.xx　積分発火型初期版
v3.01　パラメータ設定関数（再プロット用）を追加
v3.02 発火判定変数(ラスタープロット用)を追加
v4.xx チャネル概念（電流、コンダクタンス）を追加
v5.00 ホジキンハクスレー+ノイズ型
v6.00 メイン関数を並列処理化

memo: ouba=8, 10 強すぎ？　step細かくしないと発散する
"""
from multiprocessing import Pool
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from neuron6 import Neuron6 as Neuron
import pandas as pd
import datetime

#子プロセスのprintはspyderのコンソールに表示されない仕様なので、loggingを使う
import logging

starttime = time.time()
elapsed_time = 0

class Main():
    def plot(self, process):
        #parallel processing on each setting value
        if process == 0:
            self.pid = os.getpid()
            self.progress_co = 0
            self.neuron = Neuron(process, oua=1, oud=1, mean=0, variance=5, pmax=0.003)        

            for i in range(0, np.size(self.neuron.tmhist)-1):      
                self.neuron.propagation()
                if self.progress_co % 100000 == 0:
                    logging.warning('process id : %d : %d steps', self.pid, self.progress_co)
                self.progress_co += 1                      
            return self.neuron
            
        elif process == 1:
            self.pid = os.getpid()
            self.progress_co = 0

            self.neuron = Neuron(process, oua=1, oud=1, mean=0, variance=5, pmax=0.002)        

            for i in range(0, np.size(self.neuron.tmhist)-1):      
                self.neuron.propagation()
                if self.progress_co % 100000 == 0:
                    logging.warning('process id : %d : %d steps', self.pid, self.progress_co)
                self.progress_co += 1                
            return self.neuron      
          
        elif process == 2:
            self.pid = os.getpid()
            self.progress_co = 0
            self.neuron = Neuron(process, oua=1, oud=1, mean=0, variance=5, pmax=0.1)        

            for i in range(0, np.size(self.neuron.tmhist)-1):      
                self.neuron.propagation()
                if self.progress_co % 100000 == 0:
                    logging.warning('process id : %d : %d steps', self.pid, self.progress_co)
                self.progress_co += 1                    
            return self.neuron

        elif process == 3:
            self.pid = os.getpid()
            self.progress_co = 0
            self.neuron = Neuron(process, oua=1, oud=1, mean=0, variance=5, pmax=1)        

            for i in range(0, np.size(self.neuron.tmhist)-1):      
                self.neuron.propagation()
                if self.progress_co % 100000 == 0:
                    logging.warning('process id : %d : %d steps', self.pid, self.progress_co)
                    self.progress_co += 1                    
            return self.neuron
            
        elif process == 4:
            self.pid = os.getpid()
            self.progress_co = 0
            self.neuron = Neuron(process, oua=1, oud=1, mean=0, variance=5, pmax=10)        

            for i in range(0, np.size(self.neuron.tmhist)-1):      
                self.neuron.propagation()
                if self.progress_co % 100000 == 0:
                    logging.warning('process id : %d : %d steps', self.pid, self.progress_co)
                    self.progress_co += 1                    
            return self.neuron

        else:
            pass                    
        
def main():
    numsim = 1
    pool = Pool(numsim) 
    main = Main()
    cb = pool.map(main.plot, range(numsim))
    
    for i in range(0, numsim):
        fig, ax = plt.subplots(nrows = cb[i].numneu+3, figsize=(20, 20))  
        fig.tight_layout()
        fig.subplots_adjust(left=0.05, bottom=0.03)
        fig.text(0.5, 0.01, cb[i].process, fontsize=16, ha='center', va='center')        
        
        #initialize
        lines = []
        for j in range(0, cb[i].numneu):
            lines.append([])
            if cb[i].numneu == 1: 
                ax[j].set_ylim(-100, 50)        
                lines[j], = ax[j].plot(cb[i].tmhist, cb[i].vin[j], color="indigo", markevery=[0, -1])
            else:
                ax[j].set_ylim(-100, 50)        
                lines[j], = ax[j].plot(cb[i].tmhist, cb[i].vin[j], color="indigo", markevery=[0, -1])
    
        #plot
        for j in range(0, cb[i].numneu):
            d = datetime.datetime.today()
            filename = str(d.year) + '_' + str(d.month) + '_' + str(d.day) + '_' + str(d.hour) + '_' + str(d.minute) + '_' + str(d.second) + '_' + str(i) + str(cb[i].palm) + "hh_model.csv"
            #base = np.ones(np.size(cb[i].tmhist)) * (-100)            
            if cb[i].numneu == 1:
                lines[j].set_data(cb[i].tmhist, cb[i].vin[j])
                #ax[j].fill_between(cb[i].tmhist, cb[i].vin[j], base, facecolor="thistle", alpha=0.2)           
            else:
                lines[j].set_data(cb[i].tmhist, cb[i].vin[j])              
                #ax[j].fill_between(cb[i].tmhist, cb[i].vin[j], base, facecolor="thistle", alpha=0.2)                                 

            df = pd.DataFrame({'t[ms]':cb[i].tmhist, 'V[mV]':cb[i].vin[j], 'Inoise[uA?]':cb[i].inoise[j]})
            df.to_csv('./' + filename)
            
        ax[cb[i].numneu].plot(cb[i].tmhist,cb[i].oud[0], markevery=[0, -1])
        ax[cb[i].numneu+1].plot(cb[i].tmhist,cb[i].inoise[0], markevery=[0, -1])
        ax[cb[i].numneu+2].plot(cb[i].tmhist,cb[i].inoise[1], markevery=[0, -1])
    
    elapsed_time = time.time() - starttime
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")    
    
    pool.close()
    pool.join()

    
if __name__ == '__main__':
    main()
    