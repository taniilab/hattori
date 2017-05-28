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
#import seaborn as sb
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
        

starttime = time.time()
elapsed_time = 0

class Main():
    def plot(self, process):
        #parallel processing on each setting value
        if process == 0:
            self.pid = os.getpid()
            self.progress_co = 0
            self.nr = Neuron(Iext=1)     
            for i in range(0, self.nr.allsteps-1):      
                self.nr.propagation()
                if self.progress_co % 100000 == 0:
                    logging.warning('process id : %d : %d steps', self.pid, self.progress_co)
                self.progress_co += 1                      
            return self.nr
            
        elif process == 1:
            self.pid = os.getpid()
            self.progress_co = 0
            self.nr = Neuron(Iext=2)        
            for i in range(0, self.nr.allsteps-1):      
                self.nr.propagation()
                if self.progress_co % 100000 == 0:
                    logging.warning('process id : %d : %d steps', self.pid, self.progress_co)
                self.progress_co += 1                
            return self.nr      
          
        elif process == 2:
            self.pid = os.getpid()
            self.progress_co = 0
            self.nr = Neuron(Iext=2.5)        
            for i in range(0, self.nr.allsteps-1):      
                self.nr.propagation()
                if self.progress_co % 100000 == 0:
                    logging.warning('process id : %d : %d steps', self.pid, self.progress_co)
                self.progress_co += 1                    
            return self.nr

        elif process == 3:
            self.pid = os.getpid()
            self.progress_co = 0
            self.nr = Neuron()        

            for i in range(0, self.nr.allsteps-1):      
                self.nr.propagation()
                if self.progress_co % 100000 == 0:
                    logging.warning('process id : %d : %d steps', self.pid, self.progress_co)
                    self.progress_co += 1                    
            return self.nr
            
        elif process == 4:
            self.pid = os.getpid()
            self.progress_co = 0
            self.nr = Neuron()        

            for i in range(0, self.nr.allsteps-1):      
                self.nr.propagation()
                if self.progress_co % 100000 == 0:
                    logging.warning('process id : %d : %d steps', self.pid, self.progress_co)
                    self.progress_co += 1                    
            return self.nr

        else:
            pass                    
        
def main():
    numsim = 3
    pool = Pool(numsim) 
    main = Main()
    cb = pool.map(main.plot, range(numsim))
    
    for i in range(0, numsim):
        fig, ax = plt.subplots(nrows = cb[i].numneu, figsize=(20, 20))  
        """        
        fig.tight_layout()
        fig.subplots_adjust(left=0.05, bottom=0.03)
        fig.text(0.5, 0.01, cb[i].process, fontsize=16, ha='center', va='center')        
        """     
        #initialize
        lines = []
        tm = np.arange(0, cb[i].allsteps, 1)
        for j in range(0, cb[i].numneu):
            lines.append([])
            if cb[i].numneu == 1: 
                lines[j], = ax.plot(tm, cb[i].x[j], color="indigo", markevery=[0, -1])
            else:
                lines[j], = ax[j].plot(tm, cb[i].x[j], color="indigo", markevery=[0, -1])
    
        #plot
        for j in range(0, cb[i].numneu):
            d = datetime.datetime.today()
            filename = str(d.year) + '_' + str(d.month) + '_' + str(d.day) + '_' + str(d.hour) + '_' + str(d.minute) + '_' + str(d.second) + '_' + str(i) + "HR_model.csv"
            #base = np.ones(np.size(cb[i].tmhist)) * (-100)            
            if cb[i].numneu == 1:
                lines[j].set_data(tm, cb[i].x[j])
                #ax[j].fill_between(cb[i].tmhist, cb[i].vin[j], base, facecolor="thistle", alpha=0.2)           
            else:
                lines[j].set_data(tm, cb[i].x[j])              
                #ax[j].fill_between(cb[i].tmhist, cb[i].vin[j], base, facecolor="thistle", alpha=0.2)                                 

            df = pd.DataFrame({'t[ms]':tm, 'V[mV]':cb[i].x[j]})
            df.to_csv('./' + filename)
            
    
    elapsed_time = time.time() - starttime
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")    
    print(cb[0].x[0])
    print(cb[0].aaa)
    
    pool.close()
    pool.join()

    
if __name__ == '__main__':
    main()