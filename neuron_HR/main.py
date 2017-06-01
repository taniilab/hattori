"""
Created on Sat May 27 10:49:16 2017

@author: Hattori
"""
# coding: UTF-8
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

starttime = time.time()
elapsed_time = 0

#palameter setting
"""
type of synaptic coupling
1.electrical synapse
2.chemical synapse
3.alpha function
4.alpha function with excitatory and inhibitory synapse
"""
palm1 = {"Syncp":2, "Iext":2.5, "b":2.9, "r":0.006, "D":0, "tausyn":3}
palm2 = {"Syncp":2, "Iext":2.5, "b":2.9, "r":0.006, "D":0, "tausyn":3}
palm3 = {"Syncp":2, "Iext":2.5, "b":2.9, "r":0.006, "D":0, "tausyn":3}
palm4 = {"Syncp":4, "Iext":2.5, "b":2.9, "r":0.006, "D":0, "tausyn":3}
palm5 = {"Syncp":4, "Iext":2.5, "b":2.9, "r":0.006, "D":0, "tausyn":3}
palm6 = {"Syncp":4, "Iext":2.5, "b":2.9, "r":0.006, "D":0, "tausyn":3}

class Main():
    def plot(self, process):
        #parallel processing on each setting value
        if process == 0:
            self.pid = os.getpid()
            self.progress_co = 0
            self.nr = Neuron(**palm1)     
            for i in range(0, self.nr.allsteps-1):      
                self.nr.propagation()
                if self.progress_co % 100000 == 0:
                    logging.warning('process id : %d : %4d steps', self.pid, self.progress_co)
                self.progress_co += 1                      
            return self.nr
            
        elif process == 1:
            self.pid = os.getpid()
            self.progress_co = 0
            self.nr = Neuron(**palm2)        
            for i in range(0, self.nr.allsteps-1):      
                self.nr.propagation()
                if self.progress_co % 100000 == 0:
                    logging.warning('process id : %d : %d steps', self.pid, self.progress_co)
                self.progress_co += 1                
            return self.nr      
          
        elif process == 2:
            self.pid = os.getpid()
            self.progress_co = 0
            self.nr = Neuron(**palm3)        
            for i in range(0, self.nr.allsteps-1):      
                self.nr.propagation()
                if self.progress_co % 100000 == 0:
                    logging.warning('process id : %d : %d steps', self.pid, self.progress_co)
                self.progress_co += 1                    
            return self.nr

        elif process == 3:
            self.pid = os.getpid()
            self.progress_co = 0
            self.nr = Neuron(**palm4)        
            for i in range(0, self.nr.allsteps-1):      
                self.nr.propagation()
                if self.progress_co % 100000 == 0:
                    logging.warning('process id : %d : %d steps', self.pid, self.progress_co)
                    self.progress_co += 1                    
            return self.nr
            
        elif process == 4:
            self.pid = os.getpid()
            self.progress_co = 0
            self.nr = Neuron(**palm5)        
            for i in range(0, self.nr.allsteps-1):      
                self.nr.propagation()
                if self.progress_co % 100000 == 0:
                    logging.warning('process id : %d : %d steps', self.pid, self.progress_co)
                    self.progress_co += 1                    
            return self.nr
        
        elif process == 5:
            self.pid = os.getpid()
            self.progress_co = 0
            self.nr = Neuron(**palm6)        
            for i in range(0, self.nr.allsteps-1):      
                self.nr.propagation()
                if self.progress_co % 100000 == 0:
                    logging.warning('process id : %d : %d steps', self.pid, self.progress_co)
                    self.progress_co += 1                    
            return self.nr

        else:
            pass                    
        
def main():
    numsim = 6
    pool = Pool(numsim) 
    main = Main()
    cb = pool.map(main.plot, range(numsim))
        
    for i in range(0, numsim):
        fig, ax = plt.subplots(nrows = cb[i].numneu+1, figsize=(12,12))  
        
        #initialize
        lines = []
        tm = np.arange(0, cb[i].allsteps*cb[i].dt, cb[i].dt)
        
        for j in range(0, cb[i].numneu):
            lines.append([])        
            if cb[i].numneu == 1: 
                lines[j], = ax[j].plot(tm, cb[i].x[j], color="indigo", markevery=[0, -1])
            else:
                lines[j], = ax[j].plot(tm, cb[i].x[j], color="indigo", markevery=[0, -1])
        """
        ax[cb[i].numneu] = plt.subplot2grid((2, cb[i].numneu), (1, 0), colspan=cb[i].numneu)    
        ax[cb[i].numneu].plot(tm, cb[i].x[0], color="indigo", markevery=[0, -1])
        """
        ax[cb[i].numneu].plot(tm, cb[i].Isyn[0], color="indigo", markevery=[0, -1])
        
        #plot
        for j in range(0, cb[i].numneu):
            d = datetime.datetime.today()
            filename = str(d.year) + '_' + str(d.month) + '_' + str(d.day) + '_' + str(d.hour) + '_' + str(d.minute) + '_' + str(d.second) + '_' + str(i) + "HR_model.csv"
           
            if cb[i].numneu == 1:
                lines[j].set_data(tm, cb[i].x[j])
            else:
                lines[j].set_data(tm, cb[i].x[j])                             

            df = pd.DataFrame({'t':tm, 'V':cb[i].x[j], 'I':cb[i].Isyn[j]})
            df.to_csv('./' + filename)
            
    
    elapsed_time = time.time() - starttime
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")    
    print("おしまいのかしこま!!!")
    
    pool.close()
    pool.join()

    
if __name__ == '__main__':
    main()