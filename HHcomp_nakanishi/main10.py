# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 11:26:16 2017

@author: Nakanishi
"""

from multiprocessing import Pool
import numpy as np
import pandas as pd

from neuron10 import Neuron

def plot(process):
    if process == 0:
        for k in range(1, 11):
            neu = Neuron(0.005, 60000, 1, -65, 0.1, 0.1, k, 0.04, 1)
            for j in range(0, neu.numsyn):
                neu.d[j] = ((5 / neu.numsyn - 1) * j) / neu.dt
            for i in range(0, int(neu.cycle-1)):
                neu.propagation()
                print(neu.nowstep, process)
            t = np.arange(0, neu.simtime, neu.dt) / 1000
            df = pd.DataFrame({'t' : t, 'vin' : neu.vin[0], 'vde' : neu.vde[0], 'isyn' : neu.isynd[0] * (-1)})
            df.to_csv('pnoise%g-ave%d-gh%g-syng-%g-numsyn%d-5.csv' %(neu.pnoise, neu.ave , neu.gH, neu.Pmax, neu.numsyn))
            print('fin', process)

    elif process == 1:
        for k in range(1, 11):
            neu = Neuron(0.005, 60000, 1, -65, 0.1, 0.3, k, 0.04, 1)
            for j in range(0, neu.numsyn):
                neu.d[j] = ((5 / neu.numsyn - 1) * j) / neu.dt
            for i in range(0, int(neu.cycle-1)):
                neu.propagation()
                print(neu.nowstep, process)
            t = np.arange(0, neu.simtime, neu.dt) / 1000
            df = pd.DataFrame({'t' : t, 'vin' : neu.vin[0], 'vde' : neu.vde[0], 'isyn' : neu.isynd[0] * (-1)})
            df.to_csv('pnoise%g-ave%d-gh%g-syng-%g-numsyn%d-5.csv' %(neu.pnoise, neu.ave , neu.gH, neu.Pmax, neu.numsyn))
            print('fin', process)

    elif process == 2:
        for k in range(1, 11):
            neu = Neuron(0.005, 60000, 1, -65, 0.1, 0.5, k, 0.04, 1)
            for j in range(0, neu.numsyn):
                neu.d[j] = ((5 / neu.numsyn - 1) * j) / neu.dt
            for i in range(0, int(neu.cycle-1)):
                neu.propagation()
                print(neu.nowstep, process)
            t = np.arange(0, neu.simtime, neu.dt) / 1000
            df = pd.DataFrame({'t' : t, 'vin' : neu.vin[0], 'vde' : neu.vde[0], 'isyn' : neu.isynd[0] * (-1)})
            df.to_csv('pnoise%g-ave%d-gh%g-syng-%g-numsyn%d-5.csv' %(neu.pnoise, neu.ave , neu.gH, neu.Pmax, neu.numsyn))
            print('fin', process)

    elif process == 3:
        for k in range(1, 11):
            neu = Neuron(0.005, 60000, 1, -65, 0.1, 0.6, k, 0.04, 1)
            for j in range(0, neu.numsyn):
                neu.d[j] = ((5 / neu.numsyn - 1) * j) / neu.dt
            for i in range(0, int(neu.cycle-1)):
                neu.propagation()
                print(neu.nowstep, process)
            t = np.arange(0, neu.simtime, neu.dt) / 1000
            df = pd.DataFrame({'t' : t, 'vin' : neu.vin[0], 'vde' : neu.vde[0], 'isyn' : neu.isynd[0] * (-1)})
            df.to_csv('pnoise%g-ave%d-gh%g-syng-%g-numsyn%d-5.csv' %(neu.pnoise, neu.ave , neu.gH, neu.Pmax, neu.numsyn))
            print('fin', process)
        
    elif process == 4:
        for k in range(1, 11):
            neu = Neuron(0.005, 60000, 1, -65, 0.1, 0.7, k * 5, 0.04, 1)
            for j in range(0, neu.numsyn):
                neu.d[j] = ((5 / neu.numsyn - 1) * j) / neu.dt
            for i in range(0, int(neu.cycle-1)):
                neu.propagation()
                print(neu.nowstep, process)
            t = np.arange(0, neu.simtime, neu.dt) / 1000
            df = pd.DataFrame({'t' : t, 'vin' : neu.vin[0], 'vde' : neu.vde[0], 'isyn' : neu.isynd[0] * (-1)})
            df.to_csv('pnoise%g-ave%d-gh%g-syng-%g-numsyn%d-5.csv' %(neu.pnoise, neu.ave , neu.gH, neu.Pmax, neu.numsyn))
            print('fin', process)

def main():

    process = 5
    p = Pool(process)
    p.map(plot, range(process))
    p.close

if __name__=='__main__':
    main()
