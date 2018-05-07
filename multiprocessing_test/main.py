# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 15:42:07 2017

@author: Hattori
"""
import numpy as np
import matplotlib.pyplot as plt
from neuron import Neuron
from numpy.random import *
from multiprocessing import Pool
from multiprocessing import Process

def function(process):
    if process == 0:
        neu = Neuron(0.001, 100000, 1, -30, 20, 1, -70, 100)
            
        for i in range(0, int(neu.cycle-1)):
            neu.propagation(process)
            #text = 'processing : ' + str(process)
            #logger.debug('hello')
        
        t = np.arange(0, neu.simtime, neu.timestep)        
        plt.plot(t, neu.vin[0])
        plt.show()
    
    elif process == 1:
        neu = Neuron(0.001, 100000, 1, -30, 20, 1, -70, 50)
    
        for i in range(0, int(neu.cycle-1)):
            neu.propagation(process)
            #text = 'processing : ' + str(process)
            #logging.warning(text)
        
        t = np.arange(0, neu.simtime, neu.timestep)        
        plt.plot(t, neu.vin[0])
        plt.show()
    
    elif process == 2:
        neu = Neuron(0.001, 100000, 1, -30, 20, 1, -70, 10)
    
        for i in range(0, int(neu.cycle-1)):
            neu.propagation(process)
            #text = 'processing : ' + str(process)
            #logging.warning(text)
        
        t = np.arange(0, neu.simtime, neu.timestep)        
        plt.plot(t, neu.vin[0])
        plt.show()
    
    elif process == 3:
        neu = Neuron(0.001, 100000, 1, -30, 20, 1, -70, 200)
    
        for i in range(0, int(neu.cycle-1)):
            neu.propagation(process)
            #text = 'processing : ' + str(process)
            #logging.warning(text)
        
        t = np.arange(0, neu.simtime, neu.timestep)        
        plt.plot(t, neu.vin[0])
        plt.show()
    

def main():
    process = 4
    p = Pool(process)
    result = p.map(function, range(process))


if __name__ == '__main__':
    main()