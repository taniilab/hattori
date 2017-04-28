# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 15:42:07 2017

@author: Hattori
"""
import numpy as np
import matplotlib.pyplot as plt
from neuron import Neuron
from numpy.random import *

def main():
    
    neu = Neuron(0.001, 10, 1, -30, 20, 1, -70, 100)
    
    for i in range(0, int(neu.cycle-1)):
        neu.propagation()
        
    t = np.arange(0, neu.simtime, neu.timestep)
    print(t)
    print(neu.vin[0])
    
    plt.plot(t, neu.vin[0])
    plt.show()
    
        
    """ 
    a = randn(3,4)
    print(a)
    print()
    print(a[2])
    """
if __name__ == '__main__':
    main()