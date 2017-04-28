# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 15:42:16 2017

@author: Hattori
"""
import numpy as np

class Neuron():
    
    def __init__(self, timestep, simtime, numneu, vth, vfire, tau, vres, current):
        self.timestep = timestep
        self.simtime = simtime
        self.cycle = simtime/timestep
        self.numneu = numneu
        self.vin = np.ones((self.numneu, self.cycle)) * vres
        self.vth = vth
        self.vfire = vfire
        self.tau = tau
        self.vres = vres
        self.current = current
        #self.input = np.zeros((self.numneu, (simtime/timestep)))
        self.nowstep = 0
        self.fire_flag = False
        
    def propagation(self): 
        print(self.nowstep)
        for i in range(0, self.numneu):            
            if self.fire_flag == True:
                print("aaa")
                self.vin[i, self.nowstep] = self.vres    
                self.fire_flag = False
            
            self.dv = ((-(self.vin[i, self.nowstep]- self.vres) + self.current) / self.tau) * self.timestep
                      
            self.vin[i, self.nowstep + 1] = self.vin[i, self.nowstep] + self.dv
                    
            if self.vin[i, self.nowstep] >= self.vth:
                print("bbb")
                self.vin[i, self.nowstep ] = self.vfire    
                self.fire_flag = True
                        
        self.nowstep += 1