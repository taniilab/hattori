# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 13:45:42 2016

@author: aromagedon
"""
import numpy as np
from numpy.random import *

class Neuron2:
    
    def __init__(self, num_neurons, vth, vrm, tau, def_w, phigh, deltatime):
        self.numneu = num_neurons
        self.vth = vth
        self.vrm = np.ones(self.numneu) * vrm
        self.tau = tau / deltatime
        self.phigh = np.ones(self.numneu) * phigh
        self.f_flag = np.ones(self.numneu) < 0
        self.input = np.zeros((self.numneu,self.numneu)) + self.vrm
        self.output = np.zeros(self.numneu) + self.vrm
        self.vin = np.zeros(self.numneu) + self.vrm
        self.n = np.zeros((self.numneu, self.numneu)) + self.vrm
        self.w = np.ones((self.numneu, self.numneu)) * def_w
        self.sigma = np.zeros(self.numneu)
        self.temp = np.zeros(self.numneu)
        self.sigmoid_gain = 100
        
    def set_weight(self, w):
        for i in range(0, self.numneu):
            for j in range(0, self.numneu):
                self.w[i][j] = w[i][j]
        return self.w

    #信号伝達
    def propagation(self, act_function):
        self.sigma = np.zeros(self.numneu)
        self.act_f = act_function
        self.temp = ((self.input - self.vrm) + (self.n - self.vrm)) * self.w
        for i in range(0, self.numneu):
            for j in range(0, self.numneu):
                self.sigma[i] += self.temp[i][j]
        self.vin = (-(self.vin -self.vrm) + self.sigma) / self.tau + self.vin
        
        if self.act_f == "step":
            self.__step_function()
        elif self.act_f == "sigmoid":
            self.__sigmoid_function()
        elif self.act_f == "relu":
            self.__relu_function()
        elif self.act_f == "linear":
            self.__linear_function()
        else:
            return (self.vin, self.output)
        
        for i in range(0, self.numneu):
            for j in range(0, self.numneu):
                self.input[j][i] = self.output[i]
   
        return (self.vin, self.output)
   
    #活性化関数
    def __step_function(self):
        for i in range(0, self.numneu):
            if self.f_flag[i] == True:
                self.f_flag[i] = False                
                self.vin[i] = self.vrm[i]     
            elif self.vin[i] > self.vth:
                self.f_flag[i] = True                
                self.vin[i] = self.phigh[i]
            else:
                self.f_flag[i] = False
        self.output = np.array(self.vin > self.vth)
        self.output = self.output.astype(np.double)
        #オフセット、ゲイン調整
        self.output = self.output.astype(np.double)
        self.output = self.output * (self.phigh - self.vrm) + self.vrm
        
        return 0
    
    def __sigmoid_function(self):
        for i in range(0, self.numneu):
            if self.f_flag[i] == True:
                self.f_flag[i] = False                
                self.vin[i] = self.vrm[i]     
            elif self.vin[i] > self.vth:
                self.f_flag[i] = True                
                self.vin[i] = self.phigh[i]
            else:
                self.f_flag[i] = False
        
        self.output = ((self.phigh - self.vrm) / (1 + np.exp(-self.sigmoid_gain * (self.vin + self.vth)))) + self.vrm 
        return 0    

    def __relu_function(self):
        return np.maximum(0, self.vin)
    
    def __linear_function(self):
        self.output = self.vin
        return
    
    def __p_linear_function(self):
        if self.ref_period == True:
            if self.init_ref == True:
                self.init_ref == False                
                return -0.2
            elif self.ref_counter < 10:
                self.ref_counter += 1
                return self.vin /2
            else:
                self.ref_counter = 0
                self.ref_period = False
                return self.vin
        elif self.vin > self.vth:
            self.ref_period = True
            self.__init_ref = True
            return 1.5
        else:
            return self.vin