# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 11:31:47 2016

@author: aromagedon
"""

import numpy as np
from numpy.random import *

class Neuron:   
    deltatime = 0.1
    hist = 50
    
    """
    <<<コンストラクタ>>>
    numvin:入力端子数
    numvout:出力端子数
    vth:スレッショルド電圧
    hist:畳み込みする範囲
    deltatime:単位時間
    vin:入力電圧
    vout:出力電圧    
    n:ノイズ
    b:バイアス
    uir:単位インパルス応答
    sigma:畳み込み積分
    a:単位インパルス応答の減衰速度
    """
    def __init__(self, numvin, numvout, vth):
        self.numvin = numvin
        self.numvout = numvout
        self.vth = vth
        self.hist = Neuron.hist
        self.deltatime = Neuron.deltatime        
        #self.w = np.random.rand(self.numvin, self.hist)
        #self.vin = np.ones((self.numvin, self.hist))        
        #self.n = np.zeros((self.numvin, self.hist))
        self.vin = np.zeros((self.numvin, self.hist))
        self.n = np.random.randn(self.numvin, self.hist) * 0.25
        self.w = np.ones((self.numvin, self.hist))*0.55
        self.b = np.ones((self.numvin, self.hist))*0.5
        self.uir = np.zeros((self.numvin, self.hist))
        self.vout = np.zeros((self.numvin, 1))
        self.sigma = 0
        self.a = 10
        self.rmpotential = 0
        self.ref_period = False
        self.ref_counter = 0
        self.init_ref = False        
        for i in range(0, numvin):
            for j in range(0, self.hist):
                self.uir[i][j] = self.unit_impulse_res(self.deltatime * j)

    def set_weight(self, w):
        for i in range(0, self.numvin):
            for j in range(0, self.hist):
                self.w[i][j] = w[i]
        return self.w

    def set_noise(self):
        for i in range(0, self.numvin):
            self.n[i][0] = np.random.randn() * 0.25
        return self.n
    
    #伝搬関数(入力を統合して出力を生成)
    def prop_relu(self, sigma):
        self.sigma = self.integration()
        for i in range(0, self.numvin):
            for j in range(0, self.hist):
                self.vin[i][self.hist - 1 - j] = self.vin[i][self.hist - 2 - j]
                self.n[i][self.hist - 1 - j] = self.n[i][self.hist - 2 - j]
        for i in range(0, self.numvout):
            self.vout[i][0] = self.__relu_function(self.sigma)
        return 0
        
    def prop_step(self, sigma):
        self.sigma = self.integration()
        for i in range(0, self.numvin):
            for j in range(0, self.hist):
                self.vin[i][self.hist - 1 - j] = self.vin[i][self.hist - 2 - j]
                self.n[i][self.hist - 1 - j] = self.n[i][self.hist - 2 - j]
        for i in range(0, self.numvout):
            self.vout[i][0] = self.__step_function(self.sigma)        
        return 0

    def prop_sigmoid(self, sigma):
        self.sigma = self.integration()        
        for i in range(0, self.numvin):
            for j in range(0, self.hist):
                self.vin[i][self.hist - 1 - j] = self.vin[i][self.hist - 2 - j]        
                self.n[i][self.hist - 1 - j] = self.n[i][self.hist - 2 - j]
        for i in range(0, self.numvout):
            self.vout[i][0] = self.__sigmoid_function(self.sigma)
        return 0

    def prop_p_linear(self):
        self.sigma = self.integration()
        self.sigma = self.__well_function(self.sigma)
        for i in range(0, self.numvin):
            for j in range(0, self.hist):
                self.vin[i][self.hist - 1 - j] = self.vin[i][self.hist - 2 - j]        
                self.n[i][self.hist - 1 - j] = self.n[i][self.hist - 2 - j]
        for i in range(0, self.numvout):
            self.vout[i][0] = self.__p_linear_function(self.sigma)
        return 0
        
    def prop_linear(self, sigma):
        self.sigma = self.integration()        
        for i in range(0, self.numvin):
            for j in range(0, self.hist):
                self.vin[i][self.hist - 1 - j] = self.vin[i][self.hist - 2 - j]
                self.n[i][self.hist - 1 - j] = self.n[i][self.hist - 2 - j]
        for i in range(0, self.numvout):
            self.vout[i][0] = self.__linear_function(self.sigma)
        return 0
    
    def integration(self):
        self.sigma = 0
        self.temp = self.vin * self.uir * self.w + self.n * self.uir * self.w
        for i in range(0, self.numvin):
            for j in range(0, self.hist):
                self.sigma += self.temp[i][j]       
        return self.sigma 

    #単位インパルス応答
    def unit_impulse_res(self, x):
        if x >= 0:
            return np.exp(self.a * (-x))
        else:
            return 0
            
    #静止膜電位関数(ノイズ正規分布の調整だけで十分かも)
    def __well_function(self, sigma):
        self.delpo = sigma - self.rmpotential
        if self.delpo > 0:
            return self.sigma - (self.delpo * self.delpo * 0.3)
        if self.delpo < 0:
            return self.sigma + (self.delpo * self.delpo * 0.6)
        else:
            return 0

    #活性化関数
    def __relu_function(self, x):
        return np.maximum(0, x)

    def __step_function(self, x):
        return np.array(x > 1, dtype = np.double)
        
    def __sigmoid_function(self, x):
        return 1 / (1 + np.exp(-x))    

    def __linear_function(self, x):
        return x
    
    def __p_linear_function(self, x):
        if self.ref_period == True:
            if self.init_ref == True:
                self.init_ref == False                
                return -0.2
            elif self.ref_counter < 10:
                self.ref_counter += 1
                return x /2
            else:
                self.ref_counter = 0
                self.ref_period = False
                return x
        elif x > self.vth:
            self.ref_period = True
            self.__init_ref = True
            return 1.5
        else:
            return x