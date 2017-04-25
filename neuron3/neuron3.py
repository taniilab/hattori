# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 13:45:42 2016

@author: aromagedon

v1.xx 畳み込み型
v2.xx　積分発火型初期版
v3.01　パラメータ設定関数（再プロット用）を追加
v3.02 発火判定変数(ラスタープロット用)を追加
"""
import numpy as np
from numpy.random import *
import matplotlib.pyplot as plt

class Neuron3:
    
    #コンストラクタ
    def __init__(self, num_neurons, vth, vrm, tau, deltatime, phightime, phigh, def_w, sigmoid_gain, ab_refactftime, ab_weak, simtime):
        self.numneu = num_neurons
        self.vth = np.ones(self.numneu) * vth
        self.vth_refer = np.ones(self.numneu) * vth
        self.vrm = np.ones(self.numneu) * vrm
        self.tau = tau / deltatime
        self.deltatime = deltatime
        self.phightime = phightime /deltatime
        self.phcounter = np.zeros(self.numneu)
        self.phigh = np.ones(self.numneu) * phigh
        self.f_flag = np.ones(self.numneu) < 0
        self.input = np.zeros((self.numneu, self.numneu)) + self.vrm
        self.output = np.zeros(self.numneu) + self.vrm
        self.vin = np.zeros(self.numneu) + self.vrm
        self.n = np.zeros((self.numneu, self.numneu)) + self.vrm
        self.w = np.ones((self.numneu, self.numneu)) * def_w
        self.sigma = np.zeros(self.numneu)
        self.temp = np.zeros(self.numneu)
        self.sigmoid_comparison = np.zeros(self.numneu)
        self.sigmoid_gain = sigmoid_gain
        self.simtime = simtime /deltatime
        self.rasterbuf = np.zeros((self.numneu, self.simtime))
        self.raster = np.zeros((self.numneu, self.simtime))
        self.rasterco = 0
        self.prop_first = True
        
        #絶対不応期
        self.ab_vth = np.ones(self.numneu) * 10000
        self.ab_refact = np.ones(self.numneu) < 0
        self.ab_refactftime = np.ones(self.numneu) * ab_refactftime
        self.ab_elapsedtime = np.zeros(self.numneu)
        self.ab_weak = ab_weak

        #self.simtime = simtime

        #ノイズ用
        self.nsource = np.zeros((self.numneu, self.numneu)) 
        self.nth = np.ones((self.numneu, self.numneu)) * 0.9
        """
        #相対不応期        
        self.rel_refact = np.ones(self.numneu) < 0
        self.rel_elapsedtime = np.zeros(self.numneu)
        self.rel_refacttau = 5
        self.vth_vthref_ratio = np.ones(self.numneu) * self.ab_vth
        """
    #パラメータ設定    
    def set_weight(self, w):
        for i in range(0, self.numneu):
            for j in range(0, self.numneu):
                self.w[i][j] = w[i][j]
        return self.w

    def set_neuron_palameter(self, num_neurons, vth, vrm, tau, deltatime, phightime, phigh, def_w, sigmoid_gain, ab_refactftime, ab_weak, simtime):
        self.numneu = num_neurons
        self.vth = np.ones(self.numneu) * vth
        self.vth_refer = np.ones(self.numneu) * vth
        self.vrm = np.ones(self.numneu) * vrm
        self.tau = tau / deltatime
        self.deltatime = deltatime
        self.phightime = phightime /deltatime
        self.phcounter = np.zeros(self.numneu)
        self.phigh = np.ones(self.numneu) * phigh
        self.f_flag = np.ones(self.numneu) < 0
        self.input = np.zeros((self.numneu, self.numneu)) + self.vrm
        self.output = np.zeros(self.numneu) + self.vrm
        self.vin = np.zeros(self.numneu) + self.vrm
        self.n = np.zeros((self.numneu, self.numneu)) + self.vrm
        self.w = np.ones((self.numneu, self.numneu)) * def_w
        self.sigma = np.zeros(self.numneu)
        self.temp = np.zeros(self.numneu)
        self.sigmoid_comparison = np.zeros(self.numneu)
        self.sigmoid_gain = sigmoid_gain
        self.simtime = simtime /deltatime
        self.rasterbuf = np.zeros((self.numneu, self.simtime))
        self.raster = np.zeros((self.numneu, self.simtime))
        self.rasterco = 0
        self.prop_first = True        
        
        #絶対不応期
        self.ab_vth = np.ones(self.numneu) * 10000
        self.ab_refact = np.ones(self.numneu) < 0
        self.ab_refactftime = np.ones(self.numneu) * ab_refactftime
        self.ab_elapsedtime = np.zeros(self.numneu)
        self.ab_weak = ab_weak

        #self.simtime = simtime

        #ノイズ用
        self.nsource = np.zeros((self.numneu, self.numneu)) 
        self.nth = np.ones((self.numneu, self.numneu)) * 0.9
        """
        #相対不応期        
        self.rel_refact = np.ones(self.numneu) < 0
        self.rel_elapsedtime = np.zeros(self.numneu)
        self.rel_refacttau = 5
        self.vth_vthref_ratio = np.ones(self.numneu) * self.ab_vth
        """
    
    #信号伝達(メイン処理）
    def propagation(self, act_function):
        #内部電位計算        
        self.sigma = np.zeros(self.numneu)
        self.act_f = act_function
        self.temp = ((self.input - self.vrm) + (self.n - self.vrm)) * self.w
        for i in range(0, self.numneu):
            for j in range(0, self.numneu):
                self.sigma[i] += self.temp[i][j]

        for i in range(0, self.numneu):
            if self.ab_refact[i] == True:
                self.vin[i] = (-(self.vin[i] -self.vrm[i]) + self.sigma[i]) * self.ab_weak / self.tau + self.vin[i]
            else:   
                self.vin[i] = (-(self.vin[i] -self.vrm[i]) + self.sigma[i]) / self.tau + self.vin[i]
        
        #活性化関数
        if self.act_f == "step":
            self.__step_function()
        elif self.act_f == "sigmoid":
            self.__sigmoid_function()
        else:
            pass
        
        for i in range(0, self.numneu):
            for j in range(0, self.numneu):
                self.input[j][i] = self.output[i]

        #ラスタープロット用ー発火情報を記憶しておく
        if self.prop_first == True:
            self.prop_first = False 
            for i in range(0, self.numneu):
                if self.vin[i] > 0:
                    #例えばニューロン4のラスターは上から4番目にプロットされる
                    self.raster[i][self.rasterco] = self.numneu-i
                else:
                    self.raster[i][self.rasterco] = 0
            self.rasterco+=1                
        elif self.rasterco < self.simtime:
            for i in range(0, self.numneu):
                if self.vin[i] > 0:
                    #例えばニューロン4のラスターは上から4番目にプロットされる
                    self.raster[i][self.rasterco] = self.numneu-i
                else:
                    self.raster[i][self.rasterco] = 0
            self.rasterco+=1
        else:
            pass   
        
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
        for i in range(0, self.numneu):
            if self.vin[i] == self.phigh[i]:
                self.output[i] = self.phigh[i]
            else:
                self.output[i] = self.vrm[i]
        
        return 0
    
    def __sigmoid_function(self):
        self.random = np.random.rand(self.numneu)
        self.sigmoid_comparison = 1 / (1 + np.exp(-(self.vin - self.vth) * self.sigmoid_gain))
    
        for i in range(0, self.numneu):
            if self.sigmoid_comparison[i] < 0.1:
               self.sigmoid_comparison[i] = 0
               
        #不応期
        for i in range(0, self.numneu):
            if self.ab_refact[i] == True and self.ab_elapsedtime[i] >= self.ab_refactftime[i]:
                self.ab_refact[i] = False
                self.ab_elapsedtime[i] = 0
                self.vth[i] = self.vth_refer[i]
            elif self.ab_refact[i] == True:
                self.ab_elapsedtime[i] += self.deltatime            
                self.vth[i] = self.ab_vth[i]
            else:
                pass
        
        #内部電位
        for i in range(0, self.numneu):
            if self.f_flag[i] == True and self.phcounter[i] >= self.phightime:
                self.f_flag[i] = False                
                self.vin[i] = self.vrm[i] - 10     
                self.ab_refact[i] = True
                self.phcounter[i] = 0
            elif self.f_flag[i] == True:
                self.vin[i] = self.phigh[i]
                self.phcounter[i] += 1                
            elif self.sigmoid_comparison[i] > self.random[i]:
                self.f_flag[i] = True                
                self.vin[i] = self.phigh[i]
            else:
                self.f_flag[i] = False
        #出力
        for i in range(0, self.numneu):
            if self.f_flag[i] == True:
                self.output[i] = self.phigh[i]
            else:
                self.output[i] = self.vrm[i]
        return 0    
        
    #ノイズ
    def input_bumpnoise(self):
        self.nsource = np.random.rand(self.numneu, self.numneu)
        for i in range(0, self.numneu):
            for j in range(0, self.numneu):
                if  self.nsource[i][j] > self.nth[i][j]:
                    self.input[i][j] += 50
                else:
                    pass
    def input_whitenoise(self):
        self.input += np.random.randn(self.numneu, self.numneu) * 30
        #print(self.input)
    
    def input_test(self):
        self.input[0][0] = 40
        #self.input[1][1] = 40
        