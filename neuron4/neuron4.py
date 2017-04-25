# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 13:45:42 2016

@author: aromagedon

v1.xx 畳み込み型
v2.xx　積分発火型初期版
v3.01　パラメータ設定関数（再プロット用）を追加
v3.02 発火判定変数(ラスタープロット用)を追加
v4.xx チャネル概念（電流、コンダクタンス）を追加
"""
import numpy as np
from numpy.random import *
import matplotlib.pyplot as plt
import itertools as ite

class Neuron4():
    
    #コンストラクタ
    def __init__(self, num_neurons, vth, vrm, tau_mem, deltatime, phightime, phigh, e_syn, def_w, sigmoid_gain, ab_refactftime, ab_weak, simtime):       
        self.set_neuron_palameter(num_neurons, vth, vrm, tau_mem, deltatime, phightime, phigh, e_syn, def_w, sigmoid_gain, ab_refactftime, ab_weak, simtime)
        
    #パラメータ設定    
    def set_weight(self, w):
        for i, j in ite.product(range(self.numneu), range(self.numneu)):
            self.w[i][j] = w[i][j]
        return self.w

    def set_neuron_palameter(self, num_neurons, vth, vrm, tau_mem, deltatime, phightime, phigh, e_syn, def_w, sigmoid_gain, ab_refactftime, ab_weak, simtime):
        
        #ニューロン数
        self.numneu = num_neurons
        #スレッショルド電圧        
        self.vth = np.ones(self.numneu) * vth
        self.vth_refer = np.ones(self.numneu) * vth
        #静止膜電位        
        self.vrm = np.ones(self.numneu) * vrm
        #膜電位時定数
        self.tau_mem = tau_mem / deltatime
        #単位時間        
        self.deltatime = deltatime
        #発火時間        
        self.phightime = phightime /deltatime
        self.phcounter = np.zeros(self.numneu)
        self.phigh = np.ones(self.numneu) * phigh
        self.f_flag = np.ones(self.numneu) < 0
        #膜電位(tempはitot計算用)
        self.vin = np.zeros(self.numneu) + self.vrm
        self.vin_temp = np.zeros(self.numneu) + self.vrm
        #シナプス平衡電位
        self.e_syn = np.ones((self.numneu, self.numneu)) * e_syn
        #シナプスコンダクタンス
        self.g_syn = np.ones((self.numneu, self.numneu)) * 0.1
        #シナプス電流
        self.i_syn = np.zeros(self.numneu)
        #入力電流総和
        self.i_tot = np.zeros(self.numneu)
        #入力インピーダンス
        self.r_in = np.ones(self.numneu)
        #ノイズ？        
        self.n = np.zeros((self.numneu, self.numneu)) + self.vrm
        #シナプス結合荷重        
        self.w = np.ones((self.numneu, self.numneu)) * def_w
        self.i_tot = np.zeros(self.numneu)
        self.temp = np.zeros(self.numneu)
        self.sigmoid_comparison = np.zeros(self.numneu)
        self.sigmoid_gain = sigmoid_gain
        self.simtime = simtime /deltatime
        #ラスタープロット用（後で確認）        
        self.rasterbuf = np.zeros((self.numneu, self.simtime))
        self.raster = np.zeros((self.numneu, self.simtime))
        self.rasterco = 0
        #なんだっけこれ
        self.prop_first = True
        
        #絶対不応期
        self.ab_vth = np.ones(self.numneu) * 10000
        self.ab_refact = np.ones(self.numneu) < 0
        self.ab_refactftime = np.ones(self.numneu) * ab_refactftime
        self.ab_elapsedtime = np.zeros(self.numneu)
        self.ab_weak = ab_weak

        #ノイズ用
        self.nsource = np.zeros((self.numneu, self.numneu)) 
        self.nth = np.ones((self.numneu, self.numneu)) * 0.9
        
    #信号伝達(メイン処理）

    def propagation(self, act_function):
        #内部電位計算        
        self.i_tot = np.zeros(self.numneu)
        self.act_f = act_function
        #self.g_syn = np.zeros((self.numneu,self.numneu))
        
                
        self.vin_temp = self.vin
        self.vin_temp = self.vin_temp[:, np.newaxis]#縦ベクトル化
        self.temp = (self.g_syn * (self.e_syn - self.vin_temp)) * self.w
        #シナプス入力の総和にノイズを添付
        self.i_tot = np.sum(self.temp, axis=1) + np.random.randn(self.numneu) * 10


        for i in range(0, self.numneu):
            if self.ab_refact[i] == True:
                self.vin[i] = (-(self.vin[i] -self.vrm[i]) + self.r_in[i] * self.i_tot[i]) * self.ab_weak / self.tau_mem + self.vin[i]
            else:    
                self.vin[i] = (-(self.vin[i] -self.vrm[i]) + self.r_in[i] * self.i_tot[i]) / self.tau_mem + self.vin[i]

        
        #活性化関数
        if self.act_f == "step":
            self.__step_function()
        elif self.act_f == "sigmoid":
            self.__sigmoid_function()
        else:
            pass
        
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
        
        
        return self.vin
   
    #活性化関数
    def __step_function(self):     
        return 0
    
    def __sigmoid_function(self):
        self.random = np.random.rand(self.numneu)
        self.sigmoid_comparison = 1 / (1 + np.exp(-(self.vin - self.vth) * (self.sigmoid_gain / self.deltatime)))
    

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
                
        #発火ニューロンー他ニューロン間のコンダクタンス
        for j in range(0, self.numneu):
            if self.f_flag[j] == True:
                self.g_syn[:, j] = 1
            else:
                self.g_syn[:, j] = 0.01
                
        return 0    
     
    #以下ごみ
    #ノイズ
    def input_bumpnoise(self):
        self.nsource = np.random.rand(self.numneu, self.numneu)
        for i in range(0, self.numneu):
            for j in range(0, self.numneu):
                if  self.nsource[i][j] > self.nth[i][j]:
                    self.v_in[i][j] += 50
                else:
                    pass
    def input_whitenoise(self):
        self.v_in += np.random.randn(self.numneu, self.numneu) * 10
        #print(self.v_in)
    
    def input_test(self):
        self.v_in[0][0] = 40
        #self.v_in[1][1] = 40
        
