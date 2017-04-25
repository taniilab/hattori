# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 13:45:42 2016

@author: aromagedon
"単位はミリ"

v1.xx 畳み込み型
v2.xx　積分発火型初期版
v3.01　パラメータ設定関数（再プロット用）を追加
v3.02 発火判定変数(ラスタープロット用)を追加
v4.xx チャネル概念（電流、コンダクタンス）を追加
v5.00 ホジキンハクスレー+ノイズ型
"""
import numpy as np
from numpy.random import *

class Neuron5():
    
    #コンストラクタ
    def __init__(self, deltatime=0.001, simtime=50, numneu=1, vth=-62.14, cm=0.001, gleak=0.00000143, eleak=-70, gna=50, gkd=6.0, gm=0.097, τmax=932.2, ena=50, ek=-90, esyn=0, τs=2, pmax=1, oua=0.5, oub=1, L=0.60, d=0.60):       
        self.set_neuron_palameter(deltatime, simtime, numneu, vth, cm, gleak, eleak, gna, gkd, gm, τmax, ena, ek, esyn, τs, pmax, oua, oub, L , d)
        
    def set_neuron_palameter(self, deltatime=0.001, simtime=10, numneu=1, vth=-62.14, cm=1, gleak=1.43, eleak=-50, gna=50, gkd=6.0, gm=0.097, τmax=932.2, ena=50, ek=-90, esyn=0, τs=2, pmax=1, oua=0, oub=1, L=0.060, d=0.060):
        #time step        
        self.deltatime = deltatime
        #simulation time
        self.simtime = simtime        
        #history of simulation time
        self.tmhist = np.arange(0, self.simtime, self.deltatime)
        #number of neuron
        self.numneu = numneu
        #length of a neuron
        self.L = np.ones((self.numneu, len(self.tmhist))) * L
        #diameter of a neuron
        self.d = np.ones((self.numneu, len(self.tmhist))) * d
        #surface area of a neuron
        self.sufarea = 2 * self.d * np.pi * self.L
        #membrane potential
        self.vin = np.ones((self.numneu, len(self.tmhist))) * -70             
        #threshold voltage(mV)        
        self.vth = np.ones((self.numneu, len(self.tmhist))) * vth
        #sodium current
        self.ina = np.zeros((self.numneu, len(self.tmhist)))
        #potassium current
        self.ikd = np.zeros((self.numneu, len(self.tmhist)))
        #slow potassium current for spike-frequency
        self.im = np.zeros((self.numneu, len(self.tmhist)))
        #E/I PSC(excitatory/inhibitory post synaptic current)
        self.isyn = np.zeros((self.numneu, len(self.tmhist)))
        #noise current
        self.inoise = np.zeros((self.numneu, len(self.tmhist)))
        #membrane conductance(uf)
        self.cm = np.ones((self.numneu, len(self.tmhist))) * cm  * self.sufarea
        #leak conductance
        self.gleak = np.ones((self.numneu, len(self.tmhist))) * gleak
        #leak reversal potential
        self.eleak =  np.ones((self.numneu, len(self.tmhist))) * eleak     
        #maximul sodium conductance
        self.gna = np.ones((self.numneu, len(self.tmhist))) * gna * self.sufarea   
        #maximul potassium conductance
        self.gkd = np.ones((self.numneu, len(self.tmhist))) * gkd * self.sufarea        
        #maximul sodium conductance for frequency adaptation
        self.gm = np.ones((self.numneu, len(self.tmhist))) * gm * self.sufarea
        #synaptic conductance
        self.gsyn = np.zeros((self.numneu, len(self.tmhist)))
        #time constant for frequency adaptation
        self.τmax = np.ones((self.numneu, len(self.tmhist))) * τmax        
        #sodium reversal potential
        self.ena = np.ones((self.numneu, len(self.tmhist))) * ena
        #potassium reversal potential
        self.ek = np.ones((self.numneu, len(self.tmhist))) * ek
        #synaptic reversal potential
        self.esyn = np.ones((self.numneu, len(self.tmhist))) * esyn
        #time constant for alpha function
        self.τs = np.ones((self.numneu, len(self.tmhist))) * τs
        #maximul synaptic conductance
        self.pmax = np.ones((self.numneu, len(self.tmhist))) * pmax
        #channnel variables
        self.αm = np.zeros(self.numneu)
        self.βm = np.zeros(self.numneu)
        self.αh = np.zeros(self.numneu)
        self.βh = np.zeros(self.numneu)        
        self.αn = np.zeros(self.numneu)
        self.βn = np.zeros(self.numneu)
        self.m = np.zeros((self.numneu, len(self.tmhist)))
        self.h = np.zeros((self.numneu, len(self.tmhist)))
        self.n = np.zeros((self.numneu, len(self.tmhist)))
        self.p = np.zeros((self.numneu, len(self.tmhist)))
        self.gsyn = np.zeros((self.numneu, len(self.tmhist)))
        self.tfire = np.ones(self.numneu) * -1000
        #noise current
        self.inoise = np.zeros((self.numneu, len(self.tmhist)))
        #for Ornstein-Uhlenbeck process
        self.oua = oua
        self.oub = oub
        
        #appendix
        self.propinit = True
        self.curstep = 0
        self.currenttime = 0
        self.nextstep = 1
        print(self.sufarea)
        

        #以下後で整理     
        #ラスタープロット用（後で確認）        
        self.rasterbuf = np.zeros((self.numneu, self.simtime))
        self.raster = np.zeros((self.numneu, self.simtime))
        self.rasterco = 0
        #なんだっけこれ
        self.prop_first = True

    #advance time by one step
    def propagation(self):
        if self.propinit == True:
            self.propinit = False
            self.curstep = 0
            self.currenttime = 0
            self.nextstep = 1
        else:
            pass

        #slicing
        self.vintp = self.vin[:, self.curstep]
        self.vnext = self.vin[:, self.nextstep]        
        self.vthtp = self.vth[:, self.curstep]
        self.inatp = self.ina[:, self.curstep]
        self.ikdtp = self.ikd[:, self.curstep]
        self.imtp = self.im[:, self.curstep]
        self.isyntp = self.isyn[:, self.curstep]
        self.cmtp = self.cm[:, self.curstep]
        self.gleaktp = self.gleak[:, self.curstep]
        self.eleaktp = self.eleak[:, self.curstep]     
        self.gnatp = self.gna[:, self.curstep]      
        self.gkdtp = self.gkd[:, self.curstep]      
        self.gmtp = self.gm[:, self.curstep]
        self.gsyntp = self.gsyn[:, self.curstep]
        self.τmaxtp = self.τmax[:, self.curstep]       
        self.enatp = self.ena[:, self.curstep]
        self.ektp = self.ek[:, self.curstep]
        self.esyntp = self.esyn[:, self.curstep]
        self.τstp = self.τs[:, self.curstep]
        self.pmaxtp = self.pmax[:, self.curstep]
        self.mtp = self.m[:, self.curstep]
        self.htp = self.h[:, self.curstep]
        self.ntp = self.n[:, self.curstep]
        self.ptp = self.p[:, self.curstep]
        self.inoisetp = self.inoise[:, self.curstep]
        self.mnext = self.m[:, self.nextstep]        
        self.hnext = self.h[:, self.nextstep]
        self.nnext = self.n[:, self.nextstep]
        self.pnext = self.p[:, self.nextstep]
        self.inoisenext = self.inoise[:, self.nextstep]

        
        #hodgkin-huxley type equation
        self.αm = -0.32 * (self.vintp-self.vthtp-13)/(np.exp(-1 * (self.vintp-self.vthtp-13)/4) - 1) 
        self.βm = 0.28 * (self.vintp-self.vthtp-40)/(np.exp((self.vintp-self.vthtp-40)/5) - 1) 
        self.αh = 0.128 * np.exp(-1 * (self.vintp-self.vthtp-17)/18)
        self.βh = 4/(1 + np.exp(-1 * (self.vintp-self.vthtp-40)/5))
        self.αn = -0.032 * (self.vintp-self.vthtp-15)/(np.exp(-1 * (self.vintp-self.vthtp-15)/5) - 1) 
        self.βn = 0.5 * np.exp(-1 * (self.vintp-self.vthtp-10)/40)
        self.pinf = 1/(1+np.exp(-1 * (self.vintp+35)/10))
        self.τp = self.τmaxtp/(3.3*np.exp((self.vintp+35)/20) + np.exp(-1 * (self.vintp+35)/20))
                       
        self.mnext[:] = self.deltatime * (self.αm * (1-self.mtp) - self.βm * self.mtp) + self.mtp
        self.hnext[:] = self.deltatime * (self.αh * (1-self.htp) - self.βh * self.htp) + self.htp
        self.nnext[:] = self.deltatime * (self.αn * (1-self.ntp) - self.βn * self.ntp) + self.ntp
        self.pnext[:] = self.deltatime * (self.pinf - self.ptp)/self.τp + self.ptp    
        self.gsyntp[:] = self.pmaxtp * (self.currenttime - self.tfire) * np.exp(1-(self.currenttime - self.tfire)/self.τstp) / self.τstp       
        self.inoisenext[:] = -self.oua * self.inoisetp + self.oub * np.random.normal(0, 1)                
        
        self.inatp[:] = self.gnatp * (self.mtp**3) * self.htp * (self.vintp  - self.enatp)
        self.ikdtp[:] = self.gkdtp * (self.ntp**4) * (self.vintp  - self.ektp)
        self.imtp[:] = self.gmtp * self.ptp *(self.vintp  - self.ektp)
        self.isyntp[:] = self.gsyntp * (self.esyntp - self.vintp)
        
        self.vnext[:] = self.deltatime * (-self.gleaktp * (self.vintp - self.eleaktp) - self.inatp - self.ikdtp - self.imtp + self.isyntp + self.inoisetp)/ self.cmtp + self.vintp
        
        
        self.currenttime += self.deltatime
        self.curstep += 1
        self.nextstep = self.curstep + 1
        