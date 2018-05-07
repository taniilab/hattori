# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 13:45:42 2016

@author: aromagedon
単位は、s, mS, cm**2, mVなどで入力

v1.xx 畳み込み型
v2.xx　積分発火型初期版
v3.01　パラメータ設定関数（再プロット用）を追加
v3.02 発火判定変数(ラスタープロット用)を追加
v4.xx チャネル概念（電流、コンダクタンス）を追加
v5.00 ホジキンハクスレー+ノイズ型
v6.00 メイン関数を並列処理化
"""
import numpy as np
from numpy.random import *
import time
import pandas as pd
import logging

#scaling
u_to_m = 10**(-3)
n_to_m = 10**(-6)
p_to_m = 10**(-9)
u_to_c = 10**(-4)

class Neuron6():
    global m_to
    global u_to
    global n_to
    global p_to
    global u_to_c
    
    #コンストラクタ
    def __init__(self, process=0, deltatime=0.003, simtime=3000, numneu=2, vth=-62.14, cm=1, gleak=1.43, eleak=-70, gna=50, gkd=6.0, gm=0.097, τmax=932.2, ena=50, ek=-90, esyn=0, τs=3, pmax=0.001, oua=1, oud=20, iext=0, mean=0, variance=1):        
        self.set_neuron_palameter(process, deltatime, simtime, numneu, vth, cm, gleak, eleak, gna, gkd, gm, τmax, ena, ek, esyn, τs, pmax, oua, oud, iext, mean, variance)
        
    def set_neuron_palameter(self, process=0, deltatime=0.01, simtime=10, numneu=1, vth=-62.14, cm=1, gleak=1.43, eleak=-50, gna=50, gkd=6.0, gm=0.097, τmax=932.2, ena=50, ek=-90, esyn=0, τs=2, pmax=1, oua=0, oud=1, iext=0, mean=0, variance=1):
        #process number
        self.process = process
        #time step        
        self.deltatime = deltatime
        #simulation time
        self.simtime = simtime
        #history of simulation time
        self.tmhist = np.arange(0, self.simtime, self.deltatime)
        #number of neuron
        self.numneu = numneu
        #membrane potential
        self.vin = np.ones((self.numneu, len(self.tmhist))) * -60      
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
        self.cm = np.ones((self.numneu, len(self.tmhist))) * cm
        #leak conductance
        self.gleak = np.ones((self.numneu, len(self.tmhist))) * gleak * n_to_m
        #leak reversal potential
        self.eleak =  np.ones((self.numneu, len(self.tmhist))) * eleak    
        #maximul sodium conductance
        self.gna = np.ones((self.numneu, len(self.tmhist))) * gna
        #maximul potassium conductance
        self.gkd = np.ones((self.numneu, len(self.tmhist))) * gkd
        #maximul sodium conductance for frequency adaptation
        self.gm = np.ones((self.numneu, len(self.tmhist))) * gm
        #synaptic conductance
        self.gsyn = np.zeros((self.numneu, self.numneu))
        #time constant for frequency adaptation
        self.τmax = np.ones((self.numneu, len(self.tmhist))) * τmax        
        #sodium reversal potential
        self.ena = np.ones((self.numneu, len(self.tmhist))) * ena
        #potassium reversal potential
        self.ek = np.ones((self.numneu, len(self.tmhist))) * ek
        #synaptic reversal potential
        self.esyn = np.ones((self.numneu, self.numneu)) * esyn
        #time constant for alpha function
        self.τs = np.ones((self.numneu, self.numneu)) * τs
        #maximul synaptic conductance
        self.pmax = np.ones((self.numneu, self.numneu)) * pmax
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
        self.gsyn = np.zeros((self.numneu, self.numneu))
        self.tfire = np.ones((self.numneu, self.numneu)) * (-1000)
        #noise current
        self.inoise = np.zeros((self.numneu, len(self.tmhist)))
        self.mean = np.ones((self.numneu, len(self.tmhist))) * (np.sin(self.tmhist)**2) * mean
        self.variance =  np.ones((self.numneu, len(self.tmhist)))* variance
        #for Ornstein-Uhlenbeck process
        self.oua = oua
        self.fntime = np.ones((self.numneu, len(self.tmhist))) *  (-1000)
        self.oud = np.ones((self.numneu, len(self.tmhist))) * oud

        #external input current        
        #self.iext = 1 * np.ones((self.numneu, len(self.tmhist))) * (np.sin(self.tmhist)**2)
        self.iext = 0.5 * np.ones((self.numneu, len(self.tmhist))) * 0
        
        #appendix
        self.propinit = True
        self.curstep = 0
        self.currenttime = 0
        self.nextstep = 1
        

        #以下後で整理     
        #ラスタープロット用（後で確認）        
        self.rasterbuf = np.zeros((self.numneu, self.simtime))
        self.raster = np.zeros((self.numneu, self.simtime))
        self.rasterco = 0             
        
        self.palm = 'pmax' + str(pmax)

    #advance time by one step
    def propagation(self):
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
        self.τmaxtp = self.τmax[:, self.curstep]       
        self.enatp = self.ena[:, self.curstep]
        self.ektp = self.ek[:, self.curstep]
        self.mtp = self.m[:, self.curstep]
        self.htp = self.h[:, self.curstep]
        self.ntp = self.n[:, self.curstep]
        self.ptp = self.p[:, self.curstep]
        self.inoisetp = self.inoise[:, self.curstep]
        self.iexttp = self.iext[:, self.curstep]
        self.meantp = self.mean[:, self.curstep]
        self.variancetp = self.variance[:, self.curstep]
        self.mnext = self.m[:, self.nextstep]        
        self.hnext = self.h[:, self.nextstep]
        self.nnext = self.n[:, self.nextstep]
        self.pnext = self.p[:, self.nextstep]
        self.inoisenext = self.inoise[:, self.nextstep]
        self.oudtp = self.oud[:, self.curstep]
        self.fntimetp = self.fntime[:, self.curstep]
        
        ###hodgkin-huxley type equation###
        self.αm = -0.32 * (self.vintp-self.vthtp-13)/(np.exp(-1 * (self.vintp-self.vthtp-13)/4) - 1) 
        self.βm = 0.28 * (self.vintp-self.vthtp-40)/(np.exp((self.vintp-self.vthtp-40)/5) - 1) 
        self.αh = 0.128 * np.exp(-1 * (self.vintp-self.vthtp-17)/18)
        self.βh = 4/(1 + np.exp(-1 * (self.vintp-self.vthtp-40)/5))
        self.αn = -0.032 * (self.vintp-self.vthtp-15)/(np.exp(-1 * (self.vintp-self.vthtp-15)/5) - 1) 
        self.βn = 0.5 * np.exp(-1 * (self.vintp-self.vthtp-10)/40)
        self.pinf = 1/(1+np.exp(-1 * (self.vintp+35)/10))
        self.τp = self.τmaxtp/(3.3*np.exp((self.vintp+35)/20) + np.exp(-1 * (self.vintp+35)/20))
        
        if self.curstep == 0:
            self.mtp[:] = self.αm/(self.αm + self.βm)
            self.htp[:] = self.αh/(self.αh + self.βh)  
            self.ntp[:] = self.αn/(self.αn + self.βn)
        
        #input current        
        self.inatp[:] = self.gnatp * (self.mtp**3) * self.htp * (self.vintp  - self.enatp)
        self.ikdtp[:] = self.gkdtp * (self.ntp**4) * (self.vintp  - self.ektp)
        self.imtp[:] = self.gmtp * self.ptp *(self.vintp  - self.ektp)
        self.gsyn = self.pmax * (self.currenttime - self.tfire) * np.exp(1-(self.currenttime - self.tfire)/self.τs) / self.τs               
        #self.oudtp[:] += 0.002/(1+np.exp(0.2*(self.currenttime - self.fntimetp - 10)))      
        #self.oudtp[:] = 30     
        self.isyntp *= 0
        for j in range(0,self.numneu):
            self.isyntp[:] += self.gsyn[:, j] * (self.esyn[:, j] - self.vintp[:])

        #calculate the derivatives using Euler first order approximation        
        self.vnext[:] = self.deltatime * ((-self.gleaktp * (self.vintp - self.eleaktp) - self.inatp - self.ikdtp + self.isyntp + self.inoisetp + self.iexttp)/self.cmtp) + self.vintp
        self.vnext[1] -= self.deltatime * self.iexttp[1]
        self.mnext[:] = self.deltatime * (self.αm * (1-self.mtp) - self.βm * self.mtp) + self.mtp
        self.hnext[:] = self.deltatime * (self.αh * (1-self.htp) - self.βh * self.htp) + self.htp
        self.nnext[:] = self.deltatime * (self.αn * (1-self.ntp) - self.βn * self.ntp) + self.ntp
        self.pnext[:] = self.deltatime * (self.pinf - self.ptp)/self.τp + self.ptp  
        #self.inoisenext[:] = self.deltatime * (-self.oua * self.inoisetp + self.oudtp * np.random.randn(self.numneu)) + self.inoisetp                
        self.meantp[:] = 0/(1+np.exp(0.5*(self.currenttime - self.fntimetp - 30)))        
        self.inoisenext[:] = self.oudtp * np.random.normal(loc=self.meantp, scale=self.variancetp, size=self.numneu)                
        logging.warning('%d', self.meantp[0])
        logging.warning('%d', self.fntimetp[0])
                    
        
        #recording ignition time 
        for i in range(0, self.numneu):
            if self.vnext[i] > -40:
                
                self.tfire[:, i] = self.currenttime
                #initial fire
                if (self.vin[i, self.curstep-20/self.deltatime:self.curstep] < -30).all():
                    self.fntime[i] = self.currenttime

        self.currenttime += self.deltatime
        self.curstep += 1
        self.nextstep = self.curstep + 1