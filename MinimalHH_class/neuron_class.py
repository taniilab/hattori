# -*- coding: utf-8 -*-
"""
Created on Tue May 15 18:53:50 2018

@author: ishida
"""

import numpy as np

class Neuron_HH:
    def __init__(self, N = 1, dt = 0.02, T = 10000, Cm = 1,Vth = -56.2, 
                 gleak = 0.0205, eleak = -70.3, gNa = 56, eNa = 50,
                 gK = 6, eK = -90, gm = 0.075, tau_max = 608, 
                 gtCa = 0.4, eCa = 120,
                 syncp = 1, Pmax = 1, tau_syn = 5.26, esyn = 0, gsyn = 0.025, ratio = 0.5,
                 Iext_amp = 1, noise = 1, D = 1):
        # number of neuron
        self.N = N
        # time step
        self.dt = dt
        # simulation time
        self.T = T
        # all time
        self.Tsteps = np.arange(0, self.T, self.dt)
        # number of time steps
        self.allsteps = len(self.Tsteps)
        
        # membrane capacitance
        self.Cm = Cm
        # threshold voltage
        self.Vth = Vth
        
        # leak
        self.gleak = gleak * np.ones(self.N)
        self.eleak = eleak * np.ones(self.N)
        
        # Na
        self.gNa = gNa * np.ones(self.N)
        self.eNa = eNa * np.ones(self.N)
        
        # K
        self.gK = gK * np.ones(self.N)
        self.eK = eK * np.ones(self.N)
        self.tau_max = tau_max
        self.gm = gm * np.ones(self.N)
        
        # T type Ca
        self.gtCa = gtCa * np.ones(self.N)
        self.eCa = eCa * np.ones(self.N)
        
        # synapse
        self.tau_syn = tau_syn
        self.esyn = esyn * np.ones((self.N, self.N))
        self.gsyn = 0 * np.ones((self.N, self.N)) # gsyn = 0.025 と定義したけどここでは0にしている
        self.gAMPA = 0 * np.ones((self.N, self.N))
        self.gNMDA = 0 * np.ones((self.N, self.N))
        
        # type of synaptic coupling
        self.syncp = syncp
        
        self.Pmax = Pmax
        self.fire_tmp = np.zeros(self.N)
        # amplitude of AMPA / amplitude of NMDA
        self.ratio = ratio
        
        #noise
        self.noise = noise
        self.Inoise = np.zeros((self.N, self.allsteps))
        self.D = D
        self.g = np.random.randn(self.N, self.allsteps)
        
        # firing time
        self.t_ap = -10000 * np.ones((self.N, self.N, 2))
        
        #voltage
        self.V = -65 * np.ones((self.N, self.allsteps))
        self.dV = 0 * np.ones(self.N)
        
        #current
        self.Ileak = 0 * np.ones((self.N, self.allsteps))
        self.INa = 0 * np.ones((self.N, self.allsteps))
        self.IK = 0 * np.ones((self.N, self.allsteps))
        self.Im = 0 * np.ones((self.N, self.allsteps))
        self.ItCa = 0 * np.ones((self.N, self.allsteps))
        
        # synapse current
        self.Isyn = np.zeros((self.N, self.allsteps))
        self.INMDA = np.zeros((self.N, self.allsteps))
        self.IAMPA = np.zeros((self.N, self.allsteps))
        
        self.m = 0.5 * np.ones((self.N, self.allsteps))
        self.h = 0.06 * np.ones((self.N, self.allsteps))
        self.n = 0.5 * np.ones((self.N, self.allsteps))
        self.p = 0.5 * np.ones((self.N, self.allsteps))
        self.u = 0.5 * np.ones((self.N, self.allsteps))
        
        self.alpha_m = 0 * np.ones((self.N, self.allsteps))
        self.beta_m = 0 * np.ones((self.N, self.allsteps))
        self.alpha_h = 0 * np.ones((self.N, self.allsteps))
        self.beta_h = 0 * np.ones((self.N, self.allsteps))
        self.alpha_n = 0 * np.ones((self.N, self.allsteps))
        self.beta_n = 0 * np.ones((self.N, self.allsteps))
        self.p_inf = 0 * np.ones((self.N, self.allsteps))
        self.tau_p = 0 * np.ones((self.N, self.allsteps))
        self.s_inf = 0 * np.ones((self.N, self.allsteps))
        self.u_inf = 0 * np.ones((self.N, self.allsteps))
        self.tau_u = 0 * np.ones((self.N, self.allsteps))
        
        self.dm = 0 * np.ones(self.N)
        self.dh = 0 * np.ones(self.N)
        self.dn = 0 * np.ones(self.N)
        self.dp = 0 * np.ones(self.N)
        self.du = 0 * np.ones(self.N)
        
        # external input current
        self.Iext_amp = Iext_amp
        self.Iext = np.zeros((self.N, self.allsteps))
        self.Iext[0, 10000:20000] = self.Iext_amp 
        #self.Iext[:, 10000:15000]とかのほうが良いのでは
        
        #current step
        self.curstep = 0
    
    def biexp_func(self, t, Pmax, t_rise, t_fall):
        if t < 0:
            return 0
        elif Pmax * (np.exp(- t / t_fall) - np.exp(- t / t_rise)) < 0.00001:
            return 0
        else:
            return Pmax * (np.exp(- t / t_fall) - np.exp(- t / t_rise))
        
    def calc_synaptic_input(self, i):
        # recording present fire time as previous fire time
        if self.Vi[i] > -20 and (self.curstep * self.dt - self.fire_tmp[i]) > 20 and self.curstep * self.dt > 200:
            self.t_ap[i, :, 1] = self.t_ap[i, :, 0]
            self.t_ap[i, :, 0] = self.curstep * self.dt
            self.fire_tmp[i] = self.curstep * self.dt
                    
        # sum of the synaptic current for each neuron
        
        if self.syncp == 1:
            for j in range(0, self.N):
                
                if self.curstep * self.dt > 200:
                    self.gAMPA[i, j] = self.biexp_func(self.curstep * self.dt - self.t_ap[j, i, 0], self.Pmax, 1, 2)
                    self.gNMDA[i, j] = self.biexp_func(self.curstep * self.dt - self.t_ap[j, i, 0], self.ratio * self.Pmax, 10, 150) / (1 + (4.5 / 3.57) * np.exp(- 0.062 * self.Vi))
                    self.gsyn[i, j] = self.gAMPA[i, j] + self.gNMDA[i,j]
                    
                else:
                    self.gsyn[i, j] = 0
       
        # sum        
        for j in range(0, self.N):
            self.IAMPAi[i] += self.gAMPA[i, j] * (self.esyn[i, j] - self.Vi[i])
            self.INMDAi[i] += self.gNMDA[i, j] * (self.esyn[i, j] - self.Vi[i])
            self.Isyni[i] = self.IAMPAi[i] + self.INMDAi[i]
            
            
        self.IAMPA[i, self.curstep] = self.IAMPAi[i]
        self.INMDA[i, self.curstep] = self.INMDAi[i]
        self.Isyn[i, self.curstep] = self.Isyni[i]
            
    
    def propagation(self):
        
        self.Vi = self.V[:, self.curstep]
                
        self.Ileaki = self.Ileak[:, self.curstep]
        self.INai = self.INa[:, self.curstep]
        self.IKi = self.IK[:, self.curstep]
        self.Imi = self.Im[:, self.curstep]
        self.ItCai = self.ItCa[:, self.curstep]
        
        self.Isyni = self.Isyn[:, self.curstep]
        self.IAMPAi = self.IAMPA[:, self.curstep]
        self.INMDAi = self.INMDA[:, self.curstep]
        
        self.Inoisei = self.Inoise[:, self.curstep]
                
        self.mi = self.m[:, self.curstep]
        self.hi = self.h[:, self.curstep]
        self.ni = self.n[:, self.curstep]
        self.pi = self.p[:, self.curstep]
        self.ui = self.u[:, self.curstep]
        
        self.alpha_mi = self.alpha_m[:, self.curstep]
        self.beta_mi = self.beta_m[:, self.curstep]
        self.alpha_hi = self.alpha_h[:, self.curstep]
        self.beta_hi = self.beta_h[:, self.curstep]
        self.alpha_ni = self.alpha_n[:, self.curstep]
        self.beta_ni = self.beta_n[:, self.curstep]
        self.p_infi = self.p_inf[:, self.curstep]
        self.tau_pi = self.tau_p[:, self.curstep]
        self.s_infi = self.s_inf[:, self.curstep]
        self.u_infi = self.u_inf[:, self.curstep]
        self.tau_ui = self.tau_u[:, self.curstep]
        
        for i in range(0, self.N):
            self.calc_synaptic_input(i)
        
        if self.noise == 1:
            self.Inoise[:, self.curstep + 1] = self.D * self.g[:, self.curstep]
        
        # solve a defferential equation
        self.alpha_mi = - 0.32 * (self.Vi - self.Vth - 13) /\
                        (np.exp(- np.clip((self.Vi - self.Vth - 13) / 4, -709, 10000)) - 1)
        self.beta_mi = 0.28 * (self.Vi - self.Vth - 40) /\
                        (np.exp(np.clip((self.Vi - self.Vth - 40) / 5, -709, 10000)) - 1)
        self.alpha_hi = 0.128 * np.exp(- np.clip((self.Vi - self.Vth - 17) / 18, -709, 10000))
        self.beta_hi = 4 / (1 + np.exp(- np.clip((self.Vi - self.Vth - 40) / 5, -709, 10000)))
        self.alpha_ni = - 0.032 * (self.Vi - self.Vth - 15) /\
                        (np.exp(- np.clip((self.Vi - self.Vth - 15) / 5, -709, 10000)) - 1)
        self.beta_ni = 0.5 * np.exp(- np.clip((self.Vi - self.Vth - 10) / 40, -709, 10000))
        self.p_infi = 1 / (1 + np.exp(- np.clip((self.Vi + 35) / 10, -709, 10000)))
        self.tau_pi = self.tau_max /\
                    (3.3 * np.exp(np.clip((self.Vi + 35) / 20, -709, 10000)) +\
                     np.exp(- np.clip((self.Vi + 35) / 20, -709, 10000)))
        self.s_infi = 1 / (1 + np.exp(- np.clip((self.Vi + 2 + 57) / 6.2, -709, 10000)))
        self.u_infi = 1 / (1 + np.exp(np.clip((self.Vi + 2 + 81) / 4, -709, 10000)))
        self.tau_ui = 30.8 + (211.4 + np.exp(np.clip((self.Vi + 2 + 113.2) / 5, -709, 10000))) /\
                    (3.7 * (1 + np.exp(np.clip((self.Vi + 2 + 84) / 3.2, -709, 10000))))
        
        self.Ileaki = self.gleak * (self.eleak - self.Vi)
        self.INai = self.gNa * self.mi**3 * self.hi * (self.eNa - self.Vi)
        self.IKi = self.gK * self.ni**4 * (self.eK - self.Vi)
        self.Imi = self.gm * self.pi * (self.eK - self.Vi)
        self.ItCai = self.gtCa * self.s_infi**2 * self.ui * (self.eCa - self.Vi)

        self.dV = self.dt *\
                (self.Ileaki + self.INai + self.IKi + self.Imi + self.ItCai +\
                 self.Isyni + self.Inoisei + self.Iext[:, self.curstep]) /\
                self.Cm
            
        #いらないと思う (self.curstep * self.dt) < 200　で self.Isyni は常に0では？
        if (self.curstep * self.dt) < 200:
            self.dV -= self.Isyni
       
            
        self.dm = self.dt * (self.alpha_mi * (1 - self.mi) - self.beta_mi * self.mi)
        self.dh = self.dt * (self.alpha_hi * (1 - self.hi) - self.beta_hi * self.hi)
        self.dn = self.dt * (self.alpha_ni * (1 - self.ni) - self.beta_ni * self.ni)
        self.dp = self.dt * (self.p_infi - self.pi) / self.tau_pi
        self.du = self.dt * (self.u_infi - self.ui) / self.tau_ui
                
        self.V[:, self.curstep + 1] = self.Vi + self.dV
        
        self.m[:, self.curstep + 1] = self.mi + self.dm
        self.h[:, self.curstep + 1] = self.hi + self.dh
        self.n[:, self.curstep + 1] = self.ni + self.dn
        self.p[:, self.curstep + 1] = self.pi + self.dp
        self.u[:, self.curstep + 1] = self.ui + self.du
        
        self.Ileak[:, self.curstep] = self.Ileaki
        self.INa[:, self.curstep] = self.INai
        self.IK[:, self.curstep] = self.IKi
        self.Im[:, self.curstep] = self.Imi
        self.ItCa[:, self.curstep] = self.ItCai
        
        #self.Isyn[:, self.curstep] = self.Isyni
        
        self.curstep += 1