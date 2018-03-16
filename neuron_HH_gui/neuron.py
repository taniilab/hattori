"""
Created on Wed May 24 11:37:33 2017

@author: Hattori
"""
# coding: UTF-8
import numpy as np


class Neuron_HH():
    # constructor
    # 0.02
    def __init__(self, Syncp=1, N=1, dt=0.05, T=5000,Cm=1, Vth=-56.2,
                 eNa=50, gNa=56, eK=-90, gK=6, eL=-70.3, gL=0.0205, gM=0.075,
                 tau_Syn=5.26, eSyn=0, gSyn=0.025,tau_max=608,
                 Iext_amp = 0, Pmax=0,
                 Iext_num=0, noise=0, ramda=-10, alpha=0.5,
                 beta=0, D=1):
        self.set_neuron_palm(Syncp, N, dt, T,Cm, Vth,
                 eNa, gNa, eK, gK, eL, gL, gM,
                 tau_Syn, eSyn, gSyn, tau_max,
                 Iext_amp, Pmax,
                 Iext_num, noise, ramda, alpha,
                 beta, D)

    def set_neuron_palm(self, Syncp=1, N=1, dt=0.05, T=5000,Cm=1, Vth=-56.2,
                 eNa=50, gNa=56, eK=-90, gK=6, eL=-70.3, gL=0.0205, gM=0.075,
                 tau_Syn=5.26, eSyn=0, gSyn=0.025, tau_max=608,
                 Iext_amp = 0, Pmax=0,
                 Iext_num=0, noise=0, ramda=-10, alpha=0.5,
                 beta=0, D=1):
        # parameters (used by main.py)
        self.parm_dict = {}

        # type of synaptic coupling
        self.Syncp = Syncp
        # number of neuron
        self.N = N
        # time step
        self.dt = dt
        # simulation time
        self.T = T
        # all time
        self.Tsteps = np.arange(0, self.T, self.dt)
        # number of time step
        self.allsteps = len(self.Tsteps)

        # HH model
        self.Cm = Cm
        self.Vth = Vth
        
        self.eNa = eNa * np.ones(self.N)
        self.gNa = gNa * np.ones(self.N)
        self.eK = eK * np.ones(self.N) 
        self.gK = gK * np.ones(self.N)    
        self.eL = eL * np.ones(self.N)        
        self.gL = gL * np.ones(self.N)        
        self.gM = gM * np.ones(self.N)

        self.V = -65 * np.ones((self.N, self.allsteps))
        self.m = 0.5 * np.ones((self.N, self.allsteps))
        self.h = 0.06 * np.ones((self.N, self.allsteps))
        self.n = 0.5 * np.ones((self.N, self.allsteps))
        self.p = 0.5 * np.ones((self.N, self.allsteps))
        self.alpha_m = 0 * np.ones((self.N, self.allsteps))
        self.alpha_h = 0 * np.ones((self.N, self.allsteps))
        self.alpha_n = 0 * np.ones((self.N, self.allsteps))
        self.beta_m = 0 * np.ones((self.N, self.allsteps))
        self.beta_h = 0 * np.ones((self.N, self.allsteps))
        self.beta_n = 0 * np.ones((self.N, self.allsteps))
        self.p_inf = 0 * np.ones((self.N, self.allsteps))
        self.tau_p = 0 * np.ones((self.N, self.allsteps))
        self.tau_max = tau_max

        self.k1V = 0 * np.ones(self.N)
        self.k1m = 0 * np.ones(self.N)
        self.k1h = 0 * np.ones(self.N)
        self.k1n = 0 * np.ones(self.N)

        # connection relationship
        self.W = np.zeros((self.N, self.N))

        # synaptic current
        self.Isyn = np.zeros((self.N, self.allsteps))
        self.Isyn_hist = np.zeros((self.N, self.N, 5))

        # synaptic conductance
        self.gSyn = gSyn * np.ones((self.N, self.N))
        # synaptic reversal potential
        self.eSyn = eSyn * np.ones((self.N, self.N))
        self.tau_Syn = tau_Syn
        # external input
        self.Iext_amp = Iext_amp
        self.Iext = self.Iext_amp * np.ones((self.N, self.allsteps))
      
        """
        self.Iext_co = 0
        self.Iext_amp = Iext_amp
        self.Iext_width = Iext_width
        self.Iext_duty = Iext_duty
        while self.Iext_co < Iext_num:
            if self.Iext_duty == 0:
                self.Iext[0, (1000/self.dt):(1500/self.dt)] = Iext_amp
            else:
                self.iext_tmp1 = 0 + int(self.Iext_co*Iext_width*(1+Iext_duty)/self.dt)
                self.iext_tmp2 = int(Iext_width*Iext_duty / self.dt + self.iext_tmp1)
                self.iext_tmp3 = int(1 + self.iext_tmp2)
                self.iext_tmp4 = int(Iext_width / self.dt + self.iext_tmp3)
                self.Iext[0, self.iext_tmp1:self.iext_tmp2] = 0
                self.Iext[0, self.iext_tmp3:self.iext_tmp4] = Iext_amp
            self.Iext_co += 1
        """
        
        # firing time
        self.t_ap = -100 * np.ones((self.N, self.N, 2))

        # current step
        self.curstep = 0

        # noise palameter
        self.noise = noise
        self.n = np.zeros((self.N, self.allsteps))

        self.dn = np.zeros((self.N, self.allsteps))
        self.ramda = ramda
        self.alpha = alpha
        self.beta = beta
        self.D = D
        self.g = np.random.randn(self.N, self.allsteps)

        # chemical synapse and alpha function
        self.Pmax = Pmax

        self.fire_tmp = np.zeros(self.N)

    def alpha_function(self, t):
        if t < 0:
            return 0
        elif ((self.Pmax * t/self.tau_Syn*0.1) *
              np.exp(-t/self.tau_Syn*0.1)) < 0.001:
            return 0
        else:
            return (self.Pmax * t/self.tau_Syn) * np.exp(-t/self.tau_Syn)

    def calc_synaptic_input(self, i):
        # recording fire time
        if self.Vi[i] > self.Vth and (self.curstep *
                                      self.dt - self.fire_tmp[i]) > 10:
            self.t_ap[i, :, 1] = self.t_ap[i, :, 0]
            self.t_ap[i, :, 0] = self.curstep * self.dt
            self.fire_tmp[i] = self.curstep * self.dt

        # sum of the synaptic current for each neuron
        if self.Syncp == 1:
            pass

        # alpha function
        elif self.Syncp == 4:
            for j in range(0, self.N):
                self.gSyn[i, j] =\
                    (self.alpha_function(self.curstep*self.dt -
                                         self.t_ap[j, i, 0]) +
                     self.alpha_function(self.curstep*self.dt -
                                         self.t_ap[j, i, 1]))

        # summation
        for j in range(0, self.N):
            self.Isyni[i] +=\
                      (self.W[i, j] * self.gSyn[i, j] *
                           (self.eSyn[i, j] - self.Vi[i]))

    # one step processing
    def propagation(self):
        # slice
        self.Vi = self.V[:, self.curstep]
        self.mi = self.m[:, self.curstep]
        self.hi = self.h[:, self.curstep]
        self.ni = self.n[:, self.curstep]
        self.pi = self.p[:, self.curstep]
        self.alpha_mi = self.alpha_m[:, self.curstep]
        self.beta_mi = self.beta_m[:, self.curstep]
        self.alpha_hi = self.alpha_h[:, self.curstep]
        self.beta_hi = self.beta_h[:, self.curstep]
        self.alpha_ni = self.alpha_n[:, self.curstep]
        self.beta_ni = self.beta_n[:, self.curstep]
        self.p_infi = self.p_inf[:, self.curstep]
        self.tau_pi = self.tau_p[:, self.curstep]
        self.Isyni = self.Isyn[:, self.curstep]
        self.ni = self.n[:, self.curstep]
        self.dni = self.dn[:, self.curstep]

        # calculate synaptic input
        for i in range(0, self.N):
            self.calc_synaptic_input(i)

        # Noise
        # 1 : gaussian white
        # 2 : Ornstein-Uhlenbeck process
        # 3 : sin wave
        if self.noise == 1:
            self.n[:, self.curstep+1] = self.D * self.g[:, self.curstep]

        elif self.noise == 2:
            self.n[:, self.curstep+1] = (self.ni +
                                         (-self.alpha * (self.ni - self.beta) +
                                          self.D * self.g[:, self.curstep]) *
                                         self.dt)
        elif self.noise == 3:
            self.n[:, self.curstep+1] = (self.alpha *
                                         np.sin(np.pi *
                                                self.curstep/(1000/self.dt)))

        else:
            pass

        # solve a defferential equation
        self.alpha_mi = ((-0.32) * (self.Vi - self.Vth - 13) /
                         (np.exp(-(self.Vi-self.Vth-13)/4)-1))
        self.beta_mi = (0.28 * (self.Vi - self.Vth - 40) /
                        (np.exp((self.Vi-self.Vth-40) / 5) - 1))
        self.alpha_hi = 0.128 * np.exp(-(self.Vi-self.Vth-17) / 18)
        self.beta_hi = 4 / (1 + np.exp(-(self.Vi-self.Vth-40) / 5))
        self.alpha_ni = (-0.032 * (self.Vi-self.Vth-15) /
                         (np.exp(-(self.Vi-self.Vth-15) / 5) - 1))
        self.beta_ni = 0.5 * np.exp(-(self.Vi-self.Vth-10) / 40)
        self.p_infi = 1 / (1 + np.exp(-(self.Vi+35) / 10))
        self.tau_pi = (self.tau_max /
                       (3.3 * np.exp((self.Vi+35) / 20) +
                        np.exp(-(self.Vi+35) / 20)))

        self.k1V = (self.gK * self.ni**4 * (self.eK - self.Vi) +
                    self.gNa * self.mi**3 * self.hi * (self.eNa - self.Vi) +
                    self.gL * (self.eL - self.Vi) +
                    self.gM * self.pi * (self.eK - self.Vi) +
                    self.Isyni +
                    self.Iext[:, self.curstep] +
                    self.ni)
        self.k1m = self.alpha_mi * (1-self.mi) - self.beta_mi * self.mi
        self.k1h = self.alpha_hi * (1-self.hi) - self.beta_hi * self.hi
        self.k1n = self.alpha_ni * (1-self.ni) - self.beta_ni * self.ni
        self.k1p = (self.p_infi - self.pi) / self.tau_pi

        # first order Euler method
        self.V[:, self.curstep+1] = self.Vi + self.k1V * self.dt
        self.m[:, self.curstep+1] = self.mi + self.k1m * self.dt
        self.h[:, self.curstep+1] = self.hi + self.k1h * self.dt
        self.n[:, self.curstep+1] = self.ni + self.k1n * self.dt
        self.p[:, self.curstep+1] = self.pi + self.k1p * self.dt
        self.curstep += 1
