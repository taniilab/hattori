"""
Created on Wed May 24 11:37:33 2017

@author: Hattori
"""
# coding: UTF-8
import numpy as np


class Neuron_HH():
    def __init__(self, syncp=1, N=1, dt=0.05, T=1000,Cm=1, Vth=-56.2,
                 eNa=50, gNa=56, eK=-90, gK=6, eL=-70.3, gL=0.0205, gM=0.075,
                 tau_syn=5.26, esyn=0, gsyn=0.025, tau_max=608, eCa=120, gtCa=0.4, glCa=0.0001,
                 gpNa=0,
                 Iext_amp = 0, Pmax_AMPA=0, Pmax_NMDA=0,
                 Iext_num=0, noise=0, ramda=-10, alpha=0.5,
                 beta=0, D=1, ratio=0.5, Mg_conc=4):
        self.set_neuron_palm(syncp, N, dt, T,Cm, Vth,
                 eNa, gNa, eK, gK, eL, gL, gM,
                 tau_syn, esyn, gsyn, tau_max, eCa, gtCa, glCa,
                 gpNa,
                 Iext_amp, Pmax_AMPA, Pmax_NMDA,
                 Iext_num, noise, ramda, alpha,
                 beta, D, ratio, Mg_conc)

    def set_neuron_palm(self, syncp=1, N=1, dt=0.05, T=5000,Cm=1, Vth=-56.2,
                 eNa=50, gNa=56, eK=-90, gK=6, eL=-70.3, gL=0.0205, gM=0.075,
                 tau_syn=5.26, esyn=0, gsyn=0.025, tau_max=608, eCa=120, gtCa=0.4, glCa=0.0001,
                 gpNa=0.1,
                 Iext_amp = 0, Pmax_AMPA=0, Pmax_NMDA=0,
                 Iext_num=0, noise=0, ramda=-10, alpha=0.5,
                 beta=0, D=1, ratio = 0.5, Mg_conc=4):
        # parameters (used by main.py)
        self.parm_dict = {}
        self.ratio = ratio
        
        # type of synaptic coupling
        self.syncp = syncp
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
        self.eCa = eCa * np.ones(self.N)
        self.gtCa = gtCa * np.ones(self.N)
        self.glCa = glCa * np.ones(self.N)
        self.gpNa = gpNa * np.ones(self.N)

        self.V = -65 * np.ones((self.N, self.allsteps))
        self.m = 0.5 * np.ones((self.N, self.allsteps))
        self.h = 0.06 * np.ones((self.N, self.allsteps))
        self.n = 0.5 * np.ones((self.N, self.allsteps))
        self.p = 0.5 * np.ones((self.N, self.allsteps))
        self.u = 0.5 * np.ones((self.N, self.allsteps))
        self.q = 0.5 * np.ones((self.N, self.allsteps))
        self.r = 0.5 * np.ones((self.N, self.allsteps))
        self.alpha_m = 0 * np.ones((self.N, self.allsteps))
        self.alpha_h = 0 * np.ones((self.N, self.allsteps))
        self.alpha_n = 0 * np.ones((self.N, self.allsteps))
        self.alpha_q = 0 * np.ones((self.N, self.allsteps))
        self.alpha_r = 0 * np.ones((self.N, self.allsteps))
        self.beta_m = 0 * np.ones((self.N, self.allsteps))
        self.beta_h = 0 * np.ones((self.N, self.allsteps))
        self.beta_n = 0 * np.ones((self.N, self.allsteps))
        self.beta_q = 0 * np.ones((self.N, self.allsteps))
        self.beta_r = 0 * np.ones((self.N, self.allsteps))
        self.p_inf = 0 * np.ones((self.N, self.allsteps))
        self.tau_p = 0 * np.ones((self.N, self.allsteps))
        self.tau_max = tau_max
        self.s_inf = 0 * np.ones((self.N, self.allsteps))
        self.u_inf = 0 * np.ones((self.N, self.allsteps))
        self.tau_u = 0 * np.ones((self.N, self.allsteps))
        self.INa = 0 * np.ones((self.N, self.allsteps))
        self.IK = 0 * np.ones((self.N, self.allsteps))
        self.Im = 0 * np.ones((self.N, self.allsteps))
        self.Ileak = 0 * np.ones((self.N, self.allsteps))
        self.ItCa = 0 * np.ones((self.N, self.allsteps))
        self.IlCa = 0 * np.ones((self.N, self.allsteps))
        self.INa = 0 * np.ones((self.N, self.allsteps))
        self.IK = 0 * np.ones((self.N, self.allsteps))

        self.k1V = 0 * np.ones(self.N)
        self.k1m = 0 * np.ones(self.N)
        self.k1h = 0 * np.ones(self.N)
        self.k1n = 0 * np.ones(self.N)
        self.k1p = 0 * np.ones(self.N)
        self.k1u = 0 * np.ones(self.N)
        self.k1q = 0 * np.ones(self.N)
        self.k1r = 0 * np.ones(self.N)

        # connection relationship
        self.W = np.ones((self.N, self.N))

        # synaptic current
        self.Isyn = np.zeros((self.N, self.allsteps))
        self.INMDA = np.zeros((self.N, self.allsteps))
        self.IAMPA = np.zeros((self.N, self.allsteps))
        self.Isyn_hist = np.zeros((self.N, self.N, 5))

        # synaptic conductance
        self.gsyn = np.zeros((self.N, self.N))
        self.gNMDA = np.zeros((self.N, self.N))
        self.gAMPA = np.zeros((self.N, self.N))

        # synaptic reversal potential
        self.esyn = esyn * np.ones((self.N, self.N))
        self.tau_syn = tau_syn
        self.Mg_conc = Mg_conc

        # external input
        self.Iext_amp = Iext_amp
        self.Iext = np.zeros((self.N, self.allsteps))
        self.Iext[0, 10000:20000] = -self.Iext_amp
        self.Iext[0, 30000:40000] = self.Iext_amp
        self.Iext[0, 50000:60000] = 2 * self.Iext_amp


        # self.Iext = self.Iext_amp * np.ones((self.N, self.allsteps))

        """
        self.Iext_co = 0
        self.Iext_amp = Iext_amp
        self.Iext_width = Iext_width
        self.Iext_duty = Iext_duty
        whIleake self.Iext_co < Iext_num:
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
        self.t_ap = -10000 * np.ones((self.N, self.N, 2))

        # current step
        self.curstep = 0

        # noise palameter
        self.noise = noise
        self.Inoise = np.zeros((self.N, self.allsteps))

        self.dn = np.zeros((self.N, self.allsteps))
        self.ramda = ramda
        self.alpha = alpha
        self.beta = beta
        self.D = D
        self.g = np.random.randn(self.N, self.allsteps)

        # chemical synapse and alpha function
        self.Pmax_AMPA = Pmax_AMPA
        self.Pmax_NMDA = Pmax_NMDA

        self.fire_tmp = np.zeros(self.N)

    def alpha_function(self, t):
        if t < 0:
            return 0
        elif ((self.Pmax * t/self.tau_syn*0.1) *
              np.exp(-t/self.tau_syn*0.1)) < 0.00001:
            return 0
        else:
            return (self.Pmax * t/self.tau_syn) * np.exp(-t/self.tau_syn)

    def biexp_func(self, t, Pmax, t_rise, t_fall):
        if t < 0:
            return 0
        elif Pmax*(np.exp(-t/t_fall) - np.exp(-t/t_rise)) < 0.00001:
            return 0
        else:
            return Pmax*(np.exp(-t/t_fall) - np.exp(-t/t_rise))

    def calc_synaptic_input(self, i):
        # recording fire time
        if self.Vi[i] > -20 and (self.curstep * self.dt - self.fire_tmp[i]) > 20 and self.curstep * self.dt > 200:
            self.t_ap[i, :, 1] = self.t_ap[i, :, 0]
            self.t_ap[i, :, 0] = self.curstep * self.dt
            self.fire_tmp[i] = self.curstep * self.dt
        # sum of the synaptic current for each neuron
        if self.syncp == 1:
            pass
        elif self.syncp == 2:
            pass
        elif self.syncp == 3:
            pass

        # alpha function
        elif self.syncp == 4:
            for j in range(0, self.N):
                self.gsyn[i, j] =\
                    (self.alpha_function(self.curstep*self.dt -
                                         self.t_ap[j, i, 0]) +
                     self.alpha_function(self.curstep*self.dt -
                                         self.t_ap[j, i, 1]))
        # NMDA & AMPA
        elif self.syncp == 5:
            for j in range(0, self.N):

                if self.curstep*self.dt > 200:
                    # cite from "neuronal noise"
                    self.gNMDA[i, j] = self.biexp_func(self.curstep*self.dt - self.t_ap[j, i, 0], self.Pmax_NMDA, 20, 125) / (1 + (self.Mg_conc/3.57)*np.exp(-0.062*self.Vi))
                    self.gAMPA[i, j] = self.biexp_func(self.curstep*self.dt - self.t_ap[j, i, 0], self.Pmax_AMPA, 0.8, 5)
                    self.gsyn[i, j] = self.gNMDA[i, j] + self.gAMPA[i, j]
                else:                  
                    self.gsyn[i, j] = 0

        # sum
        for j in range(0, self.N):
            self.INMDAi[i] += self.gNMDA[i, j] * (self.esyn[i, j] - self.Vi[i])
            self.IAMPAi[i] += self.gAMPA[i, j] * (self.esyn[i, j] - self.Vi[i])
            self.Isyni[i] = self.INMDAi[i] + self.IAMPAi[i]

    # countermeasures againstthe exp overflow
    def sigmoid(self, x):
        sigmoid_range = 34.538776394910684

        if x <= -sigmoid_range:
            return 1e-15
        if x >= sigmoid_range:
            return 1.0 - 1e-15

        return 1.0 / (1.0 + np.exp(-x))

    # one step processing
    def propagation(self):
        # slice

        self.Vi = self.V[:, self.curstep]
        self.mi = self.m[:, self.curstep]
        self.hi = self.h[:, self.curstep]
        self.ni = self.n[:, self.curstep]
        self.pi = self.p[:, self.curstep]
        self.ui = self.u[:, self.curstep]
        self.qi = self.u[:, self.curstep]
        self.ri = self.u[:, self.curstep]
        self.alpha_mi = self.alpha_m[:, self.curstep]
        self.beta_mi = self.beta_m[:, self.curstep]
        self.alpha_hi = self.alpha_h[:, self.curstep]
        self.beta_hi = self.beta_h[:, self.curstep]
        self.alpha_ni = self.alpha_n[:, self.curstep]
        self.beta_ni = self.beta_n[:, self.curstep]
        self.alpha_qi = self.alpha_n[:, self.curstep]
        self.beta_qi = self.beta_n[:, self.curstep]
        self.alpha_ri = self.alpha_n[:, self.curstep]
        self.beta_ri = self.beta_n[:, self.curstep]
        self.p_infi = self.p_inf[:, self.curstep]
        self.tau_pi = self.tau_p[:, self.curstep]
        self.s_infi = self.s_inf[:, self.curstep]
        self.u_infi = self.u_inf[:, self.curstep]
        self.tau_ui = self.tau_u[:, self.curstep]
        self.Isyni = self.Isyn[:, self.curstep]
        self.INMDAi = self.INMDA[:, self.curstep]
        self.IAMPAi = self.IAMPA[:, self.curstep]
        self.INai = self.INa[:, self.curstep]
        self.IKi = self.IK[:, self.curstep]
        self.Ileaki = self.Ileak[:, self.curstep]
        self.Imi = self.Im[:, self.curstep]
        self.ItCai = self.ItCa[:, self.curstep]
        self.IlCai = self.IlCa[:, self.curstep]
        self.Inoisei = self.Inoise[:, self.curstep]

        # calculate synaptic input
        for i in range(0, self.N):
            self.calc_synaptic_input(i)

        # Noise
        # 1 : gaussian white
        # 2 : Ornstein-Uhlenbeck process
        # 3 : sin wave
        if self.noise == 1:
            self.Inoise[:, self.curstep+1] = self.D * self.g[:, self.curstep]

        elif self.noise == 2:
            self.Inoise[:, self.curstep+1] = (self.Inoisei +
                                              (-self.alpha * (self.Inoisei - self.beta) * self.dt
                                               +self.D * self.g[:, self.curstep]))
        elif self.noise == 3:
            self.Inoise[:, self.curstep+1] = (self.alpha *
                                              np.sin(np.pi *
                                                     self.curstep/(1000/self.dt)))

        else:
            pass

        # solve a defferential equation
        # sodium
        self.alpha_mi = ((-0.32) * (self.Vi - self.Vth - 13) /
                         (np.exp(-np.clip((self.Vi-self.Vth-13)/4, -709, 10000))-1))
        self.beta_mi = (0.28 * (self.Vi - self.Vth - 40) /
                        (np.exp(np.clip((self.Vi-self.Vth-40) / 5, -709, 10000)) - 1))
        self.alpha_hi = 0.128 * np.exp(-np.clip((self.Vi-self.Vth-17)/18, -709, 10000))
        self.beta_hi = 4 * self.sigmoid((self.Vi-self.Vth-40) / 5)
        #potassium
        self.alpha_ni = (-0.032 * (self.Vi-self.Vth-15) /
                         (np.exp(-(self.Vi-self.Vth-15) / 5) - 1))
        self.beta_ni = 0.5 * np.exp(-(self.Vi-self.Vth-10) / 40)
        #T type calcium
        self.p_infi = 1 / (1 + np.exp(-(self.Vi+35) / 10))
        self.tau_pi = (self.tau_max /
                       (3.3 * np.exp((self.Vi+35) / 20) +
                        np.exp(-(self.Vi+35) / 20)))
        self.s_infi = self.sigmoid((self.Vi+2+57) / 6.2)
        self.u_infi = self.sigmoid(-(self.Vi+2+81) / 4)
        self.tau_ui = (30.8 + (211.4 + np.exp(np.clip((self.Vi+2+113.2)/5, -709, 10000))) /
                       (3.7 * (1 + np.exp(np.clip((self.Vi+2+84)/3.2, -709, 10000)))))
        # L type calcium
        self.alpha_qi = (0.055 * (- 27 - self.Vi) /
                         (np.exp(-np.clip((-27 - self.Vi), -709, 10000)) - 1))
        self.beta_qi = 0.94 * np.exp((-75 - self.Vi) / 17)
        self.alpha_ri = 0.000457 * np.exp((-13 - self.Vi) / 50)
        self.beta_ri = 0.0065 / (np.exp(-np.clip((-15 - self.Vi)/28, -709, 10000)) + 1)

        self.INai = self.gNa * self.mi**3 * self.hi * (self.eNa - self.Vi)
        self.IKi = self.gK * self.ni**4 * (self.eK - self.Vi)
        self.Ileaki = self.gL * (self.eL - self.Vi)
        self.Imi = self.gM * self.pi * (self.eK - self.Vi)
        self.ItCai = self.gtCa * self.s_infi**2 * self.ui * (self.eCa - self.Vi)
        self.IlCai = self.glCa * self.qi**2 * self.ri * (self.eCa - self.Vi)

        self.k1V = (self.INai +
                    self.IKi +
                    self.Ileaki +
                    self.Imi +
                    self.ItCai +
                    self.IlCai +
                    self.Isyni +
                    self.Iext[:, self.curstep] +
                    self.Inoisei)

        if (self.curstep*self.dt) < 200:
            self.k1V -= self.Isyni
        self.k1m = self.alpha_mi * (1-self.mi) - self.beta_mi * self.mi
        self.k1h = self.alpha_hi * (1-self.hi) - self.beta_hi * self.hi
        self.k1n = self.alpha_ni * (1-self.ni) - self.beta_ni * self.ni
        self.k1p = (self.p_infi - self.pi) / self.tau_pi
        self.k1u = (self.u_infi - self.ui) / self.tau_ui
        self.k1q = self.alpha_qi * (1-self.qi) - self.beta_qi * self.qi
        self.k1r = self.alpha_ri * (1-self.ri) - self.beta_ri * self.ri

        # first order Euler method
        self.V[:, self.curstep+1] = self.Vi + self.k1V * self.dt
        self.m[:, self.curstep+1] = self.mi + self.k1m * self.dt
        self.h[:, self.curstep+1] = self.hi + self.k1h * self.dt
        self.n[:, self.curstep+1] = self.ni + self.k1n * self.dt
        self.p[:, self.curstep+1] = self.pi + self.k1p * self.dt
        self.u[:, self.curstep+1] = self.ui + self.k1u * self.dt
        self.q[:, self.curstep+1] = self.qi + self.k1q * self.dt
        self.r[:, self.curstep+1] = self.ri + self.k1r * self.dt
        self.V[:, self.curstep+1] = self.Vi + self.k1V * self.dt


        # update original array
        self.INa[:, self.curstep] = self.INai
        self.IK[:, self.curstep] = self.IKi
        self.Im[:, self.curstep] = self.Imi
        self.Ileak[:, self.curstep] = self.Ileaki
        self.ItCa[:, self.curstep] = self.ItCai
        self.s_inf[:, self.curstep] = self.s_infi
        self.u_inf[:, self.curstep] = self.u_infi
        self.tau_u[:, self.curstep] = self.tau_ui
        self.IlCa[:, self.curstep] =  self.IlCai
        self.Inoise[:, self.curstep] = self.Inoisei
        self.curstep += 1
