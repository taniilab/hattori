"""
Created on Wed May 24 11:37:33 2017

@author: Hattori
"""
# coding: UTF-8
import numpy as np


class Neuron_HH():
    def __init__(self, delay=20, syn_type=1, N=1, dt=0.05, T=1000, Cm=1, Vth=-56.2,
                 eNa=50, gNa=56, eK=-90, gK=5, eL=-70.3, gL=0.0205, gM=0.075,
                 tau_syn=5.26, esyn=0, gsyn=0.025, tau_max=608, eCa=120, gtCa=0.4, glCa=0.0001,
                 gpNa=0, gkCa=0,
                 Iext_amp=0, Pmax_AMPA=0, Pmax_NMDA=0,
                 Iext_num=0, noise_type=0, ramda=-10, alpha=0.5,
                 beta=0, D=1, ratio=0.5, Mg_conc=1,
                 U_SE_AMPA=0.3, U_SE_NMDA=0.03, tau_rise_AMPA=0.9, tau_rise_NMDA=70, tau_rec_AMPA=200, tau_rec_NMDA=200,
                 tau_inact_AMPA=5, tau_inact_NMDA=30):

        self.set_neuron_palm(delay, syn_type, N, dt, T, Cm, Vth,
                             eNa, gNa, eK, gK, eL, gL, gM,
                             tau_syn, esyn, gsyn, tau_max, eCa, gtCa, glCa,
                             gpNa, gkCa,
                             Iext_amp, Pmax_AMPA, Pmax_NMDA,
                             Iext_num, noise_type, ramda, alpha,
                             beta, D, ratio, Mg_conc,
                             U_SE_AMPA, U_SE_NMDA, tau_rise_AMPA, tau_rise_NMDA, tau_rec_AMPA, tau_rec_NMDA,
                             tau_inact_AMPA, tau_inact_NMDA)

    def set_neuron_palm(self, delay=20, syn_type=1, N=1, dt=0.05, T=5000, Cm=1, Vth=-56.2,
                        eNa=50, gNa=56, eK=-90, gK=5, eL=-70.3, gL=0.0205, gM=0.075,
                        tau_syn=5.26, esyn=0, gsyn=0.025, tau_max=608, eCa=120, gtCa=0.4, glCa=0.0001,
                        gpNa=0, gkCa=0,
                        Iext_amp=0, Pmax_AMPA=0, Pmax_NMDA=0,
                        Iext_num=0, noise_type=0, ramda=-10, alpha=0.5,
                        beta=0, D=1, ratio=0.5, Mg_conc=1,
                        U_SE_AMPA=0.3, U_SE_NMDA=0.03, tau_rise_AMPA=0.9, tau_rise_NMDA=70, tau_rec_AMPA=200,
                        tau_rec_NMDA=200, tau_inact_AMPA=5, tau_inact_NMDA=30):

        # parameters (used by main.py)
        self.parm_dict = {}

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
        self.V_intra = -65 * np.ones((self.N, self.allsteps))
        self.V_extra = np.zeros((self.N, self.allsteps))
        for i in range(int(50/self.dt)):
            self.V_extra[0, int(1000 / self.dt)+i] = 30 * np.exp(- i*0.1*self.dt)
            self.V_extra[1, int(1000 / self.dt)+i] = -30 * np.exp(- i*0.1*self.dt)
        self.V = self.V_extra -self.V_intra

        self.k1V = 0 * np.ones(self.N)

        # sodium
        self.INa = 0 * np.ones((self.N, self.allsteps))
        self.eNa = eNa * np.ones(self.N)
        self.gNa = gNa * np.ones(self.N)
        self.m = 0.5 * np.ones((self.N, self.allsteps))
        self.h = 0.06 * np.ones((self.N, self.allsteps))
        self.alpha_m = 0 * np.ones((self.N, self.allsteps))
        self.alpha_h = 0 * np.ones((self.N, self.allsteps))
        self.beta_m = 0 * np.ones((self.N, self.allsteps))
        self.beta_h = 0 * np.ones((self.N, self.allsteps))
        self.k1m = 0 * np.ones(self.N)
        self.k1h = 0 * np.ones(self.N)
        # persistent sodium
        self.IpNa = 0 * np.ones((self.N, self.allsteps))
        self.gpNa = gpNa * np.ones(self.N)
        self.pna = 0.06 * np.ones((self.N, self.allsteps))
        self.alpha_pna = 0 * np.ones((self.N, self.allsteps))
        self.beta_pna = 0 * np.ones((self.N, self.allsteps))
        # potassium
        self.IK = 0 * np.ones((self.N, self.allsteps))
        self.eK = eK * np.ones(self.N)
        self.gK = gK * np.ones(self.N)
        self.n = 0.5 * np.ones((self.N, self.allsteps))
        self.alpha_n = 0 * np.ones((self.N, self.allsteps))
        self.beta_n = 0 * np.ones((self.N, self.allsteps))
        self.k1n = 0 * np.ones(self.N)
        # leak
        self.Ileak = 0 * np.ones((self.N, self.allsteps))
        self.eL = eL * np.ones(self.N)
        self.gL = gL * np.ones(self.N)
        # slow voltage-dependent potassium
        self.Im = 0 * np.ones((self.N, self.allsteps))
        self.gM = gM * np.ones(self.N)
        self.p = 0.5 * np.ones((self.N, self.allsteps))
        self.p_inf = 0 * np.ones((self.N, self.allsteps))
        self.tau_p = 0 * np.ones((self.N, self.allsteps))
        self.tau_max = tau_max
        self.k1p = 0 * np.ones(self.N)

        self.Isyn = np.zeros((self.N, self.allsteps))

        # external input
        self.Iext_amp = Iext_amp
        self.Iext = np.zeros((self.N, self.allsteps))
        self.Iext[0, int(1000 / self.dt):int(1001 / self.dt)] = self.Iext_amp
        self.Iext[0, int(1001 / self.dt):int(1002 / self.dt)] = -self.Iext_amp

        # firing time
        self.t_fire = -10000 * np.ones((self.N, self.N))
        self.t_fire_list = np.zeros((self.N, self.allsteps))

        # compartment model
        self.g_extra = 0
        self.g_intra = 0.1

        # current step
        self.curstep = 0



    # activation functions
    # a / (1 + exp(b(v-c)))
    def activation_func_sigmoid(self, a, b, c, v):
        return a / (1.0 + np.exp(np.clip(b * (v - c), -500, 500)))

    # a * exp(b(v-c))
    def activation_func_exp(self, a, b, c, v):
        return a * np.exp(np.clip(b * (v - c), -500, 500))

    # a(v-b)/(exp(c(v-d))-1)
    def activation_func_ReLUlike(self, a, b, c, d, v):
        return a * (v - b) / (np.exp(np.clip(c * (v - d), -500, 500)) - 1)

    # one step processing
    def propagation(self):
        self.V = self.V_intra - self.V_extra

        # slice
        self.Vi = self.V[:, self.curstep]
        self.V_intrai = self.V_intra[:, self.curstep]
        self.V_extrai = self.V_extra[:, self.curstep]
        # sodium
        self.INai = self.INa[:, self.curstep]
        self.mi = self.m[:, self.curstep]
        self.hi = self.h[:, self.curstep]
        self.alpha_mi = self.alpha_m[:, self.curstep]
        self.beta_mi = self.beta_m[:, self.curstep]
        self.alpha_hi = self.alpha_h[:, self.curstep]
        self.beta_hi = self.beta_h[:, self.curstep]
        # potassium
        self.IKi = self.IK[:, self.curstep]
        self.ni = self.n[:, self.curstep]
        self.alpha_ni = self.alpha_n[:, self.curstep]
        self.beta_ni = self.beta_n[:, self.curstep]
        # slow voltage dependent potassium
        self.Imi = self.Im[:, self.curstep]
        self.pi = self.p[:, self.curstep]
        self.p_infi = self.p_inf[:, self.curstep]
        self.tau_pi = self.tau_p[:, self.curstep]
        # leak
        self.Ileaki = self.Ileak[:, self.curstep]

        # ODE
        # sodium
        self.alpha_mi = self.activation_func_ReLUlike(-0.32, self.Vth + 13, -1 / 4, self.Vth + 13, self.Vi)
        self.beta_mi = self.activation_func_ReLUlike(0.28, self.Vth + 40, 1 / 5, self.Vth + 40, self.Vi)
        self.alpha_hi = self.activation_func_exp(0.128, -1 / 18, self.Vth + 17, self.Vi)
        self.beta_hi = self.activation_func_sigmoid(4, -1 / 5, self.Vth + 40, self.Vi)
        self.INai = self.gNa * self.mi ** 3 * self.hi * (self.eNa - self.Vi)
        # potassium
        self.alpha_ni = self.activation_func_ReLUlike(-0.032, self.Vth + 15, -1 / 5, self.Vth + 15, self.Vi)
        self.beta_ni = self.activation_func_exp(0.5, -1 / 40, self.Vth + 10, self.Vi)
        self.IKi = self.gK * self.ni ** 4 * (self.eK - self.Vi)
        # slow voltage dependent potassium
        self.Imi = self.gM * self.pi * (self.eK - self.Vi)
        # leak
        self.Ileaki = self.gL * (self.eL - self.Vi)
        self.p_infi = self.activation_func_sigmoid(1, -1 / 10, -35, self.Vi)
        self.tau_pi = (self.tau_max /
                       (3.3 * np.exp((self.Vi + 35) / 20) +
                        np.exp(-(self.Vi + 35) / 20)))

        self.k1V = (self.INai +
                    self.IKi +
                    self.Ileaki +
                    self.Imi +
                    self.Iext[:, self.curstep])

        self.k1m = self.alpha_mi * (1 - self.mi) - self.beta_mi * self.mi
        self.k1h = self.alpha_hi * (1 - self.hi) - self.beta_hi * self.hi
        self.k1n = self.alpha_ni * (1 - self.ni) - self.beta_ni * self.ni
        self.k1p = (self.p_infi - self.pi) / self.tau_pi

        # first order Euler method
        #self.V_extra[:, self.curstep + 1] = self.V_extrai  - self.Iext[:, self.curstep] * self.dt
        #self.V_intra[:, self.curstep + 1] = self.V_intrai + (self.INai + self.IKi + self.Ileaki + self.Imi)* self.dt

        #self.V_extra[0, self.curstep + 1] = self.V_extrai[0] - self.Iext[0, self.curstep] * self.dt - self.g_extra * (self.V_extrai[0]-self.V_extrai[1])
        self.V_intra[0, self.curstep + 1] = self.V_intrai[0] + (self.INai[0] + self.IKi[0] + self.Ileaki[0] + self.Imi[0]) * self.dt  + self.g_intra * (self.V_intrai[1]-self.V_intrai[0])
        #self.V_extra[1, self.curstep + 1] = self.V_extrai[1] + self.Iext[0, self.curstep] * self.dt  - self.g_extra * (self.V_extrai[1]-self.V_extrai[0])
        self.V_intra[1, self.curstep + 1] = self.V_intrai[1] + (self.INai[1] + self.IKi[1] + self.Ileaki[1] + self.Imi[1]) * self.dt  + self.g_intra * (self.V_intrai[0]-self.V_intrai[1])

        self.m[:, self.curstep + 1] = self.mi + self.k1m * self.dt
        self.h[:, self.curstep + 1] = self.hi + self.k1h * self.dt
        self.n[:, self.curstep + 1] = self.ni + self.k1n * self.dt
        self.p[:, self.curstep + 1] = self.pi + self.k1p * self.dt

        # update original array
        self.INa[:, self.curstep] = self.INai
        self.IK[:, self.curstep] = self.IKi
        self.Im[:, self.curstep] = self.Imi
        self.Ileak[:, self.curstep] = self.Ileaki

        self.curstep += 1
