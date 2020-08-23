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

        # type of synaptic coupling
        self.syn_type = syn_type
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
        self.V = -65 * np.ones((self.N, self.allsteps))
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
        # T type calcium
        self.ItCa = 0 * np.ones((self.N, self.allsteps))
        self.eCa = eCa * np.ones(self.N)
        self.gtCa = gtCa * np.ones(self.N)
        self.u = 0.5 * np.ones((self.N, self.allsteps))
        self.s_inf = 0 * np.ones((self.N, self.allsteps))
        self.u_inf = 0 * np.ones((self.N, self.allsteps))
        self.tau_u = 0 * np.ones((self.N, self.allsteps))
        self.k1u = 0 * np.ones(self.N)
        # L type calcium
        self.IlCa = 0 * np.ones((self.N, self.allsteps))
        self.glCa = glCa * np.ones(self.N)
        self.q = 0.5 * np.ones((self.N, self.allsteps))
        self.r = 0.5 * np.ones((self.N, self.allsteps))
        self.alpha_q = 0 * np.ones((self.N, self.allsteps))
        self.alpha_r = 0 * np.ones((self.N, self.allsteps))
        self.beta_q = 0 * np.ones((self.N, self.allsteps))
        self.beta_r = 0 * np.ones((self.N, self.allsteps))
        self.k1q = 0 * np.ones(self.N)
        self.k1r = 0 * np.ones(self.N)
        # Ca activated K
        self.IkCa = 0 * np.ones((self.N, self.allsteps))
        self.gkCa = gkCa * np.ones(self.N)
        self.ca_influx = 0 * np.ones((self.N, self.allsteps))
        self.tau_ca_influx = 2700
        self.ca_influx_step = 100

        # connection relationship
        #self.Syn_weight = np.ones((self.N, self.N))
        #self.Syn_weight = np.identity(self.N)
        self.Syn_weight = np.zeros((self.N, self.N))
        self.Syn_weight[0, 0] = 1

        """
        self.Syn_weight[0, 1] = 1
        self.Syn_weight[1, 2] = 1
        self.Syn_weight[2, 0] = 1
        """
        # synaptic current
        self.Isyn = np.zeros((self.N, self.allsteps))
        self.INMDA = np.zeros((self.N, self.allsteps))
        self.IAMPA = np.zeros((self.N, self.allsteps))

        # synaptic conductance
        self.gsyn = np.zeros((self.N, self.N))
        self.gNMDA = np.zeros((self.N, self.N))
        self.gAMPA = np.zeros((self.N, self.N))

        # synaptic reversal potential
        self.esyn = esyn * np.ones((self.N, self.N))
        self.tau_syn = tau_syn
        self.Mg_conc = Mg_conc

        # maximal synaptic conductance
        self.Pmax_AMPA = Pmax_AMPA
        self.Pmax_NMDA = Pmax_NMDA

        # external input
        self.Iext_amp = Iext_amp
        self.Iext = np.zeros((self.N, self.allsteps))
        self.Iext[0, int(1000 / self.dt):int(1005 / self.dt)] = self.Iext_amp
        """
        self.Iext[0, int(220/self.dt):int(225/self.dt)] = self.Iext_amp
        self.Iext[0, int(240/self.dt):int(245/self.dt)] = self.Iext_amp
        self.Iext[0, int(260/self.dt):int(265/self.dt)] = self.Iext_amp
        self.Iext[0, int(280/self.dt):int(285/self.dt)] = self.Iext_amp
        self.Iext[0, int(300/self.dt):int(305/self.dt)] = self.Iext_amp
        self.Iext[0, int(320/self.dt):int(325/self.dt)] = self.Iext_amp
        """

        # firing time
        self.t_fire = -10000 * np.ones((self.N, self.N))
        self.t_fire_list = np.zeros((self.N, self.allsteps))

        # current step
        self.curstep = 0

        # noise
        self.noise_type = noise_type
        self.Inoise = np.zeros((self.N, self.allsteps))

        self.dn = np.zeros((self.N, self.allsteps))
        self.ramda = ramda
        self.alpha = alpha
        self.beta = beta
        self.D = D
        self.dWt = np.random.normal(0, self.dt ** (1 / 2), (self.N, self.allsteps))

        # dynamic synapse
        self.delay = delay
        self.R_AMPA = np.ones((self.N, self.N, self.allsteps))
        self.R_NMDA = np.ones((self.N, self.N, self.allsteps))
        self.E_AMPA = np.zeros((self.N, self.N, self.allsteps))
        self.E_NMDA = np.zeros((self.N, self.N, self.allsteps))
        self.I_AMPA = np.zeros((self.N, self.N, self.allsteps))
        self.I_NMDA = np.zeros((self.N, self.N, self.allsteps))
        self.dR = 0
        self.dE = 0
        self.U_SE_AMPA = U_SE_AMPA
        self.U_SE_NMDA = U_SE_NMDA
        self.tau_rise_AMPA = tau_rise_AMPA
        self.tau_rise_NMDA = tau_rise_NMDA
        self.tau_rec_AMPA = tau_rec_AMPA
        self.tau_rec_NMDA = tau_rec_AMPA
        self.tau_inact_AMPA = tau_inact_AMPA
        self.tau_inact_NMDA = tau_inact_NMDA

    def alpha_function(self, t):
        if t < 0:
            return 0
        elif ((self.Pmax * t / self.tau_syn * 0.1) *
              np.exp(-t / self.tau_syn * 0.1)) < 0.00001:
            return 0
        else:
            return (self.Pmax * t / self.tau_syn) * np.exp(-t / self.tau_syn)

    def biexp_func(self, t, Pmax, t_rise, t_fall):
        if t < 0:
            return 0
        elif Pmax * (np.exp(-t / t_fall) - np.exp(-t / t_rise)) < 0.00001:
            return 0
        else:
            return Pmax * (np.exp(-t / t_fall) - np.exp(-t / t_rise))

    def exp_decay(self, x, tau_rise):
        if (x / tau_rise) > 100:
            return 0
        else:
            return np.exp(- x / tau_rise)

    def calc_synaptic_input(self, i):
        # recording fire time (positive edge)
        if self.V[i, self.curstep - 1] <= 0 and self.V[i, self.curstep] > 0 and self.curstep * self.dt > 200:
            self.t_fire[i, :] = self.curstep * self.dt
            self.t_fire_list[i, self.curstep] = 50
        # sum of the synaptic current for each neuron
        if self.syn_type == 1:
            pass
        elif self.syn_type == 2:
            pass
        elif self.syn_type == 3:
            pass

        # alpha synapse
        elif self.syn_type == 4:
            for j in range(0, self.N):
                self.gsyn[i, j] = \
                    (self.alpha_function(self.curstep * self.dt -
                                         self.t_fire[j, i]) +
                     self.alpha_function(self.curstep * self.dt -
                                         self.t_fire[j, i, 1]))
        # NMDA & AMPA
        elif self.syn_type == 5:
            for j in range(0, self.N):

                if self.curstep * self.dt > 200:
                    # cite from "neuronal noise"
                    self.gNMDA[i, j] = self.biexp_func(self.curstep * self.dt - self.t_fire[j, i], self.Pmax_NMDA, 20,
                                                       125) / (1 + (self.Mg_conc / 3.57) * np.exp(-0.062 * self.Vi))
                    self.gAMPA[i, j] = self.biexp_func(self.curstep * self.dt - self.t_fire[j, i], self.Pmax_AMPA, 0.8,
                                                       5)
                    self.gsyn[i, j] = self.gNMDA[i, j] + self.gAMPA[i, j]
                else:
                    self.gsyn[i, j] = 0

        # NMDA & AMPA with STP
        elif self.syn_type == 6:
            for j in range(0, self.N):
                self.dR_AMPA = (self.dt * ((self.I_AMPA[i, j, self.curstep] / self.tau_rec_AMPA)
                                           - self.R_AMPA[i, j, self.curstep] * self.U_SE_AMPA * self.exp_decay(
                        self.curstep * self.dt - self.t_fire[j, i] - self.delay, self.tau_rise_AMPA)))
                self.dR_NMDA = (self.dt * ((self.I_AMPA[i, j, self.curstep] / self.tau_rec_NMDA)
                                           - self.R_NMDA[i, j, self.curstep] * self.U_SE_NMDA * self.exp_decay(
                        self.curstep * self.dt - self.t_fire[j, i] - self.delay, self.tau_rise_NMDA)))
                self.dE_AMPA = (self.dt * ((- self.E_AMPA[i, j, self.curstep] / self.tau_inact_AMPA)
                                           + self.U_SE_AMPA * self.R_AMPA[i, j, self.curstep] * self.exp_decay(
                        self.curstep * self.dt - self.t_fire[j, i] - self.delay, self.tau_rise_AMPA)))
                self.dE_NMDA = (self.dt * ((- self.E_NMDA[i, j, self.curstep] / self.tau_inact_NMDA)
                                           + self.U_SE_NMDA * self.R_NMDA[i, j, self.curstep] * self.exp_decay(
                        self.curstep * self.dt - self.t_fire[j, i] - self.delay, self.tau_rise_NMDA)))

                self.R_AMPA[i, j, self.curstep + 1] = self.R_AMPA[i, j, self.curstep] + self.dR_AMPA
                self.R_NMDA[i, j, self.curstep + 1] = self.R_NMDA[i, j, self.curstep] + self.dR_NMDA
                self.E_AMPA[i, j, self.curstep + 1] = self.E_AMPA[i, j, self.curstep] + self.dE_AMPA
                self.E_NMDA[i, j, self.curstep + 1] = self.E_NMDA[i, j, self.curstep] + self.dE_NMDA
                self.I_AMPA[i, j, self.curstep + 1] = 1 - self.R_AMPA[i, j, self.curstep + 1] - self.E_AMPA[
                    i, j, self.curstep + 1]
                self.I_NMDA[i, j, self.curstep + 1] = 1 - self.R_NMDA[i, j, self.curstep + 1] - self.E_NMDA[
                    i, j, self.curstep + 1]

                self.gNMDA[i, j] = self.Pmax_NMDA * (1 / 0.43) * self.E_NMDA[i, j, self.curstep] / (
                        1 + (self.Mg_conc / 3.57) * np.exp(-0.062 * self.Vi[i]))
                self.gAMPA[i, j] = self.Pmax_AMPA * (1 / 0.37) * self.E_AMPA[i, j, self.curstep]
        else:
            pass

        # sum
        for j in range(0, self.N):
            self.INMDAi[i] += self.Syn_weight[j, i] * self.gNMDA[i, j] * (self.esyn[i, j] - self.Vi[i])
            self.IAMPAi[i] += self.Syn_weight[j, i] * self.gAMPA[i, j] * (self.esyn[i, j] - self.Vi[i])
            self.Isyni[i] = self.INMDAi[i] + self.IAMPAi[i]

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
        # slice
        self.Vi = self.V[:, self.curstep]
        # sodium
        self.INai = self.INa[:, self.curstep]
        self.mi = self.m[:, self.curstep]
        self.hi = self.h[:, self.curstep]
        self.alpha_mi = self.alpha_m[:, self.curstep]
        self.beta_mi = self.beta_m[:, self.curstep]
        self.alpha_hi = self.alpha_h[:, self.curstep]
        self.beta_hi = self.beta_h[:, self.curstep]
        # persistent sodium
        """
        self.IpNai = self.IpNa[:, self.curstep]
        self.pnai = self.pna[:, self.curstep]
        self.alpha_pnai = self.alpha_pna[:, self.curstep]
        self.beta_pnai = self.beta_pna[:, self.curstep]
        """
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
        # T type calcium
        """
        self.ItCai = self.ItCa[:, self.curstep]
        self.ui = self.u[:, self.curstep]
        self.s_infi = self.s_inf[:, self.curstep]
        self.u_infi = self.u_inf[:, self.curstep]
        self.tau_ui = self.tau_u[:, self.curstep]
        """
        # L type calcium
        """
        self.IlCai = self.IlCa[:, self.curstep]
        self.qi = self.u[:, self.curstep]
        self.ri = self.u[:, self.curstep]
        self.alpha_qi = self.alpha_n[:, self.curstep]
        self.beta_qi = self.beta_n[:, self.curstep]
        self.alpha_ri = self.alpha_n[:, self.curstep]
        self.beta_ri = self.beta_n[:, self.curstep]
        """
        # Ka activated calcium
        self.IkCai = self.IkCa[:, self.curstep]
        self.ca_influxi = self.ca_influx[:, self.curstep]
        # synapse
        self.Isyni = self.Isyn[:, self.curstep]
        self.INMDAi = self.INMDA[:, self.curstep]
        self.IAMPAi = self.IAMPA[:, self.curstep]
        # synaptic noise
        self.Inoisei = self.Inoise[:, self.curstep]

        # calculate synaptic input
        for i in range(0, self.N):
            self.calc_synaptic_input(i)

        # Noise
        # 1 : gaussian white
        # 2 : Ornstein-Uhlenbeck process
        # 3 : sin wave
        if self.noise_type == 1:
            self.Inoise[:, self.curstep + 1] = self.D * self.dWt[:, self.curstep]
        elif self.noise_type == 2:
            self.Inoise[:, self.curstep + 1] = (self.Inoisei +
                                                (-self.alpha * (self.Inoisei - self.beta) * self.dt
                                                 + self.D * self.dWt[:, self.curstep]))
        elif self.noise_type == 3:
            self.Inoise[:, self.curstep + 1] = (self.alpha *
                                                np.sin(np.pi *
                                                       self.curstep / (1000 / self.dt)))
        else:
            pass

        # ODE
        # sodium
        self.alpha_mi = self.activation_func_ReLUlike(-0.32, self.Vth + 13, -1 / 4, self.Vth + 13, self.Vi)
        self.beta_mi = self.activation_func_ReLUlike(0.28, self.Vth + 40, 1 / 5, self.Vth + 40, self.Vi)
        self.alpha_hi = self.activation_func_exp(0.128, -1 / 18, self.Vth + 17, self.Vi)
        self.beta_hi = self.activation_func_sigmoid(4, -1 / 5, self.Vth + 40, self.Vi)
        self.INai = self.gNa * self.mi ** 3 * self.hi * (self.eNa - self.Vi)
        # persistent sodium
        """
        self.alpha_pnai = self.activation_func_ReLUlike(-0.0353, -27, -1/10.2, -27, self.Vi)
        self.beta_pnai = self.activation_func_ReLUlike(0.000883, -34, 1/10, -34, self.Vi)
        self.IpNai = self.gpNa * self.pnai**3 * (self.eNa - self.Vi)
        """
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

        # T type calcium
        """

        self.s_infi = self.activation_func_sigmoid(1, -1/6.2, -2-57, self.Vi)
        self.u_infi = self.activation_func_sigmoid(1, 1/4, -2-81, self.Vi)
        self.tau_ui = (30.8 + (211.4 + np.exp(np.clip((self.Vi+2+113.2)/5, -709, 10000))) /
               (3.7 * (1 + np.exp(np.clip((self.Vi+2+84)/3.2, -709, 10000)))))
        self.ItCai = self.gtCa * self.s_infi**2 * self.ui * (self.eCa - self.Vi)
        """
        # L type calcium
        """
        self.alpha_qi = self.activation_func_ReLUlike(-0.055, -27, -1/3.8, -27, self.Vi)
        self.beta_qi = self.activation_func_exp(0.94, -1/17, -75, self.Vi)
        self.alpha_ri = self.activation_func_exp(0.000457, -1/50, -13, self.Vi)
        self.beta_ri = self.activation_func_sigmoid(0.0065, 1/28, -15, self.Vi)
        self.IlCai = self.glCa * self.qi**2 * self.ri * (self.eCa - self.Vi)
        """
        # K activated calcium
        self.IkCai = self.gkCa * self.ca_influxi * (self.eK - self.Vi)

        self.k1V = (self.INai +
                    self.IKi +
                    self.Ileaki +
                    self.Imi +
                    self.Isyni +
                    self.IkCai +
                    self.Iext[:, self.curstep] +
                    self.Inoisei)

        if (self.curstep * self.dt) < 200:
            self.k1V -= self.Isyni
        self.k1m = self.alpha_mi * (1 - self.mi) - self.beta_mi * self.mi
        self.k1h = self.alpha_hi * (1 - self.hi) - self.beta_hi * self.hi
        # self.k1pna = self.alpha_pnai * (1 - self.pnai) - self.beta_pnai * self.pnai
        self.k1n = self.alpha_ni * (1 - self.ni) - self.beta_ni * self.ni
        self.k1p = (self.p_infi - self.pi) / self.tau_pi

        """
        self.k1u = (self.u_infi - self.ui) / self.tau_ui
        self.k1q = self.alpha_qi * (1-self.qi) - self.beta_qi * self.qi
        self.k1r = self.alpha_ri * (1-self.ri) - self.beta_ri * self.ri
        """

        # first order Euler method
        self.V[:, self.curstep + 1] = self.Vi + self.k1V * self.dt
        self.m[:, self.curstep + 1] = self.mi + self.k1m * self.dt
        self.h[:, self.curstep + 1] = self.hi + self.k1h * self.dt
        # self.pna[:, self.curstep+1] = self.pnai + self.k1pna * self.dt
        self.n[:, self.curstep + 1] = self.ni + self.k1n * self.dt
        self.p[:, self.curstep + 1] = self.pi + self.k1p * self.dt
        if self.V[i, self.curstep - 1] > 0 and self.V[i, self.curstep] <= 0 and self.curstep * self.dt > 200:
            self.ca_influx[:, self.curstep + 1] = self.ca_influxi - (
                    self.ca_influxi / self.tau_ca_influx) + self.dt + self.ca_influx_step
        else:
            self.ca_influx[:, self.curstep + 1] = self.ca_influxi - (self.ca_influxi / self.tau_ca_influx) + self.dt
        """
        self.u[:, self.curstep+1] = self.ui + self.k1u * self.dt
        self.q[:, self.curstep+1] = self.qi + self.k1q * self.dt
        self.r[:, self.curstep+1] = self.ri + self.k1r * self.dt
        self.V[:, self.curstep+1] = self.Vi + self.k1V * self.dt
        """

        # update original array
        self.INa[:, self.curstep] = self.INai
        # self.IpNa[:, self.curstep] = self.IpNai
        self.IK[:, self.curstep] = self.IKi
        self.Im[:, self.curstep] = self.Imi
        self.Ileak[:, self.curstep] = self.Ileaki
        self.IkCa[:, self.curstep] = self.IkCai
        self.ca_influx[:, self.curstep] = self.ca_influxi

        """
        self.ItCa[:, self.curstep] = self.ItCai
        self.s_inf[:, self.curstep] = self.s_infi
        self.u_inf[:, self.curstep] = self.u_infi
        self.tau_u[:, self.curstep] = self.tau_ui
        self.IlCa[:, self.curstep] = self.IlCai
        """
        self.Inoise[:, self.curstep] = self.Inoisei
        self.curstep += 1
