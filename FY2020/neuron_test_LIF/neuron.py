"""
Created on Wed May 24 11:37:33 2017

@author: Hattori
"""
# coding: UTF-8
import numpy as np


class Neuron_LIF():
    def __init__(self, delay=20, syn_type=1, N=1, dt=0.05, T=1000, Cm=1, Vth=-56.2, erest=-70,
                 Iext_amp=0, Pmax=0, Pmax_AMPA=0, Pmax_NMDA=0, tau_syn=5.26, esyn=0, gsyn=0,
                 noise_type=0, alpha=0.5,
                 beta=0, D=1, ratio=0.5, Mg_conc=1,
                 U_SE_AMPA=0.3, U_SE_NMDA=0.03, tau_rise_AMPA=0.9, tau_rise_NMDA=70, tau_rec_AMPA=200, tau_rec_NMDA=200,
                 tau_inact_AMPA=5, tau_inact_NMDA=30):

        self.set_neuron_palm(delay, syn_type, N, dt, T, Cm, Vth, erest,
                             Iext_amp, Pmax, Pmax_AMPA, Pmax_NMDA, tau_syn, esyn, gsyn,
                             noise_type, alpha,
                             beta, D, ratio, Mg_conc,
                             U_SE_AMPA, U_SE_NMDA, tau_rise_AMPA, tau_rise_NMDA, tau_rec_AMPA, tau_rec_NMDA,
                             tau_inact_AMPA, tau_inact_NMDA)

    def set_neuron_palm(self, delay=20, syn_type=1, N=1, dt=0.05, T=1000, Cm=1, Vth=-56.2, erest=-70,
                 Iext_amp=0, Pmax=0, Pmax_AMPA=0, Pmax_NMDA=0, tau_syn=5.26, esyn=0, gsyn=0,
                 noise_type=0, alpha=0.5,
                 beta=0, D=1, ratio=0.5, Mg_conc=1,
                 U_SE_AMPA=0.3, U_SE_NMDA=0.03, tau_rise_AMPA=0.9, tau_rise_NMDA=70, tau_rec_AMPA=200, tau_rec_NMDA=200,
                 tau_inact_AMPA=5, tau_inact_NMDA=30):

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
        self.erest = erest
        self.V = -65 * np.ones((self.N, self.allsteps))
        self.k1V = 0 * np.ones(self.N)

        # connection relationship
        self.Syn_weight = np.zeros((self.N, self.N))

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
        self.Pmax = Pmax
        self.Pmax_AMPA = Pmax_AMPA
        self.Pmax_NMDA = Pmax_NMDA
        self.tau_syn = tau_syn

        # external input
        self.Iext_amp = Iext_amp
        self.Iext = np.zeros((self.N, self.allsteps))
        self.Iext[0, int(1000 / self.dt):int(3000 / self.dt)] = self.Iext_amp
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
        if self.Vi[i] >= self.Vth and self.curstep * self.dt > 200:
            self.t_fire[i, :] = self.t_fire[i, :]
            self.t_fire[i, :] = self.curstep * self.dt
            self.t_fire_list[i, self.curstep] = 50
            self.Vi[i] = self.erest

        if self.syn_type == 1:
            pass
        elif self.syn_type == 2:
            pass
        elif self.syn_type == 3:
            pass

        # alpha function
        elif self.syn_type == 4:
            for j in range(0, self.N):
                self.gsyn[i, j] = \
                    (self.alpha_function(self.curstep * self.dt -
                                         self.t_fire[j, i]) +
                     self.alpha_function(self.curstep * self.dt -
                                         self.t_fire[j, i]))
            for j in range(0, self.N):
                self.Isyni[i] += self.Syn_weight[j, i] * self.gNMDA[i, j] * (self.esyn[i, j] - self.Vi[i])
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

            for j in range(0, self.N):
                self.INMDAi[i] += self.Syn_weight[j, i] * self.gNMDA[i, j] * (self.esyn[i, j] - self.Vi[i])
                self.IAMPAi[i] += self.Syn_weight[j, i] * self.gAMPA[i, j] * (self.esyn[i, j] - self.Vi[i])
                self.Isyni[i] = self.INMDAi[i] + self.IAMPAi[i]
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

            for j in range(0, self.N):
                self.INMDAi[i] += self.Syn_weight[j, i] * self.gNMDA[i, j] * (self.esyn[i, j] - self.Vi[i])
                self.IAMPAi[i] += self.Syn_weight[j, i] * self.gAMPA[i, j] * (self.esyn[i, j] - self.Vi[i])
                self.Isyni[i] = self.INMDAi[i] + self.IAMPAi[i]
        else:
            pass


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
        pass

        # slice
        self.Vi = self.V[:, self.curstep]
        self.Isyni = self.Isyn[:, self.curstep]
        self.INMDAi = self.INMDA[:, self.curstep]
        self.IAMPAi = self.IAMPA[:, self.curstep]
        self.Inoisei = self.Inoise[:, self.curstep]

        # calculate synaptic input
        if (self.curstep * self.dt) > 200:
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
        self.k1V = ((self.erest-self.Vi) + self.Isyni + self.Iext[:, self.curstep] + self.Inoisei)

        # first order Euler method
        self.V[:, self.curstep + 1] = self.Vi + self.k1V * self.dt

        # update original array
        self.Inoise[:, self.curstep] = self.Inoisei
        self.curstep += 1
