"""
***Unit of parameters***
membrane potential -> mV
conductance -> mS
capacitance -> uF
current -> uA
"""
# coding: UTF-8
import numpy as np


class Neuron_LIF():
    def __init__(self, delay=20, syn_type=1, N=1, dt=0.05, T=1000, Cm=1, G_L=25, Vth=-56.2, erest=-70,
                 Iext_amp=0, Pmax=0, Pmax_AMPA=0, Pmax_NMDA=0, tau_syn=5.26, esyn=0, gsyn=0,
                 noise_type=0, alpha=0.5,
                 beta=0, D=1):

        self.set_neuron_palm(delay, syn_type, N, dt, T, Cm, G_L, Vth, erest,
                             Iext_amp, Pmax, Pmax_AMPA, Pmax_NMDA, tau_syn, esyn, gsyn,
                             noise_type, alpha,
                             beta, D)

    def set_neuron_palm(self, delay=20, syn_type=1, N=1, dt=0.05, T=1000, Cm=1, G_L=25, Vth=-56.2, erest=-70,
                 Iext_amp=0, Pmax=0, Pmax_AMPA=0, Pmax_NMDA=0, tau_syn=5.26, esyn=0, gsyn=0,
                 noise_type=0, alpha=0.5,
                 beta=0, D=1):

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
        # LIF model
        self.Cm = Cm
        self.G_L = G_L
        self.Vth = Vth
        self.erest = erest
        self.V = -70 * np.ones((self.N, self.allsteps))
        self.k1V = 0 * np.ones(self.N)
        # connection relationship
        self.Syn_weight = np.zeros((self.N, self.N))
        self.Syn_weight[0, 0] = 1
        #self.Syn_weight[0, 1] = 1

        # synaptic current
        self.Isyn = np.zeros((self.N, self.allsteps))
        # synaptic conductance
        self.gsyn = np.zeros((self.N, self.N))
        # synaptic reversal potential
        self.esyn = esyn * np.ones((self.N, self.N))
        self.Pmax = Pmax
        self.tau_syn = tau_syn
        # external input
        self.Iext_amp = Iext_amp
        self.Iext = np.zeros((self.N, self.allsteps))
        self.Iext[0, int(150 / self.dt):int(350 / self.dt)] = self.Iext_amp
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

    def alpha_function(self, t, Pmax, tau):
        if t < 0:
            return 0
        elif ((Pmax * t / tau) * np.exp(-t / tau)) < 0.00001:
            return 0
        else:
            #print("pippi")
            return (Pmax * t / tau) * np.exp(-t / tau)

    """
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
    """

    def calc_synaptic_input(self, i):
        # recording fire time (positive edge)
        if self.syn_type == 1:
            pass
        elif self.syn_type == 2:
            pass
        elif self.syn_type == 3:
            pass
        # alpha synapse
        elif self.syn_type == 4:
            for j in range(0, self.N):
                self.gsyn[i, j] = self.alpha_function(self.curstep * self.dt - self.t_fire[j, i], self.Pmax, self.tau_syn)
            for j in range(0, self.N):
                self.Isyni[i] += self.Syn_weight[j, i] * self.gsyn[i, j] * (self.esyn[i, j] - self.Vi[i])
        elif self.syn_type == 5:
            pass
        elif self.syn_type == 6:
            pass
        else:
            pass

    # one step processing
    def propagation(self):
        # slice
        self.Vi = self.V[:, self.curstep]
        self.Isyni = self.Isyn[:, self.curstep]
        self.Inoisei = self.Inoise[:, self.curstep]

        # calculate synaptic input
        if (self.curstep * self.dt) > 50:
            for i in range(0, self.N):
                self.calc_synaptic_input(i)
                # mV
                if self.Vi[i] >= self.Vth:
                    self.t_fire[i, :] = self.curstep * self.dt
                    self.t_fire_list[i, self.curstep] = 50
                    self.Vi[i] = self.erest-10
                """
                # V
                if self.Vi[i] >= self.Vth:
                    self.t_fire[i, :] = self.curstep * self.dt
                    self.t_fire_list[i, self.curstep] = 50
                    self.Vi[i] = self.erest - 0.010
                """

        # Noise
        # 1 : gaussian white
        # 2 : Ornstein-Uhlenbeck process
        if self.noise_type == 1:
            self.Inoise[:, self.curstep + 1] = self.D * self.dWt[:, self.curstep]
        elif self.noise_type == 2:
            self.Inoise[:, self.curstep + 1] = (self.Inoisei +
                                                (-self.alpha * (self.Inoisei - self.beta) * self.dt
                                                 + self.D * self.dWt[:, self.curstep]))
        else:
            pass

        # ODE-first order Euler method
        self.k1V = (self.G_L*(self.erest-self.Vi) + self.Isyni + self.Iext[:, self.curstep] + self.Inoisei)/self.Cm
        self.V[:, self.curstep + 1] = self.Vi + self.k1V * self.dt

        self.curstep += 1