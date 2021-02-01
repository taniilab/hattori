"""
***Unit of parameters***
2 compartment minimal hodgkin-huxley neuron for extracellular electrical stimulation
membrane potential -> mV
time -> ms
conductance -> mS
capacitance -> uF
current -> uA
/cm^2
"""
# coding: UTF-8
import numpy as np


class Neuron_HH():
    def __init__(self, N=2, dt=0.05, T=5000, Cm=1, Vth=-56.2,
                        eNa=50, gNa=56, eK=-90, gK=6, eL=-70.3, gL=0.0205, g_extra=1, g_intra=3, tau_vextra=1, stim_amp=10):
        self.set_neuron_palm(N, dt, T, Cm, Vth, eNa, gNa, eK, gK, eL, gL, g_extra,  g_intra, tau_vextra, stim_amp)

    def set_neuron_palm(self, N=2, dt=0.05, T=5000, Cm=1, Vth=-56.2,
                        eNa=55, gNa=120, eK=-72, gK=36, eL=-49.387, gL=0.3, g_extra=1, g_intra=3, tau_vextra=1, stim_amp=10):

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
        #self.surface = 1e-5 # cm^2
        self.surface = 1
        self.Cm = Cm * self.surface
        self.Vth = Vth
        self.V_intra = -65 * np.ones((self.N, self.allsteps))
        self.V_extra = np.zeros((self.N, self.allsteps))
        self.tau_vextra = tau_vextra
        self.stim_amp = stim_amp
        # extracellular stimulation pattern
        for i in range(self.allsteps):
            self.nowT = i*self.dt
            print(self.nowT)
            self.peak = 60
            self.end = 500.4
            if 500 < self.nowT and self.nowT <= 500.2:
                #self.V_extra[0, i] = self.peak
                #self.V_extra[1, i] = -self.peak
                self.V_extra[0, i] = self.stim_amp * (1 - np.exp(-20*(self.nowT-500)))
                self.V_extra[1, i] = -self.stim_amp * (1 - np.exp(-20*(self.nowT-500)))
            elif 500.2 < self.nowT and self.nowT <= self.end:
                self.V_extra[0, i] = self.stim_amp * np.exp(-20*(self.nowT-500.2))
                self.V_extra[1, i] = -self.stim_amp * np.exp(-20*(self.nowT-500.2))
                pass
            else:
                pass
        #self.V_extra[0, int(500 / self.dt):int(500.1 / self.dt)] = self.stim_amp
        #self.V_extra[1, int(500 / self.dt):int(500.1 / self.dt)] = -self.stim_amp
        """
        for i in range(int(50/self.dt)):
            self.V_extra[0, int(500 / self.dt)+i] = 50 * np.exp(- i*0.1*self.tau_vextra*self.dt) # exp decay
            self.V_extra[1, int(500 / self.dt)+i] = -50 * np.exp(- i*0.1*self.tau_vextra*self.dt)
        """
        self.k1V = 0 * np.ones(self.N)

        # sodium
        self.INa = 0 * np.ones((self.N, self.allsteps))
        self.eNa = eNa * np.ones(self.N)
        self.gNa = gNa * np.ones(self.N) * self.surface
        self.m = 0.5 * np.ones((self.N, self.allsteps))
        self.h = 0.06 * np.ones((self.N, self.allsteps))
        self.alpha_m = 0 * np.ones((self.N, self.allsteps))
        self.alpha_h = 0 * np.ones((self.N, self.allsteps))
        self.beta_m = 0 * np.ones((self.N, self.allsteps))
        self.beta_h = 0 * np.ones((self.N, self.allsteps))
        self.k1m = 0 * np.ones(self.N)
        self.k1h = 0 * np.ones(self.N)
        # potassium
        self.IK = 0 * np.ones((self.N, self.allsteps))
        self.eK = eK * np.ones(self.N)
        self.gK = gK * np.ones(self.N) * self.surface
        self.n = 0.5 * np.ones((self.N, self.allsteps))
        self.alpha_n = 0 * np.ones((self.N, self.allsteps))
        self.beta_n = 0 * np.ones((self.N, self.allsteps))
        self.k1n = 0 * np.ones(self.N)
        # leak
        self.Ileak = 0 * np.ones((self.N, self.allsteps))
        self.eL = eL * np.ones(self.N)
        self.gL = gL * np.ones(self.N) * self.surface
        self.Isyn = np.zeros((self.N, self.allsteps))

        # firing time
        self.t_fire = -10000 * np.ones((self.N, self.N))
        self.t_fire_list = np.zeros((self.N, self.allsteps))

        # 2 compartment model
        self.Ilink = 0 * np.ones((self.N, self.allsteps))
        self.g_extra = g_extra * self.surface
        self.g_intra = g_intra  * self.surface
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
        # leak
        self.Ileaki = self.Ileak[:, self.curstep]
        # 2 compartment
        self.Ilinki = self.Ilink[:, self.curstep]

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
        # leak
        self.Ileaki = self.gL * (self.eL - self.Vi)

        self.k1m = self.alpha_mi * (1 - self.mi) - self.beta_mi * self.mi
        self.k1h = self.alpha_hi * (1 - self.hi) - self.beta_hi * self.hi
        self.k1n = self.alpha_ni * (1 - self.ni) - self.beta_ni * self.ni

        # first order Euler method
        self.Ilinki[0] = self.g_intra * (self.V_intrai[1] - self.V_intrai[0])
        self.Ilinki[1] = self.g_intra * (self.V_intrai[0] - self.V_intrai[1])
        self.V_intra[0, self.curstep + 1] = self.V_intrai[0] + (1/self.Cm)*(self.INai[0] + self.IKi[0] + self.Ileaki[0] + self.Ilinki[0]) * self.dt
        self.V_intra[1, self.curstep + 1] = self.V_intrai[1] + (1/self.Cm)*(self.INai[1] + self.IKi[1] + self.Ileaki[1] + self.Ilinki[1]) * self.dt
        if  self.curstep < int(500 / self.dt) or int(self.end / self.dt) < self.curstep:
            self.V_extra[0, self.curstep + 1] = self.V_extrai[0] + self.dt * self.g_extra * (self.V_extrai[1] - self.V_extrai[0])
            self.V_extra[1, self.curstep + 1] = self.V_extrai[1] + self.dt * self.g_extra * (self.V_extrai[0] - self.V_extrai[1])

        self.m[:, self.curstep + 1] = self.mi + self.k1m * self.dt
        self.h[:, self.curstep + 1] = self.hi + self.k1h * self.dt
        self.n[:, self.curstep + 1] = self.ni + self.k1n * self.dt

        # update original array
        self.INa[:, self.curstep] = self.INai
        self.IK[:, self.curstep] = self.IKi
        self.Ileak[:, self.curstep] = self.Ileaki

        self.curstep += 1
