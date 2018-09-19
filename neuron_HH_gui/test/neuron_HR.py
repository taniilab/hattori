@@ -1,293 +0,0 @@
"""
Created on Wed May 24 11:37:33 2017

@author: Hattori
"""
# coding: UTF-8
import numpy as np



class Neuron_HR():
    # constructor
    # 0.02
    def __init__(self, Syncp=1, numneu=1, dt=0.05, simtime=2000, a=1, b=3.15,
                 c=1, d=5, r=0.004, s=4, xr=-1.6, esyn=0, Pmax=3, tausyn=10,
                 xth=0.25, theta=-0.25, Iofs=0, Iext_amp=0, Iext_width=0, Iext_duty=0,
                 Iext_num=0, noise=0, ramda=-10, alpha=0.5,
                 beta=0, D=1,
                 tau_r=50, tau_i=50, use=1, ase=1, gcmp=2, delay=0):
        self.set_neuron_palm(Syncp, numneu, dt, simtime, a, b, c, d, r, s, xr,
                             esyn, Pmax, tausyn, xth, theta, Iext_amp,
                             Iofs, Iext_width, Iext_duty, Iext_num, noise,
                             ramda, alpha, beta, D, tau_r, tau_i, use, ase, gcmp, delay)

    def set_neuron_palm(self, Syncp, numneu, dt, simtime, a, b, c, d, r, s, xr,
                        esyn, Pmax, tausyn, xth, theta, Iext_amp, Iofs, 
                        Iext_width, Iext_duty, Iext_num,  noise, ramda,
                        alpha, beta, D, tau_r, tau_i, use, ase, gcmp, delay):
        # parameters (used by main.py)
        self.parm_dict = {}

        # type of synaptic coupling
        self.Syncp = Syncp
        # number of neuron
        self.numneu = numneu
        # time step
        self.dt = dt
        # simulation time
        self.simtime = simtime
        # all time
        self.tmhist = np.arange(0, self.simtime, self.dt)
        # number of time step
        self.allsteps = len(self.tmhist)
        # HR model
        self.a = a * np.ones((self.numneu, len(self.tmhist)))
        self.b = b * np.ones((self.numneu, len(self.tmhist)))
        self.c = c * np.ones((self.numneu, len(self.tmhist)))
        self.d = d * np.ones((self.numneu, len(self.tmhist)))
        self.r = r * np.ones((self.numneu, len(self.tmhist)))
        self.s = s * np.ones((self.numneu, len(self.tmhist)))
        self.xr = xr * np.ones((self.numneu, len(self.tmhist)))
        self.x = 0 * np.ones((self.numneu, len(self.tmhist)))
        self.y = 0 * np.ones((self.numneu, len(self.tmhist)))
        self.z = 0 * np.ones((self.numneu, len(self.tmhist)))
        self.k1x = 0 * np.ones(self.numneu)
        self.k1y = 0 * np.ones(self.numneu)
        self.k1z = 0 * np.ones(self.numneu)

        # compartment conductance
        self.gcmp = gcmp
        self.delay = delay

        # connection relationship
        self.cnct = np.zeros((self.numneu, self.numneu))

        # synaptic current
        self.Isyn = np.zeros((self.numneu, len(self.tmhist)))
        self.Isyn_hist = np.zeros((self.numneu, self.numneu, 5))

        # synaptic conductance
        self.gsyn = np.zeros((self.numneu, self.numneu))
        # synaptic reversal potential
        self.esyn = esyn * np.ones((self.numneu, self.numneu))
        self.tausyn = tausyn
        # external current
        self.Iext = np.zeros((self.numneu, len(self.tmhist)))
        self.Iext[0] = Iext_amp
        # offset voltage
        self.Iofs = Iofs
        """
        # square wave
        self.Iext_co = 0
        self.Iext_amp = Iext_amp
        self.Iext_width = Iext_width
        self.Iext_duty = Iext_duty
        while self.Iext_co < Iext_num:
            if self.Iext_duty == 0:
                self.Iext[0, int(500/self.dt):int(1800/self.dt)] = Iext_amp
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
        self.t_ap = -100 * np.ones((self.numneu, self.numneu, 2))

        # current step
        self.curstep = 0
        # thresholds
        self.xth = xth
        self.theta = theta
        # noise palameter
        self.noise = noise
        self.n = np.zeros((self.numneu, len(self.tmhist)))

        self.dn = np.zeros((self.numneu, len(self.tmhist)))
        self.ramda = ramda
        self.alpha = alpha
        self.beta = beta
        self.D = D
        self.g = np.random.randn(self.numneu, len(self.tmhist))

        # chemical synapse and alpha function
        self.Pmax = Pmax

        self.fire_tmp = np.zeros(self.numneu)

    def delta_func(self, t):
        y = t == 0
        return y.astype(np.int)

    def discrete_delta_func(self, steps):
        if steps <= (self.gcmp/self.dt):
            return self.Iext
        else:
            return 0

    def alpha_function(self, t):
        if t < 0:
            return 0
        elif ((self.Pmax * t/self.tausyn*0.1) *
              np.exp(-t/self.tausyn*0.1)) < 0.001:
            return 0
        else:
            return (self.Pmax * t/self.tausyn) * np.exp(-t/self.tausyn)

    def step_func(self, t):
        y = t > 0
        return y.astype(np.int)

    def calc_synaptic_input(self, i):
        # recording fire time
        if self.xi[i] > self.xth and (self.curstep *
                                      self.dt - self.fire_tmp[i]) > 10:
            self.t_ap[i, :, 1] = self.t_ap[i, :, 0]
            self.t_ap[i, :, 0] = self.curstep * self.dt
            self.fire_tmp[i] = self.curstep * self.dt

        # sum of the synaptic current for each neuron
        if self.Syncp == 1:
            pass

        # chemical synapse(frequently used in nonlinear biology)
        elif self.Syncp == 2:
            for j in range(0, self.numneu):
                self.gsyn[i, j] =\
                    (self.Pmax *
                     (1 /
                      (1 +
                       np.exp(self.ramda *
                              (self.x[j, self.curstep-self.tausyn] -
                               self.theta)))))
        # depressing synapse
        elif self.Syncp == 3:
            for j in range(0, self.numneu):
                self.gsyn[i, j] = self.ase[i] * self.syn_e[i, j, self.curstep]

        # alpha function
        elif self.Syncp == 4:
            for j in range(0, self.numneu):
                self.gsyn[i, j] =\
                    (self.alpha_function(self.curstep*self.dt -
                                         self.t_ap[j, i, 0]) +
                     self.alpha_function(self.curstep*self.dt -
                                         self.t_ap[j, i, 1]))

        # compartment
        elif self.Syncp == 5:
            for j in range(0, self.numneu):
                self.gsyn[i, j] = self.gcmp

        elif self.Syncp == 6:
            pass

        # summation
        for j in range(0, self.numneu):
            """
            if i == 1:
                self.Isyni[i] += (self.gsyn[i, j] *
                                  (self.esyn[i, j] - self.xi[i]))
            """
            if self.Syncp == 5:

                self.Isyni[i] +=\
                      (self.cnct[i, j] * self.gsyn[i, j] *
                       (self.x[j, int(self.curstep-(self.delay/self.dt))] -
                        self.x[i, self.curstep]))
                """

                self.Isyni[i] +=\
                    (self.cnct[i, j] * self.discrete_delta_func((self.curstep - (self.t_ap[j, j, 0]/self.dt))))
                """
            else:
                self.Isyni[i] +=\
                          (self.cnct[i, j] * self.gsyn[i, j] *
                           (self.esyn[i, j] - self.xi[i]))

    # one step processing
    def propagation(self):
        # slice the current time step
        self.ai = self.a[:, self.curstep]
        self.bi = self.b[:, self.curstep]
        self.ci = self.c[:, self.curstep]
        self.di = self.d[:, self.curstep]
        self.ri = self.r[:, self.curstep]
        self.si = self.s[:, self.curstep]
        self.xri = self.xr[:, self.curstep]
        self.xi = self.x[:, self.curstep]
        self.yi = self.y[:, self.curstep]
        self.zi = self.z[:, self.curstep]

        self.Isyni = self.Isyn[:, self.curstep]
        self.ni = self.n[:, self.curstep]
        self.dni = self.dn[:, self.curstep]

        # calculate synaptic input
        for i in range(0, self.numneu):
            self.calc_synaptic_input(i)

        # Noise
        # 1 : gaussian white
        # 2 : Ornstein-Uhlenbeck process
        # 3 : sin wave
        if self.noise == 1:
            self.n[0, self.curstep+1] = self.D * self.g[0, self.curstep]

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
        self.k1x = (self.yi - self.ai * self.xi**3 + self.bi * self.xi**2 -
                    self.zi + self.Isyni +
                    self.Iext[:, self.curstep] + self.ni)
        self.k1y = (self.ci - self.di * self.xi**2 - self.yi)
        self.k1z = (self.ri * (self.si * (self.xi - self.xri) -
                    self.zi))

        # the first order Euler method
        """
        self.syn_r[:, self.curstep+1] = self.syn_ri + self.k1syn_r * self.dt
        self.syn_e[:, self.curstep+1] = self.syn_ei + self.k1syn_e * self.dt
        self.syn_i[:, self.curstep+1] = 1 - (self.syn_r[:, self.curstep+1] +
                                             self.syn_e[:, self.curstep+1])
        """
        self.x[:, self.curstep+1] = self.xi + self.k1x * self.dt
        self.y[:, self.curstep+1] = self.yi + self.k1y * self.dt
        self.z[:, self.curstep+1] = self.zi + self.k1z * self.dt
        """

        # the fourth order Runge-Kutta method
        self.syn_r[:, :, self.curstep+1] = (self.syn_ri +
                                            (self.k1syn_r + 2*self.k2syn_r +
                                             2*self.k3syn_r + self.k4syn_r) *
                                            self.dt * 1/6)
        self.syn_e[:, :, self.curstep+1] = (self.syn_ei +
                                            (self.k1syn_e + 2*self.k2syn_e +
                                             2*self.k3syn_e + self.k4syn_e) *
                                            self.dt * 1/6)
        self.syn_i[:, :, self.curstep+1] = 1 - (self.syn_r[:, :, self.curstep+1] +
                                             self.syn_e[:, :, self.curstep+1])
        self.x[:, self.curstep+1] = (self.xi + (self.k1x + 2*self.k2x +
                                     2*self.k3x + self.k4x) * self.dt * 1/6)
        self.y[:, self.curstep+1] = (self.yi + (self.k1y + 2*self.k2y +
                                     2*self.k3y + self.k4y) * self.dt * 1/6)
        self.z[:, self.curstep+1] = (self.zi + (self.k1z + 2*self.k2z +
                                     2*self.k3z + self.k4z) * self.dt * 1/6)
        """

        self.curstep += 1