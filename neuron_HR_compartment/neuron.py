"""
Created on Wed May 24 11:37:33 2017

@author: Hattori
"""
# coding: UTF-8
import numpy as np


class Neuron_HR():
    # constructor
    # 0.02
    def __init__(self, Syncp=1, numneu=1, dt=0.05, simtime=1000, a=1, b=3.15,
                 c=1, d=5, r=0.004, s=4, xr=-1.6, esyn=0, Pmax=3, tausyn=10,
                 xth=1.0, theta=-0.25, Iext=0, noise=0, ramda=-10, alpha=0.5,
                 beta=0, D=1,
                 tau_r=50, tau_i=50, use=1, ase=1):
        self.set_neuron_palm(Syncp, numneu, dt, simtime, a, b, c, d, r, s, xr,
                             esyn, Pmax, tausyn, xth, theta, Iext, noise,
                             ramda, alpha, beta, D, tau_r, tau_i, use, ase)

    def set_neuron_palm(self, Syncp, numneu, dt, simtime, a, b, c, d, r, s, xr,
                        esyn, Pmax, tausyn, xth, theta, Iext, noise, ramda,
                        alpha, beta, D, tau_r, tau_i, use, ase):
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
        self.k2x = 0 * np.ones(self.numneu)
        self.k2y = 0 * np.ones(self.numneu)
        self.k2z = 0 * np.ones(self.numneu)
        self.k3x = 0 * np.ones(self.numneu)
        self.k3y = 0 * np.ones(self.numneu)
        self.k3z = 0 * np.ones(self.numneu)
        self.k4x = 0 * np.ones(self.numneu)
        self.k4y = 0 * np.ones(self.numneu)
        self.k4z = 0 * np.ones(self.numneu)

        # depressing synapse
        self.tau_r = tau_r * np.ones(self.numneu)
        self.tau_i = tau_i * np.ones(self.numneu)
        self.use = use * np.ones(self.numneu)
        self.ase = ase * np.ones(self.numneu)
        self.syn_r = 1 * np.ones((self.numneu, self.numneu, len(self.tmhist)))
        self.syn_e = 0 * np.ones((self.numneu, self.numneu, len(self.tmhist)))
        self.syn_i = 0 * np.ones((self.numneu, self.numneu, len(self.tmhist)))
        self.k1syn_r = 0 * np.ones(self.numneu)
        self.k1syn_e = 0 * np.ones(self.numneu)
        self.k2syn_r = 0 * np.ones(self.numneu)
        self.k2syn_e = 0 * np.ones(self.numneu)
        self.k3syn_r = 0 * np.ones(self.numneu)
        self.k3syn_e = 0 * np.ones(self.numneu)
        self.k4syn_r = 0 * np.ones(self.numneu)
        self.k4syn_e = 0 * np.ones(self.numneu)

        # connection relationship
        self.cnct = np.zeros((self.numneu, self.numneu))

        # synaptic current
        self.Isyn = np.zeros((self.numneu, len(self.tmhist)))
        # synaptic conductance
        self.gsyn = np.zeros((self.numneu, self.numneu))
        # synaptic reversal potential
        self.esyn = esyn * np.ones((self.numneu, self.numneu))
        self.tausyn = tausyn
        # external current
        self.Iext = np.zeros((self.numneu, len(self.tmhist)))
        self.Iext[0, :] = Iext
        # firing time
        self.t_ap = -100 * np.ones((self.numneu, self.numneu))

        # current step
        self.curstep = 0
        # thresholds
        self.xth = xth
        self.theta = theta
        # noise palameter
        self.noise = noise
        self.n = np.ones((self.numneu, len(self.tmhist)))

        """
        self.tmp = int(1000/self.dt)
        for i in range(self.tmp):
            self.n[:, i] = -Iext
            self.n[:, i+2*self.tmp] = -Iext
        """

        self.dn = np.zeros((self.numneu, len(self.tmhist)))
        self.ramda = ramda
        self.alpha = alpha
        self.beta = beta
        self.D = D
        self.g = np.random.randn(self.numneu, len(self.tmhist))

        # chemical synapse and alpha function
        self.Pmax = Pmax
        
        print(self.a[0, 0])
        print(self.b[0, 0])
        print(self.c[0, 0])
        print(self.d[0, 0])
        print(self.r[0, 0])
        print(self.s[0, 0])
        print(self.xr[0, 0])
        print("\n")
        print(self.cnct)

    def synaptic_connection(self):
        #self.cnct[0, 0] = 0.0
        self.cnct[1, 0] = 1.0
        #self.cnct[1, 1] = 0.0
        self.cnct[0, 1] = 1.0

    def delta_func(self, t):
        y = t == 0
        return y.astype(np.int)

    def alpha_function(self, t):
        if t <= 0:
            return 0
        elif ((self.Pmax * t/self.tausyn*0.1) *
              np.exp(-t/self.tausyn*0.1)) < 0.001:
            return 0
        else:
            return (self.Pmax * t/self.tausyn) * np.exp(-t/self.tausyn)

    def calc_synaptic_input(self, i):
        # recording fire time
        if self.xi[i] > self.xth:
            self.t_ap[i, :] = self.curstep * self.dt

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
                    self.alpha_function(self.curstep*self.dt - self.t_ap[j, i])

        elif self.Syncp == 5:
            pass

        # summation
        for j in range(0, self.numneu):
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
        self.syn_ri = self.syn_r[:, :, self.curstep]
        self.syn_ei = self.syn_e[:, :, self.curstep]
        self.syn_ii = self.syn_i[:, :, self.curstep]
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
        # 4 : square wave
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

        elif self.noise == 4:
            self.n[:, self.curstep+1] = 0

        else:
            # self.n[:, self.curstep+1] = 0
            pass

        # solve a defferential equation
        self.k1syn_r = ((self.syn_ii/self.tau_r) - self.use * self.syn_ri *
                        self.delta_func(self.curstep * self.dt - self.t_ap))
        self.k1syn_e = -((self.syn_ei/self.tau_i) + self.use * self.syn_ri *
                         self.delta_func(self.curstep*self.dt - self.t_ap))
        self.k1x = (self.yi - self.ai * self.xi**3 + self.bi * self.xi**2 -
                    self.zi + self.Isyni +
                    self.Iext[:, self.curstep] + self.ni)
        self.k1y = (self.ci - self.di * self.xi**2 - self.yi)
        self.k1z = (self.ri * (self.si * (self.xi - self.xri) -
                    self.zi))

        self.k2syn_r = (((1 - self.syn_ri - (self.dt/2) * self.k1syn_r -
                          self.syn_ei - (self.dt/2) * self.k1syn_e) /
                         self.tau_r) - self.use *
                        (self.syn_ri + (self.dt/2) * self.k1syn_r) *
                        self.delta_func(self.curstep * self.dt - self.t_ap))
        self.k2syn_e = -(((self.syn_ei + (self.dt/2) * self.k1syn_e) /
                          self.tau_r) + self.use *
                         (self.syn_ri + (self.dt/2) * self.k1syn_r) *
                         self.delta_func(self.curstep * self.dt - self.t_ap))
        self.k2x = ((self.yi + (self.dt/2) * self.k1y) - self.ai *
                    (self.xi + (self.dt/2) * self.k1x)**3 + self.bi *
                    (self.xi + (self.dt/2) * self.k1x)**2 -
                    (self.zi + (self.dt/2) * self.k1z) + self.Isyni +
                    self.Iext[:, self.curstep] + self.ni)
        self.k2y = (self.ci - self.di * (self.xi + (self.dt/2) *
                    self.k1x)**2 - (self.yi + (self.dt/2) * self.k1y))
        self.k2z = (self.ri * (self.si * ((self.xi + (self.dt/2) *
                    self.k1x) - self.xri) -
                    (self.zi + (self.dt/2) * self.k1z)))

        self.k3syn_r = (((1 - self.syn_ri - (self.dt/2) * self.k2syn_r -
                          self.syn_ei - (self.dt/2) * self.k2syn_e) /
                         self.tau_r) - self.use *
                        (self.syn_ri + (self.dt/2) * self.k2syn_r) *
                        self.delta_func(self.curstep * self.dt - self.t_ap))
        self.k3syn_e = -(((self.syn_ei + (self.dt/2) * self.k2syn_e) /
                          self.tau_r) + self.use *
                         (self.syn_ri + (self.dt/2) * self.k2syn_r) *
                         self.delta_func(self.curstep * self.dt - self.t_ap))
        self.k3x = ((self.yi + (self.dt/2) * self.k2y) - self.ai *
                    (self.xi + (self.dt/2) * self.k2x)**3 + self.bi *
                    (self.xi + (self.dt/2) * self.k2x)**2 -
                    (self.zi + (self.dt/2) * self.k2z) + self.Isyni +
                    self.Iext[:, self.curstep] + self.ni)
        self.k3y = (self.ci - self.di * (self.xi + (self.dt/2) *
                    self.k2x)**2 - (self.yi + (self.dt/2) * self.k2y))
        self.k3z = (self.ri * (self.si * ((self.xi + (self.dt/2) *
                    self.k2x) - self.xri) -
                    (self.zi + (self.dt/2) * self.k2z)))

        self.k4syn_r = (((1 - self.syn_ri - self.dt * self.k1syn_r -
                          self.syn_ei - self.dt * self.k3syn_e) /
                         self.tau_r) - self.use *
                        (self.syn_ri + self.dt * self.k1syn_r) *
                        self.delta_func(self.curstep * self.dt - self.t_ap))
        self.k4syn_e = -(((self.syn_ei + self.dt * self.k3syn_e) /
                          self.tau_r) + self.use *
                         (self.syn_ri + self.dt * self.k3syn_r) *
                         self.delta_func(self.curstep * self.dt - self.t_ap))
        self.k4x = ((self.yi + (self.dt) * self.k3y) - self.ai *
                    (self.xi + (self.dt) * self.k3x)**3 + self.bi *
                    (self.xi + (self.dt) * self.k3x)**2 -
                    (self.zi + (self.dt) * self.k3z) + self.Isyni +
                    self.Iext[:, self.curstep] + self.ni)
        self.k4y = (self.ci - self.di * (self.xi + (self.dt) *
                    self.k3x)**2 - (self.yi + (self.dt) * self.k3y))
        self.k4z = (self.ri * (self.si * ((self.xi + (self.dt) *
                    self.k3x) - self.xri) -
                    (self.zi + (self.dt) * self.k3z)))

        # the first order Euler method
        """
        self.syn_r[:, self.curstep+1] = self.syn_ri + self.k1syn_r * self.dt
        self.syn_e[:, self.curstep+1] = self.syn_ei + self.k1syn_e * self.dt
        self.syn_i[:, self.curstep+1] = 1 - (self.syn_r[:, self.curstep+1] +
                                             self.syn_e[:, self.curstep+1])
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

        self.curstep += 1
