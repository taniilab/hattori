"""
Created on Wed May 24 11:37:33 2017

@author: Hattori
"""
# coding: UTF-8
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
# import seaborn as sb


class Neuron_HR():
    # constructor
    # 0.02
    def __init__(self, Syncp=1, numneu=1, dt=0.05, simtime=1000, a=1, b=3, c=1,
                 d=5, r=0.004, s=4, xr=-1.56, esyn=0, Pmax=3, tausyn=10,
                 xth=1.0, theta=-0.25, Iext=0, noise=0, ramda=-10, alpha=0.5,
                 beta=0, D=1, syn_delay=0):
        self.set_neuron_palm(Syncp, numneu, dt, simtime, a, b, c, d, r, s, xr,
                             esyn, Pmax, tausyn, xth, theta, Iext, noise,
                             ramda, alpha, beta, D, syn_delay)

    def set_neuron_palm(self, Syncp, numneu, dt, simtime, a, b, c, d, r, s, xr,
                        esyn, Pmax, tausyn, xth, theta, Iext, noise, ramda,
                        alpha, beta, D, syn_delay):
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

        # connection relationship
        self.cnct = np.ones((self.numneu, self.numneu))
        for i in range(0, self.numneu):
            self.cnct[i, i] = 1
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
        self.aptm = -100 * np.ones((self.numneu, self.numneu))

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
        # muximum synaptic conductance
        self.Pmax = Pmax
        self.syn_delay = syn_delay

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
            self.aptm[i, :] = self.curstep * self.dt

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

        elif self.Syncp == 3:
            pass

        # alpha function
        elif self.Syncp == 4:
            for j in range(0, self.numneu):
                self.gsyn[i, j] =\
                    self.alpha_function(self.curstep*self.dt - self.aptm[j, i] - self.syn_delay)
                    
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
        self.Isyni = self.Isyn[:, self.curstep]
        self.ni = self.n[:, self.curstep]
        self.dni = self.dn[:, self.curstep]

        # calculate synaptic input
        for i in range(0, self.numneu):
            self.calc_synaptic_input(i)

        # noise
        # gaussian white
        if self.noise == 1:
            self.n[:, self.curstep+1] = self.D * self.g[:, self.curstep]
        # Ornstein-Uhlenbeck process
        elif self.noise == 2:
            self.n[:, self.curstep+1] = (self.ni +
                                         (-self.alpha * (self.ni - self.beta) +
                                          self.D * self.g[:, self.curstep]) *
                                         self.dt)
        # oscillate
        elif self.noise == 3:
            self.n[:, self.curstep+1] = self.alpha * np.sin(self.curstep/10000)
        else:
            self.n[:, self.curstep+1] = 0

        self.k1x = (self.yi - self.ai * self.xi**3 + self.bi * self.xi**2 -
                    self.zi + self.Isyni +
                    self.Iext[:, self.curstep] + self.ni)
        self.k1y = (self.ci - self.di * self.xi**2 - self.yi)
        self.k1z = (self.ri * (self.si * (self.xi - self.xri) -
                    self.zi))

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
        self.x[:, self.curstep+1] = self.xi + self.k1x * self.dt
        self.y[:, self.curstep+1] = self.yi + self.k1y * self.dt
        self.z[:, self.curstep+1] = self.zi + self.k1z * self.dt
              """
        # the fourth order Runge-Kutta method

        self.x[:, self.curstep+1] = (self.xi + (self.k1x + 2*self.k2x +
                                     2*self.k3x + self.k4x) * self.dt * 1/6)
        self.y[:, self.curstep+1] = (self.yi + (self.k1y + 2*self.k2y +
                                     2*self.k3y + self.k4y) * self.dt * 1/6)
        self.z[:, self.curstep+1] = (self.zi + (self.k1z + 2*self.k2z +
                                     2*self.k3z + self.k4z) * self.dt * 1/6)

        self.curstep += 1