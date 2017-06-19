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
    def __init__(self, Syncp=1, numneu=1, dt=0.02, simtime=3000, a=1, b=3, c=1,
                 d=5, r=0.001, s=4, xr=-1.56, esyn=0, Pmax=1, tausyn=10,
                 xth=1.0, theta=-0.25, Iext=0, noise=0, ramda=-10, alpha=0.5,
                 beta=0, D=1):
        self.set_neuron_palm(Syncp, numneu, dt, simtime, a, b, c, d, r, s, xr,
                             esyn, Pmax, tausyn, xth, theta, Iext, noise,
                             ramda, alpha, beta, D)

    def set_neuron_palm(self, Syncp, numneu, dt, simtime, a, b, c, d, r, s, xr,
                        esyn, Pmax, tausyn, xth, theta, Iext, noise, ramda,
                        alpha, beta, D):
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
        self.dx = 0 * np.ones((self.numneu, len(self.tmhist)))
        self.dy = 0 * np.ones((self.numneu, len(self.tmhist)))
        self.dz = 0 * np.ones((self.numneu, len(self.tmhist)))
        # connection relationship between neurons
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
        self.Iext = Iext * np.ones((self.numneu, len(self.tmhist)))
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

        elif self.Syncp == 2:
            for j in range(0, self.numneu):
                self.gsyn[i, j] =\
                    (self.Pmax *
                     (1 /
                      (1 +
                       np.exp(self.ramda *
                              (self.x[j, self.curstep-self.tausyn] -
                               self.theta)))))
            for j in range(0, self.numneu):
                self.Isyni[i] +=\
                    (self.cnct[i, j] * self.gsyn[i, j] *
                     (self.esyn[i, j] - self.xi[i]))

        elif self.Syncp == 3:
            pass

        elif self.Syncp == 4:
            for j in range(0, self.numneu):
                self.gsyn[i, j] =\
                    self.alpha_function(self.curstep*self.dt - self.aptm[j, i])
            for j in range(0, self.numneu):
                self.Isyni[i] +=\
                    (self.cnct[i, j] * self.gsyn[i, j] *
                     (self.esyn[i, j] - self.xi[i]))

        else:
            pass

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
        self.dxi = self.dx[:, self.curstep]
        self.dyi = self.dy[:, self.curstep]
        self.dzi = self.dz[:, self.curstep]
        self.Isyni = self.Isyn[:, self.curstep]
        self.ni = self.n[:, self.curstep]
        self.dni = self.dn[:, self.curstep]

        # calculate synaptic input
        for i in range(0, self.numneu):
            self.calc_synaptic_input(i)

        # noise
        if self.noise == 1:
            self.n[:, self.curstep+1] = self.D * self.g[:, self.curstep]
        elif self.noise == 2:
            self.n[:, self.curstep+1] = (self.ni +
                                         (-self.alpha * (self.ni - self.beta) +
                                          self.D * self.g[:, self.curstep]) *
                                         self.dt)
        elif self.noise == 3:
            self.n[:, self.curstep+1] = self.alpha * np.sin(self.curstep/10000)
        else:
            self.n[:, self.curstep+1] = 0

        self.dxi = (self.yi - self.ai * self.xi**3 + self.bi * self.xi**2 -
                    self.zi + self.Isyni +
                    self.Iext[:, self.curstep] + self.ni) * self.dt
        self.dyi = (self.ci - self.di * self.xi**2 - self.yi) * self.dt
        self.dzi = (self.ri * (self.si * (self.xi - self.xri) -
                    self.zi)) * self.dt

        # Euler first order approximation
        self.x[:, self.curstep+1] = self.xi + self.dxi
        self.y[:, self.curstep+1] = self.yi + self.dyi
        self.z[:, self.curstep+1] = self.zi + self.dzi

        self.curstep += 1
