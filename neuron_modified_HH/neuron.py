"""
Created on Wed May 24 11:37:33 2017

@author: Hattori
"""
# coding: UTF-8
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
# import seaborn as sb


class Neuron_mondify_HH():
    # constructor
    def __init__(self, Syncp=1, numneu=1, dt=0.03, simtime=6000,  esyn=0, Pmax=1, tausyn=10,
                 xth=1.0, theta=-0.25, Iext=0, noise=0, ramda=-10, alpha=0.5,
                 beta=0, D=1):
        self.set_neuron_palm(Syncp, numneu, dt, simtime,
                             esyn, Pmax, tausyn, xth, theta, Iext, noise,
                             ramda, alpha, beta, D)

    def set_neuron_palm(self, Syncp, numneu, dt, simtime,
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
        # HH model
        #membrane potential
        self.vin = np.ones((self.numneu, len(self.tmhist))) * -60      
        #threshold voltage(mV)        
        self.vth = np.ones((self.numneu, len(self.tmhist))) * vth
        #sodium current
        self.ina = np.zeros((self.numneu, len(self.tmhist)))
        #potassium current
        self.ikd = np.zeros((self.numneu, len(self.tmhist)))
        #slow potassium current for spike-frequency
        self.im = np.zeros((self.numneu, len(self.tmhist)))
        #E/I PSC(excitatory/inhibitory post synaptic current)
        self.isyn = np.zeros((self.numneu, len(self.tmhist)))
        #noise current
        self.inoise = np.zeros((self.numneu, len(self.tmhist)))
        #membrane conductance(uf)
        self.cm = np.ones((self.numneu, len(self.tmhist))) * cm
        #leak conductance
        self.gleak = np.ones((self.numneu, len(self.tmhist))) * gleak * n_to_m
        #leak reversal potential
        self.eleak =  np.ones((self.numneu, len(self.tmhist))) * eleak    
        #maximul sodium conductance
        self.gna = np.ones((self.numneu, len(self.tmhist))) * gna
        #maximul potassium conductance
        self.gkd = np.ones((self.numneu, len(self.tmhist))) * gkd
        #maximul sodium conductance for frequency adaptation
        self.gm = np.ones((self.numneu, len(self.tmhist))) * gm
        #synaptic conductance
        self.gsyn = np.zeros((self.numneu, self.numneu))
        #time constant for frequency adaptation
        self.τmax = np.ones((self.numneu, len(self.tmhist))) * τmax        
        #sodium reversal potential
        self.ena = np.ones((self.numneu, len(self.tmhist))) * ena
        #potassium reversal potential
        self.ek = np.ones((self.numneu, len(self.tmhist))) * ek
        #synaptic reversal potential
        self.esyn = np.ones((self.numneu, self.numneu)) * esyn
        #time constant for alpha function
        self.τs = np.ones((self.numneu, self.numneu)) * τs
        #channnel variables
        self.αm = np.zeros(self.numneu)
        self.βm = np.zeros(self.numneu)
        self.αh = np.zeros(self.numneu)
        self.βh = np.zeros(self.numneu)        
        self.αn = np.zeros(self.numneu)
        self.βn = np.zeros(self.numneu)
        self.m = np.zeros((self.numneu, len(self.tmhist)))
        self.h = np.zeros((self.numneu, len(self.tmhist)))
        self.n = np.zeros((self.numneu, len(self.tmhist)))
        self.p = np.zeros((self.numneu, len(self.tmhist)))

        self.dvin = np.zeros(self.numneu)  
        self.dm = np.zeros(self.numneu)  
        self.dh = np.zeros(self.numneu)  
        self.dn = np.zeros(self.numneu)    

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
        
    def double_exp_function(self, t):
        return 0

    def synaptic_current(self, i):
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
        self.vintp = self.vin[:, self.curstep]
        self.vnext = self.vin[:, self.nextstep]        
        self.vthtp = self.vth[:, self.curstep]
        self.inatp = self.ina[:, self.curstep]
        self.ikdtp = self.ikd[:, self.curstep]
        self.imtp = self.im[:, self.curstep]
        self.isyntp = self.isyn[:, self.curstep]
        self.cmtp = self.cm[:, self.curstep]
        self.gleaktp = self.gleak[:, self.curstep]
        self.eleaktp = self.eleak[:, self.curstep]     
        self.gnatp = self.gna[:, self.curstep]      
        self.gkdtp = self.gkd[:, self.curstep]      
        self.gmtp = self.gm[:, self.curstep]
        self.τmaxtp = self.τmax[:, self.curstep]       
        self.enatp = self.ena[:, self.curstep]
        self.ektp = self.ek[:, self.curstep]
        self.mtp = self.m[:, self.curstep]
        self.htp = self.h[:, self.curstep]
        self.ntp = self.n[:, self.curstep]
        self.ptp = self.p[:, self.curstep]
        self.inoisetp = self.inoise[:, self.curstep]
        self.iexttp = self.iext[:, self.curstep]
        self.meantp = self.mean[:, self.curstep]
        self.variancetp = self.variance[:, self.curstep]
        self.mnext = self.m[:, self.nextstep]        
        self.hnext = self.h[:, self.nextstep]
        self.nnext = self.n[:, self.nextstep]
        self.pnext = self.p[:, self.nextstep]

        ###hodgkin-huxley type equation###
        self.αm = -0.32 * (self.vintp-self.vthtp-13)/(np.exp(-1 * (self.vintp-self.vthtp-13)/4) - 1) 
        self.βm = 0.28 * (self.vintp-self.vthtp-40)/(np.exp((self.vintp-self.vthtp-40)/5) - 1) 
        self.αh = 0.128 * np.exp(-1 * (self.vintp-self.vthtp-17)/18)
        self.βh = 4/(1 + np.exp(-1 * (self.vintp-self.vthtp-40)/5))
        self.αn = -0.032 * (self.vintp-self.vthtp-15)/(np.exp(-1 * (self.vintp-self.vthtp-15)/5) - 1) 
        self.βn = 0.5 * np.exp(-1 * (self.vintp-self.vthtp-10)/40)
        self.pinf = 1/(1+np.exp(-1 * (self.vintp+35)/10))
        self.τp = self.τmaxtp/(3.3*np.exp((self.vintp+35)/20) + np.exp(-1 * (self.vintp+35)/20))
        
        if self.curstep == 0:
            self.mtp[:] = self.αm/(self.αm + self.βm)
            self.htp[:] = self.αh/(self.αh + self.βh)  
            self.ntp[:] = self.αn/(self.αn + self.βn)

        # calculate synaptic input
        for i in range(0, self.numneu):
            self.synaptic_current(i)

        # calculate voltage-dependent potasium current
        
        # calculate voltage-dependent sodium current

        # calculate leak current
        
        
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
            self.n[:, self.curstep+1] = 0] + self.ni) * self.dt
        self.dyi = (self.ci - self.di * self.xi**2 - self.yi) * self.dt
        self.dzi = (self.ri * (self.si * (self.xi - self.xri) -
                    self.zi)) * self.dt

        self.dxi = (self.yi - self.ai * self.xi**3 + self.bi * self.xi**2 -
                    self.zi + self.Isyni +
                    self.Iext[:, self.curstep

        self.dvin[]

        # Euler first order approximation
        self.vin[:, self.curstep+1] = self.vin + self.dvi
        self.n[:, self.curstep+1] = self.n + self.dn
        self.m[:, self.curstep+1] = self.m + self.dm
        self.h[:, self.curstep+1] = self.h + self.h

        self.curstep += 1
