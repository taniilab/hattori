"""
Created on Wed May 24 11:37:33 2017

@author: Hattori
"""
# coding: UTF-8
import numpy as np


class Neuron_HR():
    # constructor
    # 0.02
    def __init__(self, Syncp=1, numneu=1, dt=0.03, simtime=200,Cm=1, Ena=120,
                 Ek=-12, El=10.6, gna=120, gk=36, gl=0.3, esyn=0, Pmax=3, tausyn=10,
                 xth=0.25, theta=-0.25, Iofs=0, Iext_amp=0, Iext_width=0, Iext_duty=0,
                 Iext_num=0, noise=0, ramda=-10, alpha=0.5,
                 beta=0, D=1,
                 tau_r=50, tau_i=50, use=1, ase=1, gcmp=2, delay=0, gsyn=0):
        self.set_neuron_palm(Syncp, numneu, dt, simtime, Cm, Ena, Ek, El,
                             gna, gk, gl, esyn, Pmax, tausyn, xth, theta, Iext_amp,
                             Iofs, Iext_width, Iext_duty, Iext_num, noise,
                             ramda, alpha, beta, D, tau_r, tau_i, use, ase, gcmp, delay, gsyn)

    def set_neuron_palm(self, Syncp, numneu, dt, simtime, Cm, Ena, Ek, El,
                             gna, gk, gl, esyn, Pmax, tausyn, xth, theta, Iext_amp,
                             Iofs, Iext_width, Iext_duty, Iext_num, noise,
                             ramda, alpha, beta, D, tau_r, tau_i, use, ase, gcmp, delay, gsyn):
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

        # HH model
        self.V = 0 * np.ones((self.numneu, len(self.tmhist)))
        self.Vth = 0
        self.m = 0 * np.ones((self.numneu, len(self.tmhist)))
        self.h = 0 * np.ones((self.numneu, len(self.tmhist)))
        self.n = 0 * np.ones((self.numneu, len(self.tmhist)))
        self.alpha_m = 0 * np.ones((self.numneu, len(self.tmhist)))
        self.alpha_h = 0 * np.ones((self.numneu, len(self.tmhist)))
        self.alpha_n = 0 * np.ones((self.numneu, len(self.tmhist)))
        self.beta_m = 0 * np.ones((self.numneu, len(self.tmhist)))
        self.beta_h = 0 * np.ones((self.numneu, len(self.tmhist)))
        self.beta_n = 0 * np.ones((self.numneu, len(self.tmhist)))
        self.Ena = Ena * np.ones((self.numneu))
        self.Ek = Ek * np.ones((self.numneu))
        self.El = El * np.ones((self.numneu))
        self.gna = gna * np.ones((self.numneu))
        self.gk = gk * np.ones((self.numneu))
        self.gl = gl * np.ones((self.numneu))
        self.k1V = 0 * np.ones(self.numneu)
        self.k1m = 0 * np.ones(self.numneu)
        self.k1h = 0 * np.ones(self.numneu)
        self.k1n = 0 * np.ones(self.numneu)
        self.k2V = 0 * np.ones(self.numneu)
        self.k2m = 0 * np.ones(self.numneu)
        self.k2h = 0 * np.ones(self.numneu)
        self.k2n = 0 * np.ones(self.numneu)
        self.k3V = 0 * np.ones(self.numneu)
        self.k3m = 0 * np.ones(self.numneu)
        self.k3h = 0 * np.ones(self.numneu)
        self.k3n = 0 * np.ones(self.numneu)
        self.k4V = 0 * np.ones(self.numneu)
        self.k4m = 0 * np.ones(self.numneu)
        self.k4h = 0 * np.ones(self.numneu)
        self.k4n = 0 * np.ones(self.numneu)

        # compartment conductance
        self.gcmp = gcmp
        self.delay = delay

        # connection relationship
        self.cnct = np.zeros((self.numneu, self.numneu))

        # synaptic current
        self.Isyn = np.zeros((self.numneu, len(self.tmhist)))
        self.Isyn_hist = np.zeros((self.numneu, self.numneu, 5))

        # synaptic conductance
        self.gsyn = gsyn * np.ones((self.numneu, self.numneu))
        # synaptic reversal potential
        self.esyn = esyn * np.ones((self.numneu, self.numneu))
        self.tausyn = tausyn
        # external current
        self.Iext = np.zeros((self.numneu, len(self.tmhist)))

        # square wave
        self.Iext_amp = Iext_amp
        self.Iext = 0 * np.ones((self.numneu, len(self.tmhist)))
        for i in range(len(self.tmhist)):
            self.Iext[0, :] = self.Iext_amp
        """
        self.Iext_co = 0
        self.Iext_amp = Iext_amp
        self.Iext_width = Iext_width
        self.Iext_duty = Iext_duty
        while self.Iext_co < Iext_num:
            if self.Iext_duty == 0:
                self.Iext[0, (1000/self.dt):(1500/self.dt)] = Iext_amp
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
        if self.Vi[i] > self.Vth and (self.curstep *
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
                if self.curstep > 30/self.dt:
                    self.Isyni[i] +=\
                      ( self.gsyn[i, j] *
                       (self.V[j, self.curstep] -
                        self.V[i, self.curstep]))
                else:
                    self.Isyni[i] = 0
                """

                self.Isyni[i] +=\
                    (self.cnct[i, j] * self.discrete_delta_func((self.curstep - (self.t_ap[j, j, 0]/self.dt))))
                """
            else:
                self.Isyni[i] +=\
                          (self.cnct[i, j] * self.gsyn[i, j] *
                           (self.esyn[i, j] - self.Vi[i]))

    # one step processing
    def propagation(self):
        # slice the current time step
        self.Vi = self.V[:, self.curstep]
        self.mi = self.m[:, self.curstep]
        self.hi = self.h[:, self.curstep]
        self.ni = self.n[:, self.curstep]
        self.alpha_mi = self.alpha_m[:, self.curstep]
        self.alpha_hi = self.alpha_h[:, self.curstep]
        self.alpha_ni = self.alpha_n[:, self.curstep]
        self.beta_mi = self.beta_m[:, self.curstep]
        self.beta_hi = self.beta_h[:, self.curstep]
        self.beta_ni = self.beta_n[:, self.curstep]
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

        else:
            pass

        # solve a defferential equation
        self.alpha_mi = 0.1*(25-self.Vi)/(np.exp((25-self.Vi)/10)-1)
        self.alpha_hi = 0.07*np.exp(-self.Vi/20)
        self.alpha_ni = 0.01*(10-self.Vi)/(np.exp((10-self.Vi)/10)-1)
        self.beta_mi = 4*np.exp(-self.Vi/18)
        self.beta_hi = 1/(np.exp((30-self.Vi)/10)+1)
        self.beta_ni = 0.125*np.exp(-self.Vi/80)

        self.k1V = (self.gk * self.ni**4 * (self.Ek - self.Vi) +
                    self.gna * self.mi**3 * self.hi * (self.Ena - self.Vi) +
                    self.gl * (self.El - self.Vi) + self.Isyni +
                    self.Iext[:, self.curstep] + self.ni)
        self.k1m = self.alpha_mi * (1-self.mi) - self.beta_mi * self.mi
        self.k1h = self.alpha_hi * (1-self.hi) - self.beta_hi * self.hi
        self.k1n = self.alpha_ni * (1-self.ni) - self.beta_ni * self.ni

        """
        self.k2x = ((self.yi + (self.dt/2) * self.k1y) - self.ai *
                    (self.xi + (self.dt/2) * self.k1x)**3 + self.bi *
                    (self.xi + (self.dt/2) * self.k1x)**2 -
                    (self.zi + (self.dt/2) * self.k1z) + self.Isyni +
                    self.Iext[:, self.curstep] + self.ni + self.Iofs)
        self.k2y = (self.ci - self.di * (self.xi + (self.dt/2) *
                    self.k1x)**2 - (self.yi + (self.dt/2) * self.k1y))
        self.k2z = (self.ri * (self.si * ((self.xi + (self.dt/2) *
                    self.k1x) - self.xri) -
                    (self.zi + (self.dt/2) * self.k1z)))

        self.k3x = ((self.yi + (self.dt/2) * self.k2y) - self.ai *
                    (self.xi + (self.dt/2) * self.k2x)**3 + self.bi *
                    (self.xi + (self.dt/2) * self.k2x)**2 -
                    (self.zi + (self.dt/2) * self.k2z) + self.Isyni +
                    self.Iext[:, self.curstep] + self.ni + self.Iofs)
        self.k3y = (self.ci - self.di * (self.xi + (self.dt/2) *
                    self.k2x)**2 - (self.yi + (self.dt/2) * self.k2y))
        self.k3z = (self.ri * (self.si * ((self.xi + (self.dt/2) *
                    self.k2x) - self.xri) -
                    (self.zi + (self.dt/2) * self.k2z)))

        self.k4x = ((self.yi + (self.dt) * self.k3y) - self.ai *
                    (self.xi + (self.dt) * self.k3x)**3 + self.bi *
                    (self.xi + (self.dt) * self.k3x)**2 -
                    (self.zi + (self.dt) * self.k3z) + self.Isyni +
                    self.Iext[:, self.curstep] + self.ni + self.Iofs)
        self.k4y = (self.ci - self.di * (self.xi + (self.dt) *
                    self.k3x)**2 - (self.yi + (self.dt) * self.k3y))
        self.k4z = (self.ri * (self.si * ((self.xi + (self.dt) *
                    self.k3x) - self.xri) -
                    (self.zi + (self.dt) * self.k3z)))
        """
        # the first order Euler method
        self.V[:, self.curstep+1] = self.Vi + self.k1V * self.dt
        self.m[:, self.curstep+1] = self.mi + self.k1m * self.dt
        self.h[:, self.curstep+1] = self.hi + self.k1h * self.dt
        self.n[:, self.curstep+1] = self.ni + self.k1n * self.dt
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
