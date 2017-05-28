# -*- coding: utf-8 -*-
"""
Created on Wed May 24 11:37:33 2017

@author: Hattori
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#import seaborn as sb

class Neuron_HR():
    #constructor
    def __init__(self, numneu=2, dt=0.01, simtime=3000, a=1, b=3, c=1, d=5, r=0.001, s=4, xr=-1.6, esyn=2, tausyn=2, xth=0, Iext=0):
        self.set_neuron_palm(numneu, dt, simtime, a, b, c, d, r, s, xr, esyn, tausyn, xth, Iext)
        
    def set_neuron_palm(self, numneu, dt, simtime, a, b, c, d, r, s, xr, esyn, tausyn, xth, Iext):
        self.numneu = numneu
        self.dt = dt
        self.simtime = simtime
        self.tmhist = np.arange(0, self.simtime, self.dt)
        self.allsteps = len(self.tmhist)
        self.a = a * np.ones((self.numneu, len(self.tmhist)))
        self.b = b * np.ones((self.numneu, len(self.tmhist)))
        self.c = c * np.ones((self.numneu, len(self.tmhist)))
        self.d = d * np.ones((self.numneu, len(self.tmhist)))
        self.r = r * np.ones((self.numneu, len(self.tmhist)))
        self.s = s * np.ones((self.numneu, len(self.tmhist)))
        self.xr = xr * np.ones((self.numneu, len(self.tmhist)))
        self.Isynapse = np.zeros(self.numneu)
        self.gsyn = np.zeros((self.numneu, self.numneu))
        self.esyn = esyn * np.ones((self.numneu, self.numneu))
        self.tausyn = tausyn * np.ones((self.numneu, self.numneu))
        self.apstep = -100 * np.ones((self.numneu, self.numneu))
    
    
        self.x = -1.2 * np.ones((self.numneu, len(self.tmhist)))
        self.y = -5 * np.ones((self.numneu, len(self.tmhist)))
        self.z = 0 * np.ones((self.numneu, len(self.tmhist)))
        self.Iext = Iext * np.ones((self.numneu, len(self.tmhist)))        
        self.dx = 0 * np.ones((self.numneu, len(self.tmhist)))
        self.dy = 0 * np.ones((self.numneu, len(self.tmhist)))
        self.dz = 0 * np.ones((self.numneu, len(self.tmhist)))
    
        #current step
        self.curstep = 0
        self.xth = xth
        self.aaa = "kashikoma"

    
    def alpha_function(self, t):
        if t<=0 or t>100:
            return 0
        else:
            return (t/self.tausyn[0,0]**2) * np.exp(-t/self.tausyn[0,0])
    
    
    # 1 step processing
    def propagation(self):
        #slice the current time step
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
        
        
        #calculate synaptic input        
        for i in range(0, self.numneu):
            #recording fire time  
            if self.xi[i] > self.xth:
                self.apstep[i, :] = self.curstep
         
            #sum of the synaptic current for each neuron
            for j in range(0, self.numneu):
                self.gsyn[i, j]= self.alpha_function(self.curstep - self.apstep[i, j])
                
            for j in range(0, self.numneu):
                self.Isynapse[i] += self.gsyn[i, j] * (self.xi[i] - self.esyn[i, j])
        
        self.dxi = (self.yi - self.ai * self.xi**3 + self.bi * self.xi**2 - self.zi + self.Isynapse + self.Iext[:, self.curstep]+ np.random.randn(self.numneu)/5) * self.dt
        self.dyi = (self.ci - self.di * self.xi**2 - self.yi) * self.dt
        self.dzi = (self.ri * (self.si * (self.xi - self.xri) - self.zi)) * self.dt
        
        #Euler first order approximation   
        self.x[:, self.curstep+1] = self.xi + self.dxi
        self.y[:, self.curstep+1] = self.yi + self.dyi                      
        self.z[:, self.curstep+1] = self.zi + self.dzi

        self.Isynapse = np.zeros(self.numneu)
        self.curstep+=1
        self.aaa = "puri"
            

        
       
"""    
fig = plt.figure(figsize = (12, 18))
ax = fig.add_subplot(2, 1, 1)
lines = ax.plot(t, x)
plt.grid(True)

ax = fig.add_subplot(2, 1, 2, projection='3d')
lines = ax.plot(x, y, z)

plt.show()
"""