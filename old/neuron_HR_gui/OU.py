# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 11:41:24 2018

@author: 6969p
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

fsize = 40

dt = 0.02
t = np.arange(0, 100, dt)
steps = len(t)
noise1 = np.zeros(steps)
noise2 = np.zeros(steps)
noise3 = np.zeros(steps)
noise4 = np.zeros(steps)
noise5 = np.zeros(steps)
noise6 = np.zeros(steps)


for i in range(0, steps-1):
    noise1[i+1] = noise1[i] + dt * (-0.5*(noise1[i] - 0) + np.random.randn())
    noise2[i+1] = noise2[i] + dt * (-0.5*(noise2[i] - 0) + 3*np.random.randn())
    noise3[i+1] = noise3[i] + dt * (-0.5*(noise3[i] - 0) + 6*np.random.randn())
    noise4[i+1] = noise4[i] + dt * (-0.5*(noise4[i] - 10) + np.random.randn())
    noise5[i+1] = noise5[i] + dt * (-0.5*(noise5[i] - 10) + 3*np.random.randn())
    noise6[i+1] = noise6[i] + dt * (-0.5*(noise6[i] - 10) + 6*np.random.randn())

plt.plot(t, noise3, label="θ:0.5  μ:0   D:6")
plt.plot(t, noise2, label="θ:0.5  μ:0   D:3")
plt.plot(t, noise1, label="θ:0.5  μ:0   D:1")
plt.plot(t, noise6, label="θ:0.5  μ:10  D:6")
plt.plot(t, noise5, label="θ:0.5  μ:10  D:3")
plt.plot(t, noise4, label="θ:0.5  μ:10  D:1")
plt.legend(fontsize=fsize, loc="right")
plt.xlabel("X-axis", fontsize=fsize)
plt.ylabel("Y-axis", fontsize=fsize)
plt.tick_params(labelsize=fsize)