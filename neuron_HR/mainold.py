# -*- coding: utf-8 -*-
"""
Created on Wed May 24 11:37:33 2017

@author: Hattori
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sb

dt = 0.01
t = np.arange(0, 2000, dt)

a = 1
b = 3
c = 1
d = 5
r = 0.001
s = 4

dx = 0
dy = 0
dz = 0
xr = -1.6

x = [-2.0] * np.size(t)
y = [-5.0] * np.size(t)
z = [0] * np.size(t)
I = [2]* np.size(t)

for i in range(0, np.size(t)-1):
    dx = y[i] - a * x[i]*x[i]*x[i] + b * x[i]*x[i] - z[i] + I[i] + np.random.randn()
    dy = c - d * x[i] * x[i] - y[i]
    dz = r*(s*(x[i] - xr) - z[i])
    
    x[i+1] = x[i] + dt * dx
    y[i+1] = y[i] + dt * dy
    z[i+1] = z[i] + dt * dz

fig = plt.figure(figsize = (12, 18))
ax = fig.add_subplot(2, 1, 1)
lines = ax.plot(t, x)
plt.grid(True)

ax = fig.add_subplot(2, 1, 2, projection='3d')
lines = ax.plot(x, y, z)

plt.show()

#ベクトル平面
"""
x = np.arange(-3,3, 0.0001)
xnull = [0] * np.size(x)
ynull = [0] * np.size(x)
for i in range(0, np.size(x)):
    xnull[i] =  a*x[i]**3 - b*x[i]**2
    ynull[i] =  c - d * x[i]**2

lines, = ax.plot(x, xnull, markevery=10000)
lines, = ax.plot(x, ynull, markevery=10000)

v1 = np.arange(-2, 2, 0.1)
u1 = np.arange(-50, 20, 0.25)
V, U = np.meshgrid(v1, u1)

dV = U - a*V**3 + b*V**2 
dU = c - d*V**2 - U

#規格化
N = np.sqrt(dV**2 + dU**2)*10
dV = dV/N
dU = dU/N

ax.quiver(V, U, dV, dU, units = 'xy', width = 0.025)
ax.set_ylim(-10, 2.5)
ax.set_xlim(-3, 3)

fig.subplots_adjust(left=0.00, bottom=0.00, right=1.00, top=1.00, wspace=0.00, hspace=0.00)
fig.tight_layout(pad=0.05, w_pad=0.00, h_pad=0.00)
"""