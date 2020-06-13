# mV ms kΩ uF

import matplotlib.pyplot as plt
import numpy as np

r8 = 50
r5 = 50e-3
c3 = 100
dt = 0.1
t = np.arange(0, 100, dt)
vdd = t*0
vdd[200:] = 5000
vdd[500:] = -5000
vc = t*0

for i in range(len(t)-1):
    dv = dt*(1/c3)*((vdd[i]-vc[i])/r5-(vc[i]/r8))
    vc[i+1] = vc[i] + dv

fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(111)
ax.plot(t, vc)
plt.show()