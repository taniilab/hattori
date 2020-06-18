# mV ms kΩ uF uA

import matplotlib.pyplot as plt
import numpy as np

# parameters
r1 = 50e-3
r2 = 50e-3
r3 = 50e-3
c1 = 100
c2 = 200
c3 = 300
dt = 0.1
tau = 1
v_amp = 2500

t = np.arange(0, 50, dt)
vdd = t*0
"""
for k in range(int(tau/dt)):
    vdd[int((5/dt) + k):] += v_amp*dt/tau
"""
vdd[int(2/dt):] = -2500
vdd[int(25/dt):] = 2500
vdd[int(45/dt):] = 0
vc1 = t*0
vc2 = t*0
vc3 = t*0
dvc1 = 0
dvc2 = 0
dvc3 = 0

for i in range(len(t)-1):
    print(i)
    """
    print(i)
    print("time: {}".format(t[i]))
    print("vdd: {}".format(vdd[i]))
    print("i_vdd: {}".format(i_vdd))
    print("i_r4: {}".format(i_r4))
    print("dvc1: {}".format(dvc1))
    print("dvc2: {}".format(dvc2))
    print("dvc3: {}".format(dvc3))
    print("dvc4: {}\n".format(dvc4))
    """

    dvc1 = dt*(1/c1)*(vdd[i] - vc1[i] - vc2[i] - vc3[i]) / r1
    dvc2 = dt*(1/c2)*(-vc2[i]/r2 + (vdd[i] - vc1[i] - vc2[i] - vc3[i]) / r1)
    dvc3 = dt*(1/c3)*(-vc3[i]/r3 + (vdd[i] - vc1[i] - vc2[i] - vc3[i]) / r1)

    vc1[i+1] = vc1[i] + dvc1
    vc2[i+1] = vc2[i] + dvc2
    vc3[i+1] = vc3[i] + dvc3


fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(111)
ax.plot(t, vdd, label="vdd")
ax.plot(t, vc1, label="vc1")
ax.plot(t, vc2, label="vc2")
ax.plot(t, vc3, label="vc3")
plt.legend()
plt.show()