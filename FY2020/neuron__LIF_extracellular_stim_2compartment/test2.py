# mV ms kΩ uF uA

import matplotlib.pyplot as plt
import numpy as np

# parameters
r1 = 400000
r2 = 50
r3 = 50
r4 = 400000
r5 = 5e-3
r6 = 5e-3
c1 = 6e-3
c2 = 100
c3 = 100
c4 = 6e-3
dt = 0.00001
tau = 0.01
v_amp = 2500

t = np.arange(0, 50, dt)
vdd = t*0

for k in range(int(tau/dt)):
    vdd[int((5/dt) + k):] -= v_amp*dt/tau
for k in range(int(2*tau/dt)):
    vdd[int((25/dt) + k):] += v_amp*dt/tau
vdd[int(45/dt):] = 0

#vdd[int(5/dt):] = -2500
#vdd[int(25/dt):] = 2500
vdd[int(45/dt):] = 0
vc1 = t*0
vc2 = t*0
vc3 = t*0
vc4 = t*0
dvc1 = 0
dvc2 = 0
dvc3 = 0
dvc4 = 0

for i in range(len(t)-1):
    print(i)
    i_total = (vdd[i]-vc1[i]-vc2[i]-vc3[i]-vc4[i])/r5 + (vdd[i]-vc2[i]-vc3[i])/r6
    i_link = (vdd[i]-vc2[i]-vc3[i])/r6

    dvc1 = dt*(1/c1)*(i_total - i_link - vc1[i]/r1)
    dvc2 = dt*(1/c2)*(i_total - vc2[i]/r2)
    dvc3 = dt*(1/c3)*(i_total - vc3[i]/r3)
    dvc4 = dt*(1/c4)*(i_total - i_link - vc4[i]/r4)

    vc1[i+1] = vc1[i] + dvc1
    vc2[i+1] = vc2[i] + dvc2
    vc3[i+1] = vc3[i] + dvc3
    vc4[i+1] = vc4[i] + dvc4

fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(111)
ax.plot(t, vdd, label="vdd")
ax.plot(t, vc1, label="vc1")
ax.plot(t, vc2, label="vc2")
ax.plot(t, vc3, label="vc3")
plt.legend()
plt.show()