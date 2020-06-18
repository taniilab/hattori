# mV ms kΩ uF uA

import matplotlib.pyplot as plt
import numpy as np

# parameters
r1 = 400000
r2 = 400000
r3 = 200e-6
r4 = 200e-6
r5 = 50e-3
r6 = 50e-3
r8 = 50
r9 = 50
c1 = 6e-6
c2 = 6e-6
c3 = 100
c4 = 100
dt = 0.01
vrest = -70

t = np.arange(0, 20.02, dt)
vdd = t*0
vdd[int(20/dt):] = 1
#vdd[500:] = -5000
vc1 = t*0
vc2 = t*0
vc3 = t*0
vc4 = t*0
dvc1 = 0
dvc2 = 0
dvc3 = 0
dvc4 = 0
i_vdd = 0
i_r4 = 0

for i in range(int(20.02/dt)-1):
    print(i)
    print("time: {}".format(t[i]))
    print("vdd: {}".format(vdd[i]))
    print("i_vdd: {}".format(i_vdd))
    print("i_r4: {}".format(i_r4))
    print("dvc1: {}".format(dvc1))
    print("dvc2: {}".format(dvc2))
    print("dvc3: {}".format(dvc3))
    print("dvc4: {}\n".format(dvc4))

    i_vdd = ((r3+r4)*(vdd[i]-vc3[i]-vc4[i])-r4*(vc1[i]+vc2[i])) / \
            (r3*r4+(r3+r4)*(r5+r6))
    i_r4 = (r3*(vdd[i]-vc3[i]-vc4[i])+(r5+r6)*(vc1[i]+vc2[i])) / \
           (r3*r4+(r3+r4)*(r5+r6))

    dvc1 = dt*(1/c1)*(-(vc1[i]+vrest)/r1 + i_vdd - i_r4)
    dvc2 = dt*(1/c2)*(-(vc2[i]-vrest)/r2 + i_vdd - i_r4)
    dvc3 = dt*(1/c3)*(-vc3[i]/r8 + i_vdd)
    dvc4 = dt*(1/c4)*(-vc4[i]/r9 + i_vdd)

    vc1[i+1] = vc1[i] + dvc1
    vc2[i+1] = vc2[i] + dvc2
    vc3[i+1] = vc3[i] + dvc3
    vc4[i+1] = vc4[i] + dvc4


fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(111)
ax.plot(t, vc1, label="vc1")
ax.plot(t, vc2, label="vc2")
ax.plot(t, vc3, label="vc3")
ax.plot(t, vc4, label="vc4")
plt.legend()
plt.show()