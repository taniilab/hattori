# mV ms kΩ uF uA

import matplotlib.pyplot as plt
import numpy as np

# parameters
r1 = 400000
r2 = 400000
r3 = 200e-3
r4 = 200e-3
r5 = 50e-3
r6 = 50e-3
r8 = 50
r9 = 50
c1 = 1e-3
c2 = 1e-3
c3 = 100
c4 = 100
dt = 0.0001
vrest = -70
tau = 0.01
v_amp = 2500

t = np.arange(0, 50, dt)
vdd = t*0

for k in range(int(tau/dt)):
    vdd[int((5/dt) + k):] -= v_amp*dt/tau
for k in range(int(2*tau/dt)):
    vdd[int((25/dt) + k):] += v_amp*dt/tau
vdd[int(45/dt):] = 0

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

for i in range(len(t)-1):

    print(i)
    print("time: {}".format(t[i]))
    """
    print("vdd: {}".format(vdd[i]))
    print("i_vdd: {}".format(i_vdd))
    print("i_r4: {}".format(i_r4))
    print("dvc1: {}".format(dvc1))
    print("dvc2: {}".format(dvc2))
    print("dvc3: {}".format(dvc3))
    print("dvc4: {}\n".format(dvc4))
    """

    i_vdd = ((r3+r4)*(vdd[i]-vc3[i]-vc4[i])-r4*(vc1[i]+vc2[i])) / \
            (r3*r4+(r3+r4)*(r5+r6))
    i_r4 = (r3*(vdd[i]-vc3[i]-vc4[i])+(r5+r6)*(vc1[i]+vc2[i])) / \
           (r3*r4+(r3+r4)*(r5+r6))


    k1_vc1 = (dt/2)*(1/c1)*(-(vc1[i]+vrest)/r1 + i_vdd - i_r4)
    k1_vc2 = (dt/2)*(1/c2)*(-(vc2[i]-vrest)/r2 + i_vdd - i_r4)
    k1_vc3 = (dt/2)*(1/c3)*(-vc3[i]/r8 + i_vdd)
    k1_vc4 = (dt/2)*(1/c4)*(-vc4[i]/r9 + i_vdd)

    vc1_tmp = vc1[i] + k1_vc1
    vc2_tmp = vc2[i] + k1_vc2
    vc3_tmp = vc3[i] + k1_vc3
    vc4_tmp = vc4[i] + k1_vc4

    i_vdd_tmp = ((r3 + r4) * (vdd[i] - vc3_tmp - vc4_tmp) - r4 * (vc1_tmp + vc2_tmp)) / \
            (r3 * r4 + (r3 + r4) * (r5 + r6))
    i_r4_tmp = (r3 * (vdd[i] - vc3_tmp - vc4_tmp) + (r5 + r6) * (vc1_tmp + vc2_tmp)) / \
           (r3 * r4 + (r3 + r4) * (r5 + r6))

    k2_vc1 = (dt / 2) * (1 / c1) * (-(vc1_tmp + vrest) / r1 + i_vdd_tmp - i_r4_tmp)
    k2_vc2 = (dt / 2) * (1 / c2) * (-(vc2_tmp - vrest) / r2 + i_vdd_tmp - i_r4_tmp)
    k2_vc3 = (dt / 2) * (1 / c3) * (-vc3_tmp / r8 + i_vdd_tmp)
    k2_vc4 = (dt / 2) * (1 / c4) * (-vc4_tmp / r9 + i_vdd_tmp)

    vc1_tmp_2 = vc1_tmp + k2_vc1
    vc2_tmp_2 = vc2_tmp + k2_vc2
    vc3_tmp_2 = vc3_tmp + k2_vc3
    vc4_tmp_2 = vc4_tmp + k2_vc4

    vc1_tmp_2 = vc1[i] + k2_vc1
    vc2_tmp_2 = vc2[i] + k2_vc2
    vc3_tmp_2 = vc3[i] + k2_vc3
    vc4_tmp_2 = vc4[i] + k2_vc4

    i_vdd_tmp_2 = ((r3 + r4) * (vdd[i] - vc3_tmp_2 - vc4_tmp_2) - r4 * (vc1_tmp_2 + vc2_tmp_2)) / \
            (r3 * r4 + (r3 + r4) * (r5 + r6))
    i_r4_tmp_2 = (r3 * (vdd[i] - vc3_tmp_2 - vc4_tmp_2) + (r5 + r6) * (vc1_tmp_2 + vc2_tmp_2)) / \
           (r3 * r4 + (r3 + r4) * (r5 + r6))

    k3_vc1 = (dt / 2) * (1 / c1) * (-(vc1_tmp_2 + vrest) / r1 + i_vdd_tmp_2 - i_r4_tmp_2)
    k3_vc2 = (dt / 2) * (1 / c2) * (-(vc2_tmp_2 - vrest) / r2 + i_vdd_tmp_2 - i_r4_tmp_2)
    k3_vc3 = (dt / 2) * (1 / c3) * (-vc3_tmp_2 / r8 + i_vdd_tmp_2)
    k3_vc4 = (dt / 2) * (1 / c4) * (-vc4_tmp_2 / r9 + i_vdd_tmp_2)

    vc1_tmp_3 = vc1_tmp_2 + k3_vc1
    vc2_tmp_3 = vc2_tmp_2 + k3_vc2
    vc3_tmp_3 = vc3_tmp_2 + k3_vc3
    vc4_tmp_3 = vc4_tmp_2 + k3_vc4

    vc1_tmp_3 = vc1[i] + k3_vc1*2
    vc2_tmp_3 = vc2[i] + k3_vc2*2
    vc3_tmp_3 = vc3[i] + k3_vc3*2
    vc4_tmp_3 = vc4[i] + k3_vc4*2

    i_vdd_tmp_3 = ((r3 + r4) * (vdd[i] - vc3_tmp_3 - vc4_tmp_3) - r4 * (vc1_tmp_3 + vc2_tmp_3)) / \
                  (r3 * r4 + (r3 + r4) * (r5 + r6))
    i_r4_tmp_3 = (r3 * (vdd[i] - vc3_tmp_3 - vc4_tmp_3) + (r5 + r6) * (vc1_tmp_3 + vc2_tmp_3)) / \
                 (r3 * r4 + (r3 + r4) * (r5 + r6))

    k4_vc1 = dt * (1 / c1) * (-(vc1_tmp_3 + vrest) / r1 + i_vdd_tmp_3 - i_r4_tmp_3)
    k4_vc2 = dt * (1 / c2) * (-(vc2_tmp_3 - vrest) / r2 + i_vdd_tmp_3 - i_r4_tmp_3)
    k4_vc3 = dt * (1 / c3) * (-vc3_tmp_3 / r8 + i_vdd_tmp_3)
    k4_vc4 = dt * (1 / c4) * (-vc4_tmp_3 / r9 + i_vdd_tmp_3)

    vc1[i+1] = vc1[i] + (1/6)*(k1_vc1 + 2*k2_vc1 + 2*k3_vc1 + k4_vc1)
    vc2[i+1] = vc2[i] + (1/6)*(k1_vc2 + 2*k2_vc2 + 2*k3_vc2 + k4_vc2)
    vc3[i+1] = vc3[i] + (1/6)*(k1_vc3 + 2*k2_vc3 + 2*k3_vc3 + k4_vc3)
    vc4[i+1] = vc4[i] + (1/6)*(k1_vc4 + 2*k2_vc4 + 2*k3_vc4 + k4_vc4)


fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(111)
ax.plot(t, vdd, label="vdd")
ax.plot(t, vc1, label="vc1")
ax.plot(t, -vc2, label="vc2")
ax.plot(t, vc3, label="vc3")
ax.plot(t, -vc4, label="vc4")
plt.legend()
plt.show()