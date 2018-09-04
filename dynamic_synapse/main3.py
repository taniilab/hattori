import numpy as np
import matplotlib.pyplot as plt

# Dirac Delta function
def Delta(x):
    if 0 <= x < dt:
        return 1/dt
    else:
        return 0

def modified_delta(x):
    if 0 <= x < tau_rise:
        return -np.log(-(np.exp(tau_rise)/tau_rise)*(x-tau_rise)) + tau_rise
    else:
        return 0
# simulation time
T = 5000
dt = 0.05
t = np.arange(0, T, dt)

# firing time
t_ap = 500

# Recovered
R1 = 0 * np.ones(int(T / dt))
R2 = 0 * np.ones(int(T / dt))
# Effective
E1 = 0 * np.ones(int(T / dt))
E2 = 0 * np.ones(int(T / dt))
# Inactive
I1 = np.ones(int(T / dt))  # I = 1 - R - E
I2 = np.ones(int(T / dt))  # I = 1 - R - E

# probability of release of transmitter
U_SE = 1

tau_rise = 2
dR1 = 0
dR2 = 0
dE1 = 0
dE2 = 0

# time constant(inactive)
tau_rec = 100
# time constant(recovered)
tau_inact = 2


for i in range(0, len(t)-1):

    if t[i] == 130:
        t_ap = t[i]

    if t[i] == 132:
        t_ap = t[i]

    if t[i] == 134:
        t_ap = t[i]

    if t[i] == 136:
        t_ap = t[i]

    if t[i] == 138:
        t_ap = t[i]

    if t[i] == 140:
        t_ap = t[i]

    if t[i] == 142:
        t_ap = t[i]

    if t[i] == 144:
        t_ap = t[i]

    if t[i] == 146:
        t_ap = t[i]

    if t[i] == 900:
        t_ap = t[i]

    if t[i] == 2000:
        t_ap = t[i]

    if t[i] == 3000:
        t_ap = t[i]

    if t[i] == 3010:
        t_ap = t[i]

    if t[i] == 3020:
        t_ap = t[i]

    if t[i] == 3030:
        t_ap = t[i]

    if t[i] == 3040:
        t_ap = t[i]

    if t[i] == 3050:
        t_ap = t[i]

    if t[i] == 3060:
        t_ap = t[i]

    if t[i] == 3070:
        t_ap = t[i]

    if t[i] == 3080:
        t_ap = t[i]

    if t[i] == 3090:
        t_ap = t[i]

    if t[i] == 3200:
        t_ap = t[i]

    if t[i] == 3220:
        t_ap = t[i]

    dR1 = dt * ((I1[i] / tau_rec) - U_SE * R1[i] * modified_delta(t[i] - t_ap))
    dE1 = dt * ((- E1[i] / tau_inact) + U_SE * R1[i] * modified_delta(t[i] - t_ap))
    R1[i + 1] = R1[i] + dR1
    E1[i + 1] = E1[i] + dE1
    I1[i + 1] = 1 - R1[i + 1] - E1[i + 1]

    dR2 = dt * ((I2[i] / tau_rec) - U_SE * R2[i] * Delta(t[i] - t_ap))
    dE2 = dt * ((- E2[i] / tau_inact) + U_SE * R2[i] * Delta(t[i] - t_ap))
    R2[i + 1] = R2[i] + dR2
    E2[i + 1] = E2[i] + dE2
    I2[i + 1] = 1 - R2[i + 1] - E2[i + 1]


fig = plt.figure(figsize=(21, 14))
ax1 = fig.add_subplot(3, 2, 1)
ax1.plot(t, E1)
plt.ylim([-0.4, 1.2])
ax3 = fig.add_subplot(3, 2, 3)
ax3.plot(t, R1)
plt.ylim([-0.4, 1.2])
ax5 = fig.add_subplot(3, 2, 5)
ax5.plot(t, I1)
plt.ylim([-0.4, 1.2])

ax2 = fig.add_subplot(3, 2, 2)
ax2.plot(t, E2)
plt.ylim([-0.4, 1.2])
ax4 = fig.add_subplot(3, 2, 4)
ax4.plot(t, R2)
plt.ylim([-0.4, 1.2])
ax6 = fig.add_subplot(3, 2, 6)
ax6.plot(t, I2)
plt.ylim([-0.4, 1.2])

plt.show()

