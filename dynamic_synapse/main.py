import numpy as np
import matplotlib.pyplot as plt

# Dirac Delta function
def Delta(x):
    if 0 < x < dt:
        return 1/dt
    else:
        return 0
# simulation time
T = 5000
dt = 0.05
t = np.arange(0, T, dt)

# firing time
t_ap = 500

# Recovered
R = 0 * np.ones(int(T / dt))
# Effective
E = 0 * np.ones(int(T / dt))
# Inactive
I = np.ones(int(T / dt))  # I = 1 - R - E

# probability of release of transmitter
U_SE = 0.8

dR = 0
dE = 0

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

    dR = dt * ((I[i] / tau_rec) - U_SE * R[i] * Delta(t[i] - t_ap))
    dE = dt * ((- E[i] / tau_inact) + U_SE * R[i] * Delta(t[i] - t_ap))
    R[i + 1] = R[i] + dR
    E[i + 1] = E[i] + dE
    I[i + 1] = 1 - R[i + 1] - E[i + 1]


fig = plt.figure(figsize=(21, 14))
ax = fig.add_subplot(1, 1, 1)
ax.plot(t, E)
plt.show()

