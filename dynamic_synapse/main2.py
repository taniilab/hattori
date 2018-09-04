import numpy as np
import matplotlib.pyplot as plt

# Dirac Delta function
def Delta(x):
    if 0 <= x < dt:
        return 1/dt
    else:
        return 0
# simulation time
T = 5000
dt = 0.05
t = np.arange(0, T, dt)

# firing time
t_ap = 500

I = np.zeros(int(T / dt))  # I = 1 - R - E
gsyn = 1
s = np.ones(int(T / dt))
x = np.ones(int(T / dt))
dx = 0
ds = 0

alpha_s = 1
tau_s =2
tau_x =0.05
alpha_x = 15


for i in range(0, len(t)-1):
    if t[i] == 1000:
        t_ap = t[i]
    if t[i] == 2000:
        t_ap = t[i]
    if t[i] == 4800:
        t_ap = t[i]

    ds = dt * (alpha_s * x[i] * (1-s[i]) - s[i]/tau_s)
    dx = dt * (alpha_x * Delta(t[i] - t_ap) - x[i]/tau_x)
    s[i+1] = s[i] + ds
    x[i+1] = x[i] + dx
    print(Delta(t[i] - t_ap))

# Recovered
R = 0 * np.ones(int(T / dt))
# Effective
E = 0 * np.ones(int(T / dt))
# Inactive
I = np.ones(int(T / dt))  # I = 1 - R - E


fig = plt.figure(figsize=(21, 14))
ax = fig.add_subplot(1, 1, 1)
ax.plot(t, s)
plt.show()

