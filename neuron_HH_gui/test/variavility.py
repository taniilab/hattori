import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

dt = 0.01
t = np.arange(0, 1500, dt)
x = np.zeros(len(t))
y = np.zeros(len(t))
z = np.zeros(len(t))
a = 1
b = 2.92
c = 1
d = 5
r = 0.01
xr = -1.6
s = 4
I = np.zeros(len(t))
I[int(500/dt):int(1200/dt)] = 3.2
k1x = 0
k1y = 0
k1z = 0

t_ap = []


for i in range(0, len(t)-1):
    if x[i] > 0 and i > 200/dt:
        t_ap.append(dt*i)

    k1x = (y[i] - a * x[i] ** 3 + b * x[i] ** 2 - z[i] + I[i])
    k1y = (c - d * x[i] ** 2 - y[i])
    k1z = (r * (s * (x[i] - xr) - z[i]))

    x[i + 1] = x[i] + k1x * dt
    y[i + 1] = y[i] + k1y * dt
    z[i + 1] = z[i] + k1z * dt

fig = plt.figure(figsize=(30, 15))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(x, lags=len(t)-100, ax=ax1)
ax2 = fig.add_subplot(212)
ax2.plot(t, x)
print(t_ap[0])
print(t_ap[-1])
plt.show()
