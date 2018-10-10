import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd

dt = 0.1
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

path = "//192.168.13.10/Public/hattori/simulation/HH/" + \
       "2018_10_9_20_13_6_N0_P_AMPA0.1_P_NMDA0.1_Mg_conc0.1_delay0HH.csv"
df = pd.read_csv(path, delimiter=',', skiprows=1)
df.fillna(0)

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

t_ap2 = []
voltage = df['V [mV]']
time = df['T [ms]']
dt2 = 0.04

for i in range(0, len(df['T [ms]'])):
    if voltage[i] > 10:
        t_ap2.append(time[i])

print(t_ap2[0])
print(t_ap2[-1])
