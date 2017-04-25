import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(nrows = 2, figsize = (12, 15))
dt = 0.0001
t = np.arange(0, 50, dt)

a = 1
b = 3
c = 1
d = 5
r = 0.001
s = 4

dx = 0
dy = 0
dz = 0
i = -10
xr = -1.5

x = [-0.5] * np.size(t)
y = [-0.5] * np.size(t)
z = [-5] * np.size(t)


for i in range(0, np.size(t)-1):
    dx = y[i] - a * x[i]*x[i]*x[i] + b * x[i]*x[i]
    dy = c - d * x[i] * x[i] - y[i]
    dz = r*(s*(x[i] - xr) - z[i])
    
    x[i+1] = x[i] + dt * dx
    y[i+1] = y[i] + dt * dy
    z[i+1] = z[i] + dt * dz

plt.grid()
lines, = ax[0].plot(t, x)
#lines, = ax[1].plot(t, y)
#lines, = ax[2].plot(t, z)

x = np.arange(-3,3, 0.0001)
xnull = [0] * np.size(x)
ynull = [0] * np.size(x)
for i in range(0, np.size(x)):
    xnull[i] =  a*x[i]**3 - b*x[i]**2
    ynull[i] =  c - d * x[i]**2

lines, = ax[1].plot(x, xnull)
lines, = ax[1].plot(x, ynull)

v1 = np.arange(-2, 2, 0.1)
u1 = np.arange(-50, 20, 0.25)
V, U = np.meshgrid(v1, u1)

print(V)
print(U)
print()
print(b*V**2)
print(a*V**3)
print(U - a*V**3 + b*V**2)
dV = U - a*V**3 + b*V**2 
print(dV)
dU = c - d*V**2 - U

N = np.sqrt(dV**2 + dU**2)*10
dV = dV/N
dU = dU/N

ax[1].quiver(V, U, dV, dU, units = 'xy', width = 0.025)
ax[1].set_ylim(-10, 2.5)
ax[1].set_xlim(-3, 3)