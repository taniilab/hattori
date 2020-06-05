import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

dt = 0.1
t_length = 200
t = np.arange(0, t_length, dt)
x = t*0+0.5


beta = 2
gamma = 1
tau = int(2/dt)
n = 9.65

for i in range(tau, len(t)-1):
    x[i+1] = x[i] + dt*(beta*x[i-tau]/(1+x[i-tau]**n) - gamma*x[i])

fig = plt.figure(figsize=(20,20))
ax1 = fig.add_subplot(221)
ax1.plot(x[tau:], x[:len(x)-tau], lw=0.1)
ax2 = fig.add_subplot(222)
index = np.arange(0, len(x[tau*2:len(x)-tau]))
ax2.plot(t[tau:], x[tau:], lw=0.3)


"""
n = 7
for i in range(tau, len(t)-1):
    x[i+1] = x[i] + dt*(beta*x[i-tau]/(1+x[i-tau]**n) - gamma*x[i])
"""
#補間
t2 = np.arange(0, t_length*2, dt)
x_expand = t2*0+0.5
print(len(x))
print(len(x_expand))

for i in range(len(x)):
    x_expand[2*i] = x[i]

for j in range(len(x)-2):
    x_expand[2*j+1] = (x_expand[2*j+2] + x_expand[2*j])/2

ax3 = fig.add_subplot(223)
ax3.plot(x[tau:], x[:len(x)-tau], lw=0.1)
ax4 = fig.add_subplot(224)
#index = np.arange(0, len(x[tau*2:len(x)-tau]))
ax4.plot(t[tau:], x[tau:], lw=0.4)
ax4.plot(t[tau:], x_expand[tau:tau+len(t[tau:])], lw=0.4)

print(x_expand)

plt.tight_layout()
plt.show()
