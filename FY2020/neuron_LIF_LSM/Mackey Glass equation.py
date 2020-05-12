import numpy as np
import matplotlib.pyplot as plt


dt = 0.01
t = np.arange(0, 5000, dt)
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

n = 7
for i in range(tau, len(t)-1):
    x[i+1] = x[i] + dt*(beta*x[i-tau]/(1+x[i-tau]**n) - gamma*x[i])

ax3 = fig.add_subplot(223)
ax3.plot(x[tau:], x[:len(x)-tau], lw=0.1)
ax4 = fig.add_subplot(224)
index = np.arange(0, len(x[tau*2:len(x)-tau]))
ax4.plot(t[tau:], x[tau:], lw=0.3)

plt.tight_layout()
plt.show()
