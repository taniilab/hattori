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

fig = plt.figure(figsize=(10,15))
ax1 = fig.add_subplot(211)
ax1.plot(x[tau:], x[:len(x)-tau], lw=0.1)
ax2 = fig.add_subplot(212)
index = np.arange(0, len(x[tau*2:len(x)-tau]))
ax2.plot(x[tau:], lw=0.3)
plt.tight_layout()
plt.show()