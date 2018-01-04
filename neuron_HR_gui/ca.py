import numpy as np
import matplotlib.pyplot as plt

V = np.arange(-150, 40, 0.1)
m_inf = 1/(1 + np.exp(-(V+57)/6.2))
h_inf = 1/(1 + np.exp(-(V+81)/4))

t_m = 0.612 + 1/(np.exp(-(V+132)/16.7)+np.exp((V+16.8)/18.2))

t_hh = 28 + np.exp(-(V+22)/10.5)
for i in range(len(V)):
    if V[i] <-81:
        t_hh[i] = 0    

t_hl = np.exp((V+467)/66.6)
for i in range(len(V)):
    if V[i] >-81:
        t_hl[i] = 0    


plt.figure()
plt.plot(V, m_inf)
plt.plot(V, h_inf)

plt.figure()
plt.plot(V, t_m)
plt.plot(V, t_hh)
plt.plot(V, t_hl)