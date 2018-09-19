@@ -1,53 +0,0 @@
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


Vhr = np.arange(-5, 3, 0.01)
m_infhr = 1/(1 + np.exp(-(Vhr+1.5)/0.25))
h_infhr = 1/(1 + np.exp(-(Vhr+2.3)/0.17))

t_mhr = (0.612 + 1/(np.exp(-(Vhr+4)/0.5)+np.exp((Vhr+0.42)/0.5)))*10

t_hhhr = (28 + np.exp(-(Vhr+0.6)/0.26))*10
for i in range(len(Vhr)):
    if Vhr[i] <-2.1:
        t_hhhr[i] = 0    

t_hlhr = (np.exp((Vhr+20)/3))*10
for i in range(len(Vhr)):
    if Vhr[i] >=-2.1:
        t_hlhr[i] = 0    

plt.figure()
plt.plot(Vhr, m_infhr)
plt.plot(Vhr, h_infhr)

plt.figure()
plt.plot(Vhr, t_mhr)
plt.plot(Vhr, t_hhhr)
plt.plot(Vhr, t_hlhr)