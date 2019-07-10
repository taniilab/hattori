import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

V = np.arange(-150, 40, 0.1)

m_inf = 1/(1 + np.exp(-(V+2+57)/6.2))
h_inf = 1/(1 + np.exp((V+2+81)/4))
tau_u = ((30.8 + (211.4 + np.exp((V + 2 + 113.2)/5)))/
         (3.7*(1 + np.exp((V + 2 + 84)/3.2))))

fsize=32
plt.figure()

plt.plot(V, h_inf, label="u_inf")
plt.plot(V, m_inf, label="s_inf")
plt.xlabel("membrane potential [mV]", fontsize = fsize)
plt.ylabel("probability [a.u.]", fontsize = fsize)
plt.legend(fontsize=fsize)
plt.tick_params(labelsize=fsize)
plt.figure()
plt.plot(V, tau_u)
plt.xlabel("time constant [ms]", fontsize = fsize)
plt.ylabel("probability [a.u.]", fontsize = fsize)
plt.tick_params(labelsize=fsize)