import numpy as np
import matplotlib.pyplot as plt


t = np.arange(0, 1000, 0.02)
t_fire = [-10000, 500, 501, 502, 503, 504, 800, 810]
isi_co = 0

"""
isi = []
for i in range(len(t)-1):
    isi.append()
"""

Use = 0.5
D = 1.1
F =0.05

A = t*0
u = t*0 + Use
R = t*0 + 1

for i in range(len(t)-1):
    if t[i]==500 or t[i]==501 or t[i]==502 or t[i]==503 or t[i]==504 or t[i]==800 or t[i]==810:
        delta = t_fire[isi_co+1]-t_fire[isi_co]
        print(delta)
        u[i+1] = Use + u[i]*(1-Use)*np.exp(-delta/F)
        R[i+1] = 1 + (R[i]-u[i]*R[i]-1)*np.exp(-delta/D)
        isi_co+=1
    else:
        u[i+1] = u[i]
        R[i+1] = R[i]
    A[i+1] = u[i]*R[i]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(t, A, label="A")
ax.plot(t, u, label="u")
ax.plot(t, R, label="R")
plt.legend()
plt.show()