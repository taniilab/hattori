import numpy as np
import matplotlib.pyplot as plt

dt = 0.02
t = np.arange(0, 1000, dt)
t_fire = [-10000, 500, 501, 502, 503, 504, 800, 810]
isi_co = 0


Use = 0.5
tau_rec = 200
tau_inact = 2
R = t*0 + 1
E = t*0
I = 1 - R - E

def delta_func(i):
    time = i*dt
    print(time)
    if time == 500 or time == 501 or time == 502 or time == 503 or time == 504 or time == 800 or time == 810:
        return 1
        print(time)
    else:
        return 0

for i in range(len(t)-1):
    R[i+1] =  R[i] + dt*I[i]/tau_rec - Use*R[i]*delta_func(i)
    E[i+1] =  E[i] - dt*E[i]/tau_inact + Use*R[i]*delta_func(i)
    I[i+1] = 1 - R[i] - E[i]
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(t, R, label="R")
ax.plot(t, E, label="E")
ax.plot(t, I, label="I")
plt.legend()
plt.show()