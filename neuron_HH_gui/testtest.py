import numpy as np
import matplotlib.pyplot as plt

dt = 0.001
t = np.arange(0,1000, dt)
v = 0*t
v[0] = 0.1
I = 0*t - 5
I[int(200/dt):int(300/dt)] += 6

for i in range(len(t)-1):
    v[i+1] = v[i] + dt*(v[i]**2)
    if v[i+1] > 5:
        v[i+1] = -2
        print(v[i])
fig = plt.figure(figsize=(12, 12))
plt.plot(t, v)
plt.show()
