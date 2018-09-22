import numpy as np
import matplotlib.pyplot as plt

dt = 0.1
t = np.arange(0, 100000, dt)
y = 0 * t
dy = 0

for i in range(0, 999999):
    dy = dt * (-y[i] + np.abs(np.random.normal(0, 1)))
    y[i+1] = y[i] + dy

plt.plot(t, y)
plt.show()