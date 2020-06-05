import matplotlib.pyplot as plt
import numpy as np


t = np.arange(0, 10, 0.1)
y = 30*np.exp(-0.5*t)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(t, y)
plt.show()