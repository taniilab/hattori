import numpy as np
import matplotlib.pyplot as plt

t = np.arange(0,10000)
x= np.zeros(len(t))
noise = np.random.randn(len(t))
a = 1
b = 1
c = 1

for i in range(len(t)-1):
    x[i+1] = a*noise[i+1] + b*x[i] + c*noise[i]

fig = plt.figure(figsize=(15,8))
ax = fig.add_subplot(1,1,1)
ax.plot(t, x)
plt.show()