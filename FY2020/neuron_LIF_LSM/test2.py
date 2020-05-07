import numpy as np


a = np.arange(5).reshape((1, 5)).T

b= np.arange(5).reshape((1, 5)).T

print(a)
print(b)
print("")
print(a.T @ b)
print(b.T @ a)