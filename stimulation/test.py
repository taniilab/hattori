import numpy as np

A = np.random.randn(2,3)
print(A)
A = np.delete(A, -1, 1)
print(A)
A = np.insert(A, 2, [3, 3], axis=1)
print(A)