import numpy as np


A = np.arange(0, 5)

print(A)

A=np.roll(A, -1)
A[-1] = 5

print(A)


B = np.ones(10)
B[-1] = 5

print(B)
B_dummy = B[B != B[0]]
if B_dummy == B[-1]:
    print("kanopero")

print(B[B != B[0]])
print(len(B[B != B[0]]))

print(B[0:len(B)-1])