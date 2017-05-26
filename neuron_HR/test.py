import numpy as np

def test_f(x):
    return x**2

x = np.array([(1,2), (3,4)])
y = np.array([(5,6), (7,8)])
y[0,:] = x[0,:]
z = test_f(x)
w = x[:, 0]
v = w[1]
print(x)
print(w)
print(v)