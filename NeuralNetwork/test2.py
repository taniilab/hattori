import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(123)

def true_function(x, noise=False):
    # 0.2+0.4x2+0.3sin(15x)+0.05cos(50x)
    y = 0.2  + 0.3 * np.sin(15 * x) + 0.05 * np.cos(50 * x)
    if noise:
      y += 0.01 * np.random.randn(*x.shape)
    return y

x_test = np.linspace(0, 1, 100)
y_test = true_function(x_test)

plt.plot(x_test, y_test)

num_target = 20

x_train = np.random.rand(num_target, 1)
y_train = true_function(x_train, noise=True)

plt.plot(x_test, y_test, color='blue', label='True function f(x)')
plt.plot(x_train, y_train, 'o', color='black', label='Observation (=Training data)')
plt.legend(loc='lower right')

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(100, activation='tanh', input_dim=1))
model.add(Dense(1))

model.summary()

plt.show()