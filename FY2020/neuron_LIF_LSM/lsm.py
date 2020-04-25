import numpy as np

class LSM():
    def __init__(self):
        pass

    def train(self, train_data, target_data, lamda=1, regularization=0):
        if regularization == 0:
            pass
        else:
            self.output_w = (np.linalg.inv(train_data.T @ train_data + \
                                           lamda * np.identity(np.size(train_data, 1))) @ train_data.T) @ target_data

    def pridict(self):
        pass

a = np.random.randn(2,3)
print(a)
print(np.size(a, 1))
print(np.identity(np.size(a, 1)))