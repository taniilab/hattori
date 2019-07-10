import numpy as np

class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        out = x + y
        return out

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy

class MulLayer:
    def  __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        return x * y

    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx

class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1/(1 + np.exp(-x))
        return out

    def backwar(self, dout):
        dx = dout * (1 - self.out) * self.out
        return dx

def main():
    kawaii = 100
    kawaii_num = 5
    kawaii_tax = 1.1

    #initialize
    add = AddLayer()
    mul_1 = MulLayer()
    mul_2 = MulLayer()

    #forward
    kawaii_price = mul_1.forward(kawaii, kawaii_num)
    kawaii_total = mul_2.forward(kawaii_price, kawaii_tax)
    print(kawaii_total)

    #backward
    dprice = 1
    dkawaii_price, dkawaii_tax = mul_2.backward(dprice)
    dkawaii, dkawaii_num = mul_1.backward(dkawaii_price)
    print(dkawaii, dkawaii_num, dkawaii_tax, dkawaii_price)

if __name__ == '__main__':
     main()