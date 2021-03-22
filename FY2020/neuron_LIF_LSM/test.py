#メモリ節約テスト

import numpy as np
import pandas as pd

save_path = "Z:/simulation/test"

time = 100000
lump = 10
cycle = int(time/lump)
x = np.zeros((5, lump))
df = pd.DataFrame(columns=['number'])
print(df)
df.to_csv(save_path + "/test.csv", mode='a')

for j in range(cycle):
    for i in range(lump-1):
        x[:, i+1] = x[:, i]+1
    print(x)

    df = pd.DataFrame()
    df['number'] = x[0, :]
    df.to_csv(save_path + "/test.csv", mode='a', header=None)

    x = np.fliplr(x)


"""
read_path = "Z:/simulation/test/2020_4_27_21_6_46_Iext_amp0.0006_Pmax_AMPA0.0_Pmax_NMDA0_LIF.csv"

df = pd.read_csv(read_path, usecols=["V_0 [mV]"], skiprows=1)
train = df.values.T[0] + 70
t = np.arange(0, 2000, 0.04)
target = np.sin(t*0.05)


train = train.reshape((len(train), 1))
target = target.reshape((len(target), 1))

lsm = LSM()
lsm.train(train, target)
plt.figure()
plt.plot(t, train[:, 0], label="output")
plt.plot(t, target[:, 0], label="target")
print(train.shape)
print(lsm.output_w.shape)
print((train @ lsm.output_w).shape)
predict = (train @ lsm.output_w).T
plt.plot(t, predict[0], label="after training")
plt.legend()
plt.show()

"""

"""

len = 300
pitch = 0.1
steps = int(len/pitch)

T = np.arange(0, len, pitch)
V = 0.2 * np.random.randn(3, steps) + np.sin(T*0.30)
target = np.ones((1, steps)) * 5*np.sin(T*0.3)
lsm = LSM()
lsm.train(V.T[:int(steps/2)], target.T[:int(steps/2)], regularization=1, lamda=1)

print("V_T{}".format(V.T[:int(steps/2)].shape))
print("T{}".format(T.shape))
print("target_T{}".format(target.T[:int(steps/2)].shape))
print("weight{}".format(lsm.output_w.shape))

tmp = V.T @ lsm.output_w
print(tmp.T.shape)
print("norm_a{}".format(np.linalg.norm(tmp.T[0]-target[0], ord=2)))
print("norm_b{}".format(np.linalg.norm(V[0]-target[0], ord=2)))
plt.figure()
plt.plot(T, V[0], label="input")
plt.plot(T, target[0], label="target")
plt.plot(T, tmp.T[0], label="training")
plt.legend()
plt.show()
"""