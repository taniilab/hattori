import pandas as pd
from scipy import arange, hamming, sin, pi
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, ifft
import seaborn as sns


index = "15"
path = "C:/Box Sync/Personal/Documents/touhoku_patch/20180420_cortex/"
path_h = path + "voltage/voltage" + index + ".csv"
path_i = path + "current/current" + index + ".csv"

fsize = 24
sample = 20000
fig = plt.figure(figsize=(30, 15))

dfv = pd.read_csv(path_h, delimiter=',')
dfc = pd.read_csv(path_i, delimiter=',')

ax2 = fig.add_subplot(2, 1, 1)
ax2.plot(dfc['index']/sample, dfc['current(pA)'], markevery=[0, -1], color="purple")
ax2.tick_params(labelsize=fsize)
ax2.set_xlabel("time[s]", fontsize=fsize)
ax2.set_ylabel("clamp current[pA]", fontsize=fsize)

ax1 = ax2.twinx()
ax1.plot(dfv['index']/sample, dfv['voltage(mV)'], markevery=[0, -1])
ax1.tick_params(labelsize=fsize)
ax1.set_xlabel("time[s]", fontsize=fsize)
ax1.set_ylabel("membrane potential[mV]", fontsize=fsize)


ax3 = fig.add_subplot(2, 1, 2)
f = np.array(dfc['current(pA)'])
noise = f[100000:130000]
N = len(noise)  # サンプル数
dt = 1/20000  # サンプリング間隔
t = np.arange(0, N * dt, dt)  # 時間軸
freq = np.linspace(0, 1.0 / dt, N)  # 周波数軸

# 信号を生成（周波数10の正弦波+周波数20の正弦波+ランダムノイズ）
sig = (noise+60.67) * np.hanning(N)

print(noise)
# 高速フーリエ変換
F = np.fft.fft(sig)
amp = np.abs(F)
#ax3.plot(sig)
ax3.plot(freq, amp)


fig.tight_layout()
plt.show()
