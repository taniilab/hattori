import pandas as pd
from scipy import arange, hamming, sin, pi
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, ifft
import seaborn as sns


path = "F:/simulation/HH/tmp/2018_5_18_10_32_49__Iext_amp_ 0.5_ Syncp_ 5_ Pmax_ 2.0_ gT_ 0.0_ ratio_ 0.4__N0_HH.csv"

fsize = 24
sample = 20000
fig = plt.figure(figsize=(30, 15))

df = pd.read_csv(path, delimiter=',')

ax2 = fig.add_subplot(2, 1, 1)
ax2.plot(df['T [ms]']/sample, df['V [mV]'], markevery=[0, -1], color="purple")
ax2.tick_params(labelsize=fsize)
ax2.set_xlabel("time[s]", fontsize=fsize)
ax2.set_ylabel("membrane potential[mV]", fontsize=fsize)

ax3 = fig.add_subplot(2, 1, 2)
plt.grid(which="both")
noise = np.array(df['V [mV]'])
noise[np.isnan(noise)] = 0

N = len(noise)  # サンプル数
dt = 1/200  # サンプリング間隔
t = np.arange(0, N * dt, dt)  # 時間軸
freq = np.linspace(0, 1.0 / dt, N)  # 周波数軸

# 信号を生成（周波数10の正弦波+周波数20の正弦波+ランダムノイズ）
sig = noise * np.hanning(N)

print(noise)
# 高速フーリエ変換
F = np.fft.fft(sig)
amp = np.abs(F)
#plt.xscale("symlog")
ax3.plot(freq, amp)

fig.tight_layout()
plt.show()
