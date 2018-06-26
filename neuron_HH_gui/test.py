import numpy as np
import matplotlib.pyplot as plt

# サンプルデータの定義 ========================================================
N = 250         # Signal length
fs = 1000       # Sampling rate
fn = fs * 0.5   # Nyquist frequency

t = np.arange(N) / fs
# 400 Hz
sine_400 = np.sin(2. * np.pi * 400 * t)
# 800 Hz
sine_800 = 2 * np.sin(2. * np.pi * 800 * t)
# 1280 Hz
sine_1280 = 2 * np.sin(2. * np.pi * 1280 * t)

y = sine_400 + sine_800 + sine_1280

fig = plt.figure(figsize=(21, 14))
Rr = np.correlate(y, y, mode='full')

ax1 = fig.add_subplot(1, 1, 1)
ax1.plot(Rr)

plt.show()
