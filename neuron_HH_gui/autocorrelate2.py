import numpy as np
import pandas as pd
from scipy import stats

# グラフ描画
from matplotlib import pylab as plt
import seaborn as sns

# グラフを横長にする
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6

# 統計モデル
import statsmodels.api as sm


t = np.arange(0, 100, 0.1)
y = np.sin(t)+np.sin(t)

"""
path = "//192.168.13.10/Public/hattori/simulation/HH/rawdata/" + \
       "2018_10_8_19_12_24_N0_P_AMPA0.0_P_NMDA0.2_Mg_conc0.4_delay0HH.csv"
"""
path = "//192.168.13.10/Public/hattori/simulation/HH/rawdata/" + \
       "2018_10_2_23_44_38__T_ 70000_ Iext_amp_ 10_ Mg_ 1.0_ noise_ 2_ syncp_ 2_ U_SE_AMPA_ 0.7_ A_SE_AMPA_ 0.2_ A_SE_NMDA_ 0.4__N0_HH.csv"

df = pd.read_csv(path, delimiter=',', skiprows=1)
df.fillna(0)


fig = plt.figure(figsize=(30, 15))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df['V [mV]'], lags=100000, ax=ax1)
ax2 = fig.add_subplot(212)
ax2.plot(df['T [ms]'], df['V [mV]'])
plt.show()

