import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


path = 'C:/0_current.csv'
df = pd.read_csv(path)
plt.rcParams["font.size"] = 22
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.plot(df['index']/20000, df['V[mV]'], markevery=[0, -1])
