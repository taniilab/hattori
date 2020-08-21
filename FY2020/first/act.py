import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot

df = pd.read_csv("C:/Users/Kouhei/Downloads/voltage2.csv")
df2 = pd.read_csv("C:/Users/Kouhei/Downloads/voltage5.csv")

print(df)
df = df.values
print(len(df))
mem = df[::100]
print(len(mem))
print(df2)
df2 = df2.values
print(len(df2))
mem2 = df2[::100]
print(len(mem))

autocorrelation_plot(pd.Series(mem[:, 2]))
autocorrelation_plot(pd.Series(mem2[:, 2]))

fig = plt.figure(figsize=(15, 10))
ax2 = fig.add_subplot(211)
ax2.plot(mem[:, 1], mem[:, 2])

ax2 = fig.add_subplot(212)
ax2.plot(mem2[:, 1], mem2[:, 2])
plt.show()
