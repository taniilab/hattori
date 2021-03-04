import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

path = "G:/Box/Personal/Paper/second/electrochemistry/電流計測/抽出_シミュレーション結果含/sim/"

filename0 = "re1vpp電圧刺激等価回路_論文用_20210204_IV特性.csv"
filename1 = "re2vpp電圧刺激等価回路_論文用_20210204_IV特性.csv"
filename2 = "re3vpp電圧刺激等価回路_論文用_20210204_IV特性.csv"
filename3 = "re4vpp電圧刺激等価回路_論文用_20210204_IV特性.csv"
filename4 = "re5vpp電圧刺激等価回路_論文用_20210204_IV特性.csv"
filename5 = "re6vpp電圧刺激等価回路_論文用_20210204_IV特性.csv"
filename6 = "re7vpp電圧刺激等価回路_論文用_20210204_IV特性.csv"
filename7 = "re8vpp電圧刺激等価回路_論文用_20210204_IV特性.csv"
filename8 = "re9vpp電圧刺激等価回路_論文用_20210204_IV特性.csv"
filename9 = "re10vpp電圧刺激等価回路_論文用_20210204_IV特性.csv"


df0 = pd.read_csv(path+filename0, skiprows=1, names=["time", "V(n001)", "V(n002)"])
df1 = pd.read_csv(path+filename1, skiprows=1, names=["time", "V(n001)", "V(n002)"])
df2 = pd.read_csv(path+filename2, skiprows=1, names=["time", "V(n001)", "V(n002)"])
df3 = pd.read_csv(path+filename3, skiprows=1, names=["time", "V(n001)", "V(n002)"])
df4 = pd.read_csv(path+filename4, skiprows=1, names=["time", "V(n001)", "V(n002)"])
df5 = pd.read_csv(path+filename5, skiprows=1, names=["time", "V(n001)", "V(n002)"])
df6 = pd.read_csv(path+filename6, skiprows=1, names=["time", "V(n001)", "V(n002)"])
df7 = pd.read_csv(path+filename7, skiprows=1, names=["time", "V(n001)", "V(n002)"])
df8 = pd.read_csv(path+filename8, skiprows=1, names=["time", "V(n001)", "V(n002)"])
df9 = pd.read_csv(path+filename9, skiprows=1, names=["time", "V(n001)", "V(n002)"])

print(df5)

plt.figure(figsize=(20, 15))
plt.plot(df0["time"], df0["V(n001)"])
plt.plot(df0["time"], df0["V(n002)"])
plt.plot(df1["time"], df1["V(n001)"])
plt.plot(df1["time"], df1["V(n002)"])
plt.plot(df2["time"], df2["V(n001)"])
plt.plot(df2["time"], df2["V(n002)"])
plt.plot(df3["time"], df3["V(n001)"])
plt.plot(df3["time"], df3["V(n002)"])
plt.plot(df4["time"], df4["V(n001)"])
plt.plot(df4["time"], df4["V(n002)"])
plt.plot(df5["time"], df5["V(n001)"])
plt.plot(df5["time"], df5["V(n002)"])
plt.plot(df6["time"], df6["V(n001)"])
plt.plot(df6["time"], df6["V(n002)"])
plt.plot(df7["time"], df7["V(n001)"])
plt.plot(df7["time"], df7["V(n002)"])
plt.plot(df8["time"], df8["V(n001)"])
plt.plot(df8["time"], df8["V(n002)"])
plt.plot(df9["time"], df9["V(n001)"])
plt.plot(df9["time"], df9["V(n002)"])

plt.show()

plt.figure()
print()