import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

path = "G:/Box/Personal/Paper/second/electrochemistry/電流計測/抽出_シミュレーション結果含/"
filename = "all.csv"
df = pd.read_csv(path+filename)
print(df)

fig = plt.figure(figsize=(20, 15))
ax = fig.add_subplot(111)

estimlw = 0
vlw =6
splw = 5
fontsize = 72

ax.plot(df["time"], df["Estim_1vpp"], linestyle="dotted", linewidth=estimlw, color="cyan")
ax.plot(df["time"], df["V_1vpp"]*0.1, linestyle="solid", linewidth=vlw, color="cyan")
ax.plot(df["time"], df["Estim_2vpp"], linestyle="dotted", linewidth=estimlw, color="olive")
ax.plot(df["time"], df["V_2vpp"]*0.1, linestyle="solid", linewidth=vlw, color="olive")
ax.plot(df["time"], df["Estim_3vpp"], linestyle="dotted", linewidth=estimlw, color="gray")
ax.plot(df["time"], df["V_3vpp"]*0.1, linestyle="solid", linewidth=vlw, color="gray")
ax.plot(df["time"], df["Estim_4vpp"], linestyle="dotted", linewidth=estimlw, color="pink")
ax.plot(df["time"], df["V_4vpp"]*0.1, linestyle="solid", linewidth=vlw, color="pink")
ax.plot(df["time"], df["Estim_5vpp"], linestyle="dotted", linewidth=estimlw, color="brown")
ax.plot(df["time"], df["V_5vpp"]*0.1, linestyle="solid", linewidth=vlw, color="brown")
ax.plot(df["time"], df["Estim_6vpp"], linestyle="dotted", linewidth=estimlw, color="purple")
ax.plot(df["time"], df["V_6vpp"]*0.1, linestyle="solid", linewidth=vlw, color="purple")
ax.plot(df["time"], df["Estim_7vpp"], linestyle="dotted", linewidth=estimlw, color="red")
ax.plot(df["time"], df["V_7vpp"]*0.1, linestyle="solid", linewidth=vlw, color="red")
ax.plot(df["time"], df["Estim_8vpp"], linestyle="dotted", linewidth=estimlw, color="green")
ax.plot(df["time"], df["V_8vpp"]*0.1, linestyle="solid", linewidth=vlw, color="green")
ax.plot(df["time"], df["Estim_9vpp"], linestyle="dotted", linewidth=estimlw, color="orange")
ax.plot(df["time"], df["V_9vpp"]*0.1, linestyle="solid", linewidth=vlw, color="orange")
ax.plot(df["time"], df["Estim_10vpp"], linestyle="dotted", linewidth=estimlw, color="blue")
ax.plot(df["time"], df["V_10vpp"]*0.1, linestyle="solid", linewidth=vlw, color="blue")

"""
ax.plot(df["time"], df["Estim_1vpp"], linestyle="dotted", linewidth=estimlw, color="blue")
ax.plot(df["time"], df["V_1vpp"]*0.1, linestyle="solid", linewidth=vlw, color="blue")
ax.plot(df["time"], df["Estim_2vpp"], linestyle="dotted", linewidth=estimlw, color="orange")
ax.plot(df["time"], df["V_2vpp"]*0.1, linestyle="solid", linewidth=vlw, color="orange")
ax.plot(df["time"], df["Estim_3vpp"], linestyle="dotted", linewidth=estimlw, color="green")
ax.plot(df["time"], df["V_3vpp"]*0.1, linestyle="solid", linewidth=vlw, color="green")
ax.plot(df["time"], df["Estim_4vpp"], linestyle="dotted", linewidth=estimlw, color="red")
ax.plot(df["time"], df["V_4vpp"]*0.1, linestyle="solid", linewidth=vlw, color="red")
ax.plot(df["time"], df["Estim_5vpp"], linestyle="dotted", linewidth=estimlw, color="purple")
ax.plot(df["time"], df["V_5vpp"]*0.1, linestyle="solid", linewidth=vlw, color="purple")
ax.plot(df["time"], df["Estim_6vpp"], linestyle="dotted", linewidth=estimlw, color="brown")
ax.plot(df["time"], df["V_6vpp"]*0.1, linestyle="solid", linewidth=vlw, color="brown")
ax.plot(df["time"], df["Estim_7vpp"], linestyle="dotted", linewidth=estimlw, color="pink")
ax.plot(df["time"], df["V_7vpp"]*0.1, linestyle="solid", linewidth=vlw, color="pink")
ax.plot(df["time"], df["Estim_8vpp"], linestyle="dotted", linewidth=estimlw, color="gray")
ax.plot(df["time"], df["V_8vpp"]*0.1, linestyle="solid", linewidth=vlw, color="gray")
ax.plot(df["time"], df["Estim_9vpp"], linestyle="dotted", linewidth=estimlw, color="olive")
ax.plot(df["time"], df["V_9vpp"]*0.1, linestyle="solid", linewidth=vlw, color="olive")
ax.plot(df["time"], df["Estim_10vpp"], linestyle="dotted", linewidth=estimlw, color="cyan")
ax.plot(df["time"], df["V_10vpp"]*0.1, linestyle="solid", linewidth=vlw, color="cyan")
"""
ax.spines["top"].set_linewidth(0)
ax.spines["bottom"].set_linewidth(splw)
ax.spines["left"].set_linewidth(splw)
ax.spines["right"].set_linewidth(0)
ax.tick_params(axis="both", length=25, width=5)
plt.setp(ax.get_xticklabels(), fontsize=fontsize)
plt.setp(ax.get_yticklabels(), fontsize=fontsize)

plt.tight_layout()
plt.show()
