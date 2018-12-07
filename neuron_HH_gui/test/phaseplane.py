import pandas as pd
import matplotlib.pyplot as plt

path = "//192.168.13.10/Public/hattori/simulation/HH/raw_data/2018_10_10_9_46_37/Mg_0.4/" + \
       "2018_10_9_20_52_7_N0_P_AMPA0.4_P_NMDA0.2_Mg_conc0.4_delay0HH.csv"

fsize = 72
sample = 20000
fig = plt.figure(figsize=(21, 14))

df = pd.read_csv(path, delimiter=',', skiprows=1)
x_axis = df["V [mV]"]
y_axis = df["I_K [uA]"]
initial = 1000
last = 1100
dt = 0.04
plt.plot(x_axis[int(initial/dt):int(last/dt)], y_axis[int(initial/dt):int(last/dt)])
plt.show()
