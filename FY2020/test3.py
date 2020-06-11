import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

save_path = "Z:/simulation/test"
filename = "2020_6_10_23_30_23_Iext_amp0.001_Pmax_AMPA6e-05_Pmax_NMDA7e-05_LIF"

#parameters#
numneu = 5
simtime = 1000
lump = 500
num_lump = int(simtime/lump)
dt = 0.02


# Rastergram
if not os.path.isdir(save_path + '/rastergram'):
    os.mkdir(save_path + '/rastergram')
num_read_nodes = numneu
raster_line_length = 1
raster_line_width = 0.5
read_cols = ['T_0 [ms]']
ytick_list = []
for i in range(num_read_nodes):
    ytick_list.append(i+1)
    read_cols.append('fire_{}'.format(i))

df = pd.read_csv(save_path + '/' + filename + '.csv', usecols=read_cols, skiprows=1)[read_cols]
plt.rcParams["font.size"] = 28
fig_r = plt.figure(figsize=(20,10))
ax = fig_r.add_subplot(111)
ax.set_ylim(0, num_read_nodes+1)
ax.set_yticks(ytick_list)
ax.set_xlabel("Time [ms]")
ax.set_ylabel("Neuron number")
for i in range(num_read_nodes):
    for j in range(len(df.values[:, 0])):
        if df.values[j, i + 1] != 0:
            x = df.values[j, 0]
            ax.plot([x, x], [i+1-(raster_line_length/2), i+1+(raster_line_length/2)],
                    linestyle="solid",
                    linewidth=raster_line_width,
                    color="black")
plt.tight_layout()
plt.savefig(save_path + '/rastergram/' + filename + '.png')
