import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import numpy as np

t = np.arange(0, 100)
y = t

numneu = 5
fig = plt.figure(figsize=(20, 15))
gs_master = GridSpec(nrows=numneu+1, ncols=2)

gs_1 = GridSpecFromSubplotSpec(nrows=1, ncols=2, subplot_spec=gs_master[0, 0:2])
ax_rc = fig.add_subplot(gs_1[:, :])

gs_2_and_3 = GridSpecFromSubplotSpec(nrows=numneu, ncols=2, subplot_spec=gs_master[1:, :], hspace=0.4, wspace=0.1)
ax_status_v = []
ax_status_i = []
for i in range(numneu):
    ax_status_v.append(fig.add_subplot(gs_2_and_3[i, 0]))
    ax_status_i.append(fig.add_subplot(gs_2_and_3[i, 1]))

ax_status_v[4].plot(t, y)

plt.show()