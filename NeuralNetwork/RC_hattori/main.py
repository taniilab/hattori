import numpy as np
from reservoir_network import ReservoirNetWork
import matplotlib.pyplot as plt

T = np.pi * 16
RATIO_TRAIN = 0.6

dt = np.pi * 0.06
AMPLITUDE = 0.9
LEAK_RATE=0.1
NUM_RESERVOIR_NODES = 150

def main():
    rc = ReservoirNetWork(Time=100,
                          dt=0.02,
                          N=4)

    for i in range(0, rc.allsteps-1):
        rc.propagation()

    rc.learning()

    fig = plt.figure(figsize=(20, 13))
    ax = []
    for i in range(0, rc.N):
        ax.append(fig.add_subplot(2,2,i+1))
        ax[i].plot(rc.time, rc.V[i], color="blue")
        ax2 = ax[i].twinx()
        ax2.plot(rc.time, rc.Isyn[i], color="gray")
    fig.tight_layout()
    plt.show()


if __name__=="__main__":
    main()