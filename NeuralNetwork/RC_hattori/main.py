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
    rc.predict()

    fig = plt.figure(figsize=(15, 10))
    ax = []
    for i in range(0, rc.N):
        ax.append(fig.add_subplot(2,2,i+1))
        line1, = ax[i].plot(rc.time, rc.V[i], color="blue")
        ax2 = ax[i].twinx()
        line2, = ax2.plot(rc.time, rc.Isyn[i], color="gray")
        line3, = ax2.plot(rc.time, rc.Iext[i], color="red")
        line4, = ax2.plot(rc.time, rc.predicted_results,  color="green")
        ax[i].legend([line1, line2, line3, line4], ["V", "Isyn", "Iext", "res"])
    fig.tight_layout()
    plt.show()

if __name__=="__main__":
    main()