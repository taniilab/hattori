"""
Created on Sat May 27 10:49:16 2017

@author: Hattori
"""
# coding: UTF-8
from multiprocessing import Pool
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as grs
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sb
from neuron import Neuron_HR as Neuron
import pandas as pd
import time
import datetime 
import logging
import itertools

starttime = time.time()
elapsed_time = 0

# palameter setting
"""
type of synaptic coupling(Syncp)
1.electrical synapse
2.chemical synapse
3.alpha function
4.alpha function with excitatory and inhibitory synapse
5.depressing synapse

type of noise(noise)
0.none
1.White Gaussian process
2.Ornstein-Uhlenbeck process
3.sin wave
"""


class Main():
    def __init__(self, numproc):
        self.numproc = numproc

    def simulate(self, process):
        # parallel processing on each setting value
        self.pid = os.getpid()
        self.progress_co = 0
        self.nr = Neuron(**self.parm[process+self.multiproc_co])
        self.nr.parm_dict = self.parm[process+self.multiproc_co]

        for i in range(0, self.nr.allsteps-1):
            self.nr.propagation()
            if self.progress_co % 100000 == 0:
                logging.warning('process id : %d : %4d steps',
                                self.pid, self.progress_co)

            self.progress_co += 1
        return self.nr

    def form_parm(self):
        self.parm = []
        self.cycle_multiproc = int(120 / 6)
        self.multiproc_co = 0
        self.parm_counter = 0

        """
        for i, j, k, l in itertools.product(range(12), range(1), range(1),
                                            range(1)):
            self.parm.append({})
            self.parm[self.parm_counter] = {"numneu": 10, "Syncp": 4,
                                            "Iext": round(i*0.1+1.5, 1)}
            self.parm_counter += 1

        """
        for i, j, k, l in itertools.product(range(1), range(15), range(6),
                                            range(1)):
            self.parm.append({})
            self.parm[self.parm_counter] = {"numneu": 10, "noise": 2,
                                            "D": round(j, 1),
                                            "Syncp": 4,
                                            "Pmax": round(k*0.4, 1),
                                            }
            self.parm_counter += 1


def main():
    process = 6
    main = Main(process)
    main.form_parm()

    for i in range(0, main.cycle_multiproc):

        pool = Pool(process)
        cb = pool.map(main.simulate, range(process))

        # for recording
        tmp = []
        for k in range(process):
            tmp.append(cb[k].numneu)

        print(main.multiproc_co)
        main.multiproc_co += process

        # record
        for k, j in itertools.product(range(process), range(tmp[k])):
            d = datetime.datetime.today()

            # generate file name
            cb[k].parm_dict = str(cb[k].parm_dict)
            cb[k].parm_dict = cb[k].parm_dict.replace(':', '_')
            cb[k].parm_dict = cb[k].parm_dict.replace('{', '_')
            cb[k].parm_dict = cb[k].parm_dict.replace('}', '_')
            cb[k].parm_dict = cb[k].parm_dict.replace('\'', '')
            cb[k].parm_dict = cb[k].parm_dict.replace(',', '_')
            filename = (str(d.year) + '_' + str(d.month) + '_' +
                        str(d.day) + '_' + str(d.hour) + '_' +
                        str(d.minute) + '_' + str(d.second) + '_' +
                        cb[k].parm_dict + '_' + 'N' + str(j) + '_' + "HR.csv")

            df = pd.DataFrame({'t': cb[k].tmhist, 'Iext': cb[k].Iext[j, 1],
                               'x': cb[k].x[j],
                               'y': cb[k].y[j], 'z': cb[k].z[j],
                               'Isyn': cb[k].Isyn[j], 'alpha': cb[k].alpha,
                               'beta': cb[k].beta, 'D': cb[k].D,
                               'tausyn': cb[k].tausyn, 'Pmax': cb[k].Pmax})
            df.to_csv('C:/Users/Hattori/Documents/HR_results/' + filename)

        pool.close()
        pool.join()

        print("じゅ")
        print("ぴっぴ")

    # sample plotting
    for i in range(0, process):
        # initialize
        ax = []
        lines = []
        tm = np.arange(0, cb[i].allsteps*cb[i].dt, cb[i].dt)

        fig = plt.figure(figsize=(12, 12))
        gs = grs.GridSpec(3, cb[i].numneu)

        for j in range(0, cb[i].numneu):
            ax.append(plt.subplot(gs[0, j]))
        ax.append(plt.subplot(gs[1, :]))
        ax.append(plt.subplot(gs[2, :]))

        # plot
        for j in range(0, cb[i].numneu):
            lines.append([])
            if cb[i].numneu == 1:
                lines[j], = ax[j].plot(tm, cb[i].x[j], color="indigo",
                                       markevery=[0, -1])
            else:
                lines[j], = ax[j].plot(tm, cb[i].x[j], color="indigo",
                                       markevery=[0, -1])

        ax[cb[i].numneu].plot(tm, cb[i].Isyn[0], color="coral",
                              markevery=[0, -1])
        ax2 = ax[cb[i].numneu].twinx()
        ax2.plot(tm, cb[i].x[0], color="indigo", markevery=[0, -1])

        ax[cb[i].numneu+1].plot(tm, cb[i].z[0], color="coral",
                                markevery=[0, -1])
        ax2 = ax[cb[i].numneu+1].twinx()
        ax2.plot(tm, cb[i].x[0], color="indigo", markevery=[0, -1])

        # adjusting
        for j in range(0, cb[i].numneu+2):
            ax[j].grid(which='major', color='thistle', linestyle='-')
        fig.tight_layout()

    elapsed_time = time.time() - starttime
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
    print("ちょう終わりました～♪")


if __name__ == '__main__':
    main()
