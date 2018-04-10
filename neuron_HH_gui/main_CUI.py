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
from neuron import Neuron_HH as Neuron
import pandas as pd
import time
import datetime
import itertools
from picture import Picture

starttime = time.time()
elapsed_time = 0
# save_path = "C:/Users/Hattori/Documents/HR_results/20171115alphafunction"
# save_path = "C:/Users/6969p/Documents/HR_results/20171115alphafunction"
save_path = "F:/simulation/HH"

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
                self.log = 'process id : ' + str(self.pid) + ' : ' + \
                            str(self.progress_co) + ' steps'
                print(self.log)

            self.progress_co += 1
        return self.nr

    def form_parm(self):
        self.parm = []
        self.cycle_multiproc = int(6 / 6)
        self.multiproc_co = 0
        self.parm_counter = 0

        for i, j, k, l in itertools.product(range(6), range(1), range(1),
                                            range(1)):
            self.parm.append({})
            self.parm[self.parm_counter] = {'Iext_amp': 0,
                                            'Syncp': 5, 'Pmax': 3}
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
            tmp.append(cb[k].N)

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
                        cb[k].parm_dict + '_' + 'N' + str(j) + '_' + "HH.csv")

            df = pd.DataFrame({'t': cb[k].Tsteps, 'v': cb[k].V[j],
                               'syn': cb[k].Isyn[j]})
            df.to_csv(save_path + '/' + filename)

        pool.close()
        pool.join()

    # sample plotting
    for i in range(0, process):
        # initialize
        ax = []
        lines = []
        tm = np.arange(0, cb[i].allsteps*cb[i].dt, cb[i].dt)

        # matrix
        fig = plt.figure(figsize=(12, 12))
        gs = grs.GridSpec(4, cb[i].N)

        for j in range(0, cb[i].N):
            ax.append(plt.subplot(gs[0, j]))
            ax.append(plt.subplot(gs[1, j]))
            ax.append(plt.subplot(gs[2, j]))
            ax.append(plt.subplot(gs[3, j]))

        # plot
        for j in range(0, cb[i].N):
            lines.append([])
            lines[j], = ax[j].plot(tm, cb[i].V[j], color="indigo",
                                   markevery=[0, -1])

        ax[cb[i].N].plot(tm, cb[i].Isyn[0], color="coral", markevery=[0, -1])

        ax[cb[i].N+1].plot(tm, cb[i].INMDA[0], color="coral",
                           markevery=[0, -1])

        ax[cb[i].N+2].plot(tm, cb[i].IAMPA[0], color="coral",
                           markevery=[0, -1])
        """
        # adjusting
        for j in range(0, cb[i].N+2):
            ax[j].grid(which='major', color='thistle', linestyle='-')
        """

        fig.tight_layout()

    plt.show()

    elapsed_time = time.time() - starttime
    pic = Picture(save_path)
    pic.run()

    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
    print("")
    print("続行するには何かキーを押してください . . .")
    input()


if __name__ == '__main__':
    main()
