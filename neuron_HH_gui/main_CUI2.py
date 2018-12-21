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
from picture_multi_thread import Picture
#import matplotlib as mpl
# for overflow error
#mpl.rcParams['agg.path.chunksize'] = 100000

starttime = time.time()
elapsed_time = 0
save_path = "//192.168.13.10/Public/hattori/simulation/HH"
#save_path = "Z:/simulation/HH"
#save_path = "G:/simulation/HH"

# number of processors
process = 14

class Main():
    def __init__(self):
        self.parm = []

        #combination
        self.i = 21
        self.j = 21
        self.k = 8
        self.l = 1

        self.cycle_multiproc = int(self.i * self.j*self.k*self.l/process)
        self.multiproc_co = 0
        self.parm_counter = 0

        for i, j, k, l in itertools.product(range(self.i),
                                            range(self.j),
                                            range(self.k),
                                            range(self.l)):
            self.parm.append({})
            self.parm[self.parm_counter] = {'T': 5000,
                                            'dt': 0.02,
                                            'Iext_amp': 10,
                                            'syncp': 6,
                                            'noise': 2,
                                            'gpNa': 0,
                                            'Pmax_AMPA': round(0.1*i,2),
                                            'Pmax_NMDA': round(0.1*j,2),
                                            'gtCa': 0,
                                            'Mg_conc': round(0.1+k*0.3,2),
                                            'alpha': 0.5,
                                            'beta': 0.1,
                                            'D': 0.5,
                                            'U_SE_AMPA':0.7,
                                            'U_SE_NMDA':0.03,
                                            'tau_rise_AMPA':1.1,
                                            'tau_rise_NMDA':145,
                                            'tau_rec_AMPA':200,
                                            'tau_rec_NMDA':200,
                                            'tau_inact_AMPA':5,
                                            'tau_inact_NMDA':55,
                                            'delay': 0}
            self.parm_counter += 1


    def simulate(self, process):
        # parallel processing on each setting value
        self.pid = os.getpid()
        self.progress_co = 0
        self.neuron = Neuron(**self.parm[process+self.multiproc_co])
        self.neuron.parm_dict = self.parm[process+self.multiproc_co]

        for i in range(0, self.neuron.allsteps-1):
            self.neuron.propagation()

            if self.progress_co % 100000 == 0:
                self.log = 'process id : ' + str(self.pid) + ' : ' + \
                            str(self.progress_co) + ' steps'
                print(self.log)

            self.progress_co += 1
        return self.neuron


def main():
    main = Main()

    for i in range(0, main.cycle_multiproc):
        pool = Pool(process)
        res = pool.map(main.simulate, range(process))

        # for recording
        tmp = []
        for k in range(process):
            tmp.append(res[k].N)

        print(main.multiproc_co)
        main.multiproc_co += process

        # record
        for k, j in itertools.product(range(process), range(tmp[k])):
            d = datetime.datetime.today()

            # generate file name
            res[k].parm_dict = str(res[k].parm_dict)
            res[k].parm_dict = res[k].parm_dict.replace(':', '_')
            res[k].parm_dict = res[k].parm_dict.replace('{', '_')
            res[k].parm_dict = res[k].parm_dict.replace('}', '_')
            res[k].parm_dict = res[k].parm_dict.replace('\'', '')
            res[k].parm_dict = res[k].parm_dict.replace(',', '_')
            """
            filename = (str(d.year) + '_' + str(d.month) + '_' +
                        str(d.day) + '_' + str(d.hour) + '_' +
                        str(d.minute) + '_' + str(d.second) + '_' +
                        res[k].parm_dict + '_' + 'N' + str(j) + '_' + "HH.csv")
            """
            filename = (str(d.year) + '_' + str(d.month) + '_' +
                        str(d.day) + '_' + str(d.hour) + '_' +
                        str(d.minute) + '_' + str(d.second) + '_' +
                        'N' + str(j) +
                        "_P_AMPA" + str(res[k].Pmax_AMPA) + "_P_NMDA" + str(res[k].Pmax_NMDA) +
                        "_Mg_conc" + str(res[k].Mg_conc) + '_' + 'delay' + str(res[k].delay) + "HH.csv")

            df = pd.DataFrame({'T [ms]': res[k].Tsteps,
                               'V [mV]': res[k].V[j],
                               'I_K [uA]': res[k].IK[j],
                               'I_Na [uA]': res[k].INa[j],
                               'I_m [uA]': res[k].Im[j],
                               'I_leak [uA]': res[k].Ileak[j],
                               'I_syn [uA]': res[k].Isyn[j],
                               'I_AMPA [uA]': res[k].IAMPA[j],
                               'I_NMDA [uA]': res[k].INMDA[j],
                               'E_AMPA [uA]': res[k].E_AMPA[0, j],
                               'E_NMDA [uA]': res[k].E_NMDA[0, j],
                               'Iext [uA]': res[k].Iext[j],
                               'I_noise [uA]': res[k].Inoise[j]})
            config = pd.DataFrame(columns=[filename])
            config.to_csv(save_path + '/' + filename)
            df.to_csv(save_path + '/' + filename, mode='a')

        pool.close()
        pool.join()

    # sample plotting
    for i in range(0, process):
        # initialize
        ax = []
        lines = []
        tm = np.arange(0, res[i].allsteps*res[i].dt, res[i].dt)

        # matrix
        fig = plt.figure(figsize=(12, 12))
        gs = grs.GridSpec(4, res[i].N)

        for j in range(0, res[i].N):
            ax.append(plt.subplot(gs[0, j]))
            ax.append(plt.subplot(gs[1, j]))
            ax.append(plt.subplot(gs[2, j]))
            ax.append(plt.subplot(gs[3, j]))

        # plot
        for j in range(0, res[i].N):
            lines.append([])
            lines[j], = ax[j].plot(tm, res[i].V[j], color="indigo",
                                   markevery=[0, -1])

        ax[res[i].N].plot(tm, res[i].INa[0], color="coral", markevery=[0, -1])

        ax[res[i].N+1].plot(tm, res[i].INMDA[0], color="coral",
                           markevery=[0, -1])

        ax[res[i].N+2].plot(tm, res[i].IAMPA[0], color="coral",
                           markevery=[0, -1])
        fig.tight_layout()

    plt.show()

    elapsed_time = time.time() - starttime
    #pic = Picture(save_path)
    #pic.run()

    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
    print("")
    print("続行するには何かキーを押してください . . .")
    input()


if __name__ == '__main__':
     main()
