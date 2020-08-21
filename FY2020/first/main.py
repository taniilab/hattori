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
save_path = "H:/simulation/HH"

process = 15  # number of processors
numneu = 1
simtime = 5000
lump = 2500
num_lump = int(simtime/lump)
dt = 0.04

class Main():
    def __init__(self):
        self.parm = []

        #combination
        self.i = 15
        self.j = 1
        self.k = 1
        self.l = 1

        self.cycle_multiproc = int(self.i * self.j*self.k*self.l/process)
        self.process_counter = 0
        self.process_counter = 0
        self.lump_counter = 0
        self.now_cycle_multiproc = 0
        self.parm_counter = 0

        for i, j, k, l in itertools.product(range(self.i),
                                            range(self.j),
                                            range(self.k),
                                            range(self.l)):
            self.parm.append({})
            self.parm[self.parm_counter] = {'N': numneu,
                                            'T': lump,
                                            'dt': dt,
                                            'Iext_amp': 10,
                                            'syncp': 6,
                                            'noise': 2,
                                            'gpNa': 0,
                                            #'gkCa': 0.00002,
                                            'gkCa': 0,
                                            'Pmax_AMPA': round(0.1*i, 3),
                                            'Pmax_NMDA': 0,
                                            'gtCa': 0,
                                            'esyn': -70,
                                            'Mg_conc': 1.3,
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
            self.overall_steps = int(self.i*self.j*self.k*self.l*simtime/(dt*process))


    def simulate(self, process):
        # parallel processing on each setting value
        self.pid = os.getpid()
        self.neuron = Neuron(**self.parm[process+self.process_counter])
        self.neuron.parm_dict = self.parm[process+self.process_counter]
        self.progress_counter = self.now_cycle_multiproc*self.neuron.allsteps

        # record
        d = datetime.datetime.today()
        filename = "{0}_{1}_{2}_{3}_{4}_{5}_" \
                   "N{6}_P_AMPA{7}_P_NMDA{8}_Mg_conc{9}_gkCa{10}_HH".format(d.year,
                                                                            d.month,
                                                                            d.day,
                                                                            d.hour,
                                                                            d.minute,
                                                                            d.second,
                                                                            numneu,
                                                                            self.neuron.Pmax_AMPA,
                                                                            self.neuron.Pmax_NMDA,
                                                                            self.neuron.Mg_conc,
                                                                            self.gkCa)
        df = pd.DataFrame(columns=[filename])
        df.to_csv(save_path + '/' + filename + '.csv')
        df = pd.DataFrame()


        for k in range(numneu):
            df['T_{} [ms]'.format(k)] = ""
            df['V_{} [mV]'.format(k)] = ""
            df['fire_{}'.format(k)] = ""
            df['I_K_{} [uA]'.format(k)] = ""
            df['I_Na_{} [uA]'.format(k)] = ""
            df['Ca_conc_{} [nm?]'.format(k)] = ""
            df['I_kCa_{} [uA]'.format(k)] = ""
            df['I_m_{} [uA]'.format(k)] = ""
            df['I_leak_{} [uA]'.format(k)] = ""
            df['I_syn_{} [uA]'.format(k)] = ""
            df['I_AMPA_{} [uA]'.format(k)] = ""
            df['I_NMDA_{} [uA]'.format(k)] = ""
            df['E_AMPA_{}'.format(k)] = ""
            df['E_NMDA_{}'.format(k)] = ""
            df['Iext_{} [uA]'.format(k)] = ""
            df['I_noise_{} [uA]'.format(k)] = ""
        df.to_csv(save_path + '/' + filename + '.csv', mode='a')

        ####### MAIN PROCESS #######
        for j in range(num_lump):
            for i in range(0, self.neuron.allsteps-1):
                self.neuron.propagation()

                if self.progress_counter % 5000 == 0:
                    self.log = 'process id : ' + str(self.pid) + ' : ' + \
                                str(self.progress_counter) + ' steps : ' + \
                                str(round(self.progress_counter*100/self.overall_steps, 1)) + "%"
                    print(self.log)
                self.progress_counter += 1

            # record
            df = pd.DataFrame()
            for k in range(numneu):
                df['T_{} [ms]'.format(k)] = self.neuron.Tsteps
                df['V_{} [mV]'.format(k)] = self.neuron.V[k]
                df['fire_{}'.format(k)] = self.neuron.t_fire_list[k]
                df['I_K_{} [uA]'.format(k)] = self.neuron.IK[k]
                df['I_Na_{} [uA]'.format(k)] = self.neuron.INa[k]
                df['Ca_conc_{} [nm?]'.format(k)] = self.neuron.ca_influx[k]
                df['I_kCa_{} [uA]'.format(k)] = self.neuron.IlCa[k]
                df['I_m_{} [uA]'.format(k)] = self.neuron.Im[k]
                df['I_leak_{} [uA]'.format(k)] = self.neuron.Ileak[k]
                df['I_syn_{} [uA]'.format(k)] = self.neuron.Isyn[k]
                df['I_AMPA_{} [uA]'.format(k)] = self.neuron.IAMPA[k]
                df['I_NMDA_{} [uA]'.format(k)] = self.neuron.INMDA[k]
                df['E_AMPA_{}'.format(k)] = self.neuron.E_AMPA[0, k]
                df['E_NMDA_{}'.format(k)] = self.neuron.E_NMDA[0, k]
                df['Iext_{} [uA]'.format(k)] = self.neuron.Iext[k]
                df['I_noise_{} [uA]'.format(k)] = self.neuron.Inoise[k]
            df = df[:-1]
            df.to_csv(save_path + '/' + filename + '.csv', mode='a', header=None)

            # Preparation for calculating the next lump
            self.neuron.Tsteps = self.neuron.Tsteps + lump
            self.neuron.V = np.fliplr(self.neuron.V)
            self.neuron.Isyn = np.fliplr(self.neuron.Isyn)
            self.neuron.Isyn[:, 1:] = 0
            self.neuron.IAMPA = np.fliplr(self.neuron.IAMPA)
            self.neuron.IAMPA[:, 1:] = 0
            self.neuron.INMDA = np.fliplr(self.neuron.INMDA)
            self.neuron.INMDA[:, 1:] = 0
            self.neuron.R_AMPA = np.flip(self.neuron.R_AMPA, axis=2)
            self.neuron.R_AMPA[:, :, 1:] = 0
            self.neuron.R_NMDA = np.flip(self.neuron.R_NMDA, axis=2)
            self.neuron.R_NMDA[:, :, 1:] = 0
            self.neuron.E_AMPA = np.flip(self.neuron.E_AMPA, axis=2)
            self.neuron.E_AMPA[:, :, 1:] = 0
            self.neuron.E_NMDA = np.flip(self.neuron.E_NMDA, axis=2)
            self.neuron.E_NMDA[:, :, 1:] = 0
            self.neuron.I_AMPA = np.flip(self.neuron.I_AMPA, axis=2)
            self.neuron.I_AMPA[:, :, 1:] = 0
            self.neuron.I_NMDA = np.flip(self.neuron.I_NMDA, axis=2)
            self.neuron.I_NMDA[:, :, 1:] = 0
            self.neuron.Iext = np.fliplr(self.neuron.Iext)
            self.neuron.t_fire_list = 0 * self.neuron.t_fire_list
            self.neuron.Inoise = np.fliplr(self.neuron.Inoise)
            self.neuron.dn = np.fliplr(self.neuron.dn)
            self.neuron.dWt = np.fliplr(self.neuron.dWt)
            self.neuron.curstep = 0
            self.lump_counter += 1
        ####### MAIN PROCESS END#######


def main():
    main = Main()

    for i in range(0, main.cycle_multiproc):
        pool = Pool(process)
        pool.map(main.simulate, range(process))

        # for recording
        tmp = []
        for k in range(process):
            tmp.append(res[k].N)

        print(main.multiproc_counter)
        main.multiproc_counter += process

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
                        "_Mg_conc" + str(res[k].Mg_conc) + '_' + 'gkCa' + str(res[k].gkCa) + "HH.csv")

            df = pd.DataFrame({'T [ms]': res[k].Tsteps,
                               'V [mV]': res[k].V[j],
                               'I_K [uA]': res[k].IK[j],
                               'I_Na [uA]': res[k].INa[j],
                               'Ca_conc [nm?]': res[k].ca_influx[j],
                               'I_kCa [uA]': res[k].IlCa[j],
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

        ax[res[i].N+2].plot(tm, res[i].ca_influx[0], color="coral",
                           markevery=[0, -1])
        fig.tight_layout()

    plt.show()

    pic = Picture(save_path)
    pic.run()

    d = datetime.datetime.today()
    print("{0}/{1}/{2}/{3}:{4}:{5}".format(d.year, d.month, d.day, d.hour, d.minute, d.second))
    elapsed_time = time.time() - starttime
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")


if __name__ == '__main__':
     main()
