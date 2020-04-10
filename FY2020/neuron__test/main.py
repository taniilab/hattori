"""
Created on Sat May 27 10:49:16 2017

@author: Hattori
"""
# coding: UTF-8
from multiprocessing import Pool
import os
from neuron import Neuron_HH as Neuron
import pandas as pd
import time
import datetime
import itertools


starttime = time.time()
elapsed_time = 0
save_path = "Z:/simulation/test"
process = 6 #number of processors
simtime = 1000
deltatime = 0.04

class Main():
    def __init__(self):
        self.parm = []

        #combination
        self.i = 6
        self.j = 3
        self.k = 1
        self.l = 1

        self.cycle_multiproc = int(self.i * self.j*self.k*self.l/process)
        self.process_counter = 0
        self.now_cycle_multiproc = 0
        self.parm_counter = 0

        for i, j, k, l in itertools.product(range(self.i),
                                            range(self.j),
                                            range(self.k),
                                            range(self.l)):
            self.parm.append({})
            self.parm[self.parm_counter] = {'N': 2,
                                            'T': simtime,
                                            'dt': deltatime,
                                            'Iext_amp': 10,
                                            'syn_type': 6,
                                            'noise': 2,
                                            'gpNa': 0,
                                            'gkCa': round(0.00005*i, 6),
                                            'Pmax_AMPA': 0,
                                            'Pmax_NMDA': 0.8,
                                            'gtCa': 0,
                                            'Mg_conc': 1.3,
                                            'alpha': 0,
                                            'beta': 0,
                                            'D': 0,
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
            self.overall_steps = int(self.i*self.j*self.k*self.l*simtime/(deltatime*process))

    def simulate(self, process):
        # parallel processing on each setting value
        self.pid = os.getpid()
        self.neuron = Neuron(**self.parm[process+self.process_counter])
        self.neuron.parm_dict = self.parm[process+self.process_counter]
        self.progress_counter = self.now_cycle_multiproc*self.neuron.allsteps

        for i in range(0, self.neuron.allsteps-1):
            self.neuron.propagation()

            if self.progress_counter % 10000 == 0:
                self.log = 'process id : ' + str(self.pid) + ' : ' + \
                            str(self.progress_counter) + ' steps : ' + \
                            str(round(self.progress_counter*100/self.overall_steps, 1)) + "%"
                print(self.log)

            self.progress_counter += 1
        return self.neuron


def main():
    main = Main()

    for i in range(0, main.cycle_multiproc):
        pool = Pool(process)
        res = pool.map(main.simulate, range(process))
        res_list = []
        for k in range(process):
            res_list.append(res[k].N)

        # record
        for k in range(process):
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
                        'N' + str(j) +
                        "_P_AMPA" + str(res[k].Pmax_AMPA) + "_P_NMDA" + str(res[k].Pmax_NMDA) +
                        "_Mg_conc" + str(res[k].Mg_conc) + '_' + 'gkCa' + str(res[k].gkCa) + "HH.csv")
            """
            filename = "{0}_{1}_{2}_{3}_{4}_{5}_P_AMPA{6}_" \
                       "P_NMDA{7}_Mg_conc{8}_gkCa{9}_HH.csv".format(d.year,
                                                                     d.month,
                                                                     d.day,
                                                                     d.hour,
                                                                     d.minute,
                                                                     d.second,
                                                                     res[k].Pmax_AMPA,
                                                                     res[k].Pmax_NMDA,
                                                                     res[k].Mg_conc,
                                                                     res[k].gkCa)
            df = pd.DataFrame()
            """
            df = pd.DataFrame({'T_{} [ms]'.format(j): res[k].Tsteps,
                               'V_{} [mV]'.format(j): res[k].V[j],
                               'I_K_{} [uA]'.format(j): res[k].IK[j],
                               'I_Na_{} [uA]'.format(j): res[k].INa[j],
                               'Ca_conc_{} [uM]'.format(j): res[k].ca_influx[j],
                               'I_kCa_{} [uA]'.format(j): res[k].IlCa[j],
                               'I_m_{} [uA]'.format(j): res[k].Im[j],
                               'I_leak_{} [uA]'.format(j): res[k].Ileak[j],
                               'I_syn_{} [uA]'.format(j): res[k].Isyn[j],
                               'I_AMPA_{} [uA]'.format(j): res[k].IAMPA[j],
                               'I_NMDA_{} [uA]'.format(j): res[k].INMDA[j],
                               'E_AMPA_{} [uA]'.format(j): res[k].E_AMPA[0, j],
                               'E_NMDA_{} [uA]'.format(j): res[k].E_NMDA[0, j],
                               'Iext_{} [uA]'.format(j): res[k].Iext[j],
                               'I_noise_{} [uA]'.format(j): res[k].Inoise[j]})
            """
            for j in range()
            df['kashikoma'] =  res[k].Tsteps
            df['kanopero'] = res[k].V[j]


            config = pd.DataFrame(columns=[filename])
            config.to_csv(save_path + '/' + filename)
            df.to_csv(save_path + '/' + filename, mode='a')

        pool.close()
        pool.join()
        print("---------------------------\n")
        main.process_counter += process
        main.now_cycle_multiproc += 1

    # sample plotting
    """
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
    """

    elapsed_time = time.time() - starttime
    #pic = Picture(save_path)
    #pic.run()

    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
    """
    print("")
    print("終了するには何かキーを押してください . . .")
    input()
    """

if __name__ == '__main__':
     main()