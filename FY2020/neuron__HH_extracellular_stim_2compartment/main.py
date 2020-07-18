# coding: UTF-8
from multiprocessing import Pool
import os
from neuron import Neuron_HH as Neuron
import pandas as pd
import time
import datetime
import itertools
import matplotlib.pyplot as plt



starttime = time.time()
elapsed_time = 0
save_path = "G:/Box Sync/Personal/xxx"
process = 6 #number of processors
numneu = 2 # fix(2 compartment)
simtime = 2000
deltatime = 0.04

class Main():
    def __init__(self):
        self.parm = []

        #combination
        self.i = 6
        self.j = 1
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
            self.parm[self.parm_counter] = {'N': numneu,
                                            'T': simtime,
                                            'dt': deltatime,
                                            'g_intra': 2,
                                            'tau_vextra':round(10*i*i, 4)}
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

            if self.progress_counter % 1000 == 0:
                self.log = 'process id : ' + str(self.pid) + ' : ' + \
                            str(self.progress_counter) + ' steps : ' + \
                            str(round(self.progress_counter*100/self.overall_steps, 1)) + "%"
                print(self.log)

            self.progress_counter += 1
        return self.neuron


def main():
    d = datetime.datetime.today()
    print("{0}/{1}/{2}/{3}:{4}:{5}".format(d.year, d.month, d.day, d.hour, d.minute, d.second))
    fig_folder_path = save_path + '/sample_plot'
    if not os.path.isdir(fig_folder_path):
        os.mkdir(fig_folder_path)
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

            filename = "{0}_{1}_{2}_{3}_{4}_{5}_gintra{6}_HH.csv".format(d.year,
                                                                     d.month,
                                                                     d.day,
                                                                     d.hour,
                                                                     d.minute,
                                                                     d.second,
                                                                     res[k].g_intra)
            df = pd.DataFrame()
            for j in range(numneu):
                df['T_{} [ms]'.format(j)]       = res[k].Tsteps
                df['V_{} [mV]'.format(j)]       = res[k].V[j]
                df['V_intra{} [mV]'.format(j)] = res[k].V_intra[j]
                df['V_extra{} [mV]'.format(j)] = res[k].V_extra[j]
                df['fire_{} [mV]'.format(j)]    = res[k].t_fire_list[j]
                df['I_K_{} [uA]'.format(j)]     = res[k].IK[j]
                df['I_Na_{} [uA]'.format(j)]    = res[k].INa[j]
                df['I_leak_{} [uA]'.format(j)]  = res[k].Ileak[j]
                df['I_link_{} [uA]'.format(j)] = res[k].Ilink[j]

            config = pd.DataFrame(columns=[filename])
            config.to_csv(save_path + '/' + filename)
            df.to_csv(save_path + '/' + filename, mode='a')

            # plot

            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111)
            for j in range(numneu):
                ax.plot(res[k].Tsteps, res[k].V_extra[j], linewidth=3)
                ax.plot(res[k].Tsteps, res[k].V[j], linewidth=3)
            ax.set_xlim(998, 1006)
            plt.tight_layout()
            plt.savefig(fig_folder_path+'/'+str(os.path.splitext(os.path.basename(filename))[0])+'.png')
            plt.close(fig)
            #plt.show()

        pool.close()
        pool.join()
        print("---------------------------\n")
        main.process_counter += process
        main.now_cycle_multiproc += 1


    d = datetime.datetime.today()
    print("{0}/{1}/{2}/{3}:{4}:{5}".format(d.year, d.month, d.day, d.hour, d.minute, d.second))
    elapsed_time = time.time() - starttime

    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

if __name__ == '__main__':
     main()