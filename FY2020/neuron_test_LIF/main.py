"""
***Unit of parameters***
membrane potential -> mV
conductance -> mS
capacitance -> uF
current -> uA
"""
from multiprocessing import Pool
import os
from neuron import Neuron_LIF as Neuron
import pandas as pd
import time
import datetime
import itertools


starttime = time.time()
elapsed_time = 0
save_path = "Z:/simulation/test"
process = 6 #number of processors

#parameters#
numneu = 1
simtime = 500
deltatime = 0.01

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
                                            'Cm': 100e-6,
                                            'Vth': -50,
                                            'G_L': 10e-6,
                                            'erest': -70,
                                            #'Iext_amp': round(j*0.1, 2),
                                            'Iext_amp': 210e-6,
                                            'syn_type': 4,
                                            'Pmax': 0,
                                            #'Pmax': round(i*0.01, 2),
                                            'tau_syn': 5.26,
                                            'noise_type': 1,
                                            'D': 0}
            self.parm_counter += 1
            self.overall_steps = int(self.i*self.j*self.k*self.l*simtime/(deltatime*process))

    def simulate(self, process):
        # parallel processing on each setting value
        self.pid = os.getpid()
        print(process)
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

            filename = "{0}_{1}_{2}_{3}_{4}_{5}Iext_amp{6}_Pmax{7}_LIF.csv".format(d.year,
                                                                        d.month,
                                                                        d.day,
                                                                        d.hour,
                                                                        d.minute,
                                                                        d.second,
                                                                        res[k].Iext_amp,
                                                                        res[k].Pmax)
            df = pd.DataFrame()
            for j in range(numneu):
                df['T_{} [ms]'.format(j)]       = res[k].Tsteps
                df['V_{} [mV]'.format(j)]       = res[k].V[j]
                df['fire_{} [mV]'.format(j)]    = res[k].t_fire_list[j]
                df['I_syn_{} [uA]'.format(j)]   = res[k].Isyn[j]
                df['Iext_{} [uA]'.format(j)]    = res[k].Iext[j]
                df['I_noise_{} [uA]'.format(j)] = res[k].Inoise[j]

            config = pd.DataFrame(columns=[filename])
            config.to_csv(save_path + '/' + filename)
            df.to_csv(save_path + '/' + filename, mode='a')

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