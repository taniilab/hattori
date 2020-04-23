"""
***Unit of parameters***
Brunel & Wang 2001 integrate and fire neuron
membrane potential -> mV
time -> ms
conductance -> mS
capacitance -> uF
current -> uA
kashikoma
"""
from multiprocessing import Pool
import os
from neuron import Neuron_LIF as Neuron
import pandas as pd
import time
import datetime
import itertools
import numpy as np


starttime = time.time()
elapsed_time = 0
save_path = "Z:/simulation/test"
process = 6 #number of processors

#parameters#
numneu = 1
simtime = 4000
lump = 1000
num_lump = int(simtime/lump)
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
                                            'T': lump,
                                            'dt': deltatime,
                                            'Cm': 0.5e-3,
                                            'G_L': 25e-6,
                                            'Vreset': -55,
                                            'Vth': -50,
                                            'erest': -70,
                                            #'Iext_amp': round(j*0.1, 2),
                                            'Iext_amp': 6e-4,
                                            'syn_type': 4,
                                            'Pmax_AMPA': round(i*0.00001, 5),
                                            #'Pmax_AMPA': 0.00003,
                                            'Pmax_NMDA': 0,
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

        # record
        d = datetime.datetime.today()
        filename = "{0}_{1}_{2}_{3}_{4}_{5}_" \
                   "Iext_amp{6}_Pmax_AMPA{7}_Pmax_NMDA{8}_LIF.csv".format(d.year,
                                                                          d.month,
                                                                          d.day,
                                                                          d.hour,
                                                                          d.minute,
                                                                          d.second,
                                                                          self.neuron.Iext_amp,
                                                                          self.neuron.Pmax_AMPA,
                                                                          self.neuron.Pmax_NMDA)
        df = pd.DataFrame(columns=[filename])
        df.to_csv(save_path + '/' + filename)
        df = pd.DataFrame()
        for k in range(numneu):
            df['T_{} [ms]'.format(k)] = ""
            df['V_{} [mV]'.format(k)] = ""
            df['fire_{} [mV]'.format(k)] = ""
            df['I_syn_{} [uA]'.format(k)] = ""
            df['Iext_{} [uA]'.format(k)] = ""
            df['I_noise_{} [uA]'.format(k)] = ""
            df['g_ampa'.format(k)] = ""

        df.to_csv(save_path + '/' + filename, mode='a')

        for j in range(num_lump):
            for i in range(0, self.neuron.allsteps-1):
                self.neuron.propagation()

                if self.progress_counter % 1000 == 0:
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
                df['fire_{} [mV]'.format(k)] = self.neuron.t_fire_list[k]
                df['I_syn_{} [uA]'.format(k)] = self.neuron.Isyn[k]
                df['Iext_{} [uA]'.format(k)] = self.neuron.Iext[k]
                df['I_noise_{} [uA]'.format(k)] = self.neuron.Inoise[k]
                df['g_ampa'.format(k)] = -self.neuron.Isyn[k]/self.neuron.V[k]

            df.to_csv(save_path + '/' + filename, mode='a', header=None)

            # Preparation for calculating the next lump
            print(self.neuron.t_fire)
            print(self.neuron.t_fire - lump)
            self.neuron.Tsteps = self.neuron.Tsteps + lump
            self.neuron.V = np.fliplr(self.neuron.V)
            self.neuron.Isyn = np.fliplr(self.neuron.Isyn)
            self.neuron.Iext = np.fliplr(self.neuron.Iext)
            self.neuron.t_fire_list = 0 * self.neuron.t_fire_list
            self.neuron.Inoise = np.fliplr(self.neuron.Inoise)
            self.neuron.dn = np.fliplr(self.neuron.dn)
            self.neuron.dWt = np.fliplr(self.neuron.dWt)
            self.neuron.t_fire = self.neuron.t_fire - lump
            self.neuron.curstep = 0
            self.neuron.flip_flag = True
            self.neuron.flip_counter += 1


def main():
    d = datetime.datetime.today()
    print("{0}/{1}/{2}/{3}:{4}:{5}".format(d.year, d.month, d.day, d.hour, d.minute, d.second))
    main = Main()

    for i in range(0, main.cycle_multiproc):
        pool = Pool(process)
        pool.map(main.simulate, range(process))
        
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