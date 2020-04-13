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
numneu = 10
simtime = 1000
deltatime = 0.04

class Main():
    def __init__(self):
        self.parm = []

        #combination
        self.i = 1
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
                                            'Iext_amp': 10,
                                            'syn_type': 4,
                                            'noise': 2,
                                            'gpNa': 0,
                                            'gkCa': 0,
                                            'Pmax_AMPA': round(i*0.1, 2),
                                            'Pmax_NMDA': round(j*0.1, 2),
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

            filename = "{0}_{1}_{2}_{3}_{4}_{5}_P_AMPA{6}_" \
                       "P_NMDA{7}_Mg_conc{8}_HH.csv".format(d.year,
                                                                     d.month,
                                                                     d.day,
                                                                     d.hour,
                                                                     d.minute,
                                                                     d.second,
                                                                     res[k].Pmax_AMPA,
                                                                     res[k].Pmax_NMDA,
                                                                     res[k].Mg_conc)
            df = pd.DataFrame()
            for j in range(numneu):
                df['T_{} [ms]'.format(j)]       = res[k].Tsteps
                df['V_{} [mV]'.format(j)]       = res[k].V[j]
                df['fire_{} [mV]'.format(j)]    = res[k].t_fire_list[j]
                df['I_K_{} [uA]'.format(j)]     = res[k].IK[j]
                df['I_Na_{} [uA]'.format(j)]    = res[k].INa[j]
                df['Ca_conc_{} [uM]'.format(j)] = res[k].ca_influx[j]
                df['I_kCa_{} [uA]'.format(j)]   = res[k].IlCa[j]
                df['I_m_{} [uA]'.format(j)]     = res[k].Im[j]
                df['I_leak_{} [uA]'.format(j)]  = res[k].Ileak[j]
                df['I_syn_{} [uA]'.format(j)]   = res[k].Isyn[j]
                df['I_AMPA_{} [uA]'.format(j)]  = res[k].IAMPA[j]
                df['I_NMDA_{} [uA]'.format(j)]  = res[k].INMDA[j]
                df['g_AMPA_{} [x100 mS]'.format(j)]  = -100 * res[k].IAMPA[j]/res[k].V[j]
                df['g_NMDA_{} [x100 mS]'.format(j)]  = -100 * res[k].INMDA[j]/res[k].V[j]
                df['E_AMPA_{} [uA]'.format(j)]  = res[k].E_AMPA[0, j]
                df['E_NMDA_{} [uA]'.format(j)]  = res[k].E_NMDA[0, j]
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
    #pic = Picture(save_path)
    #pic.run()
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

if __name__ == '__main__':
     main()