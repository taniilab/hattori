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
#mpl.rcParams['agg.path.chunksize'] = 100000ほくろ

starttime = time.time()
elapsed_time = 0
save_path = "C:/sim"

process = 20  # number of processors
numneu = 1
simtime = 5000
lump = simtime
num_lump = int(simtime/lump)
dt = 0.02

class Main():
    def __init__(self):
        self.parm = []

        #combination
        self.i = 16
        self.j = 16
        self.k = 1
        self.l = 5

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
                                            'syncp': 3,
                                            'noise': 2,
                                            'gpNa': 0,
                                            'gkCa': 0.0002,
                                            #'gkCa': 0,
                                            'Pmax_AMPA': round(i*0.1, 3),
                                            'Pmax_NMDA': round(j*0.1, 3),
                                            'gtCa': 0,
                                            'esyn': 0,
                                            'Mg_conc': round(2.2+0*k, 3),
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
                                            'delay': 0,
                                            'buf': l}
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
                   "N{6}_P_AMPA{7}_P_NMDA{8}_Mg_conc{9}_gkCa{10}_buf{11}_HH".format(d.year,
                                                                                    d.month,
                                                                                    d.day,
                                                                                    d.hour,
                                                                                    d.minute,
                                                                                    d.second,
                                                                                    numneu,
                                                                                    self.neuron.Pmax_AMPA_base,
                                                                                    self.neuron.Pmax_NMDA_base,
                                                                                    self.neuron.Mg_conc,
                                                                                    self.neuron.gkCa,
                                                                                    self.neuron.buf)
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
                df['I_kCa_{} [uA]'.format(k)] = self.neuron.IkCa[k]
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
            self.neuron.INa = np.fliplr(self.neuron.INa)
            self.neuron.m = np.fliplr(self.neuron.m)
            self.neuron.h = np.fliplr(self.neuron.h)
            self.neuron.alpha_m = np.fliplr(self.neuron.alpha_m)
            self.neuron.alpha_h = np.fliplr(self.neuron.alpha_h)
            self.neuron.beta_m = np.fliplr(self.neuron.beta_m)
            self.neuron.beta_h = np.fliplr(self.neuron.beta_h)
            self.neuron.IpNa = np.fliplr(self.neuron.IpNa)
            self.neuron.pna = np.fliplr(self.neuron.pna)
            self.neuron.alpha_pna = np.fliplr(self.neuron.alpha_pna)
            self.neuron.beta_pna = np.fliplr(self.neuron.beta_pna)
            self.neuron.IK = np.fliplr(self.neuron.IK)
            self.neuron.n = np.fliplr(self.neuron.n)
            self.neuron.alpha_n = np.fliplr(self.neuron.alpha_n)
            self.neuron.beta_n = np.fliplr(self.neuron.beta_n)
            self.neuron.Ileak = np.fliplr(self.neuron.Ileak)
            self.neuron.Im = np.fliplr(self.neuron.Im)
            self.neuron.p = np.fliplr(self.neuron.p)
            self.neuron.p_inf = np.fliplr(self.neuron.p_inf)
            self.neuron.tau_p = np.fliplr(self.neuron.tau_p)
            self.neuron.ItCa = np.fliplr(self.neuron.ItCa)
            self.neuron.u = np.fliplr(self.neuron.u)
            self.neuron.s_inf = np.fliplr(self.neuron.s_inf)
            self.neuron.u_inf = np.fliplr(self.neuron.u_inf)
            self.neuron.tau_u = np.fliplr(self.neuron.tau_u)
            self.neuron.IlCa = np.fliplr(self.neuron.IlCa)
            self.neuron.q = np.fliplr(self.neuron.q)
            self.neuron.r = np.fliplr(self.neuron.r)
            self.neuron.alpha_q = np.fliplr(self.neuron.alpha_q)
            self.neuron.alpha_r = np.fliplr(self.neuron.alpha_r)
            self.neuron.beta_q = np.fliplr(self.neuron.beta_q)
            self.neuron.beta_r = np.fliplr(self.neuron.beta_r)
            self.neuron.IkCa = np.fliplr(self.neuron.IkCa)
            self.neuron.ca_influx = np.fliplr(self.neuron.ca_influx)
            self.neuron.Isyn = np.fliplr(self.neuron.Isyn)
            self.neuron.Isyn[:, 1:] = 0
            self.neuron.IAMPA = np.fliplr(self.neuron.IAMPA)
            self.neuron.IAMPA[:, 1:] = 0
            self.neuron.INMDA = np.fliplr(self.neuron.INMDA)
            self.neuron.INMDA[:, 1:] = 0
            self.neuron.R_AMPA = np.flip(self.neuron.R_AMPA, axis=2)
            self.neuron.R_NMDA = np.flip(self.neuron.R_NMDA, axis=2)
            self.neuron.E_AMPA = np.flip(self.neuron.E_AMPA, axis=2)
            self.neuron.E_NMDA = np.flip(self.neuron.E_NMDA, axis=2)
            self.neuron.I_AMPA = np.flip(self.neuron.I_AMPA, axis=2)
            self.neuron.I_NMDA = np.flip(self.neuron.I_NMDA, axis=2)
            self.neuron.Iext = np.fliplr(self.neuron.Iext)
            self.neuron.Iext[:, 1:] = 0
            self.neuron.t_fire_list = np.fliplr(self.neuron.t_fire_list)
            self.neuron.t_fire_list[:, 1:] = 0
            self.neuron.Inoise = np.fliplr(self.neuron.Inoise)
            self.neuron.dn = np.fliplr(self.neuron.dn)
            self.neuron.dWt = np.fliplr(self.neuron.dWt)
            self.neuron.curstep = 0
            self.lump_counter += 1
        ####### MAIN PROCESS END#######


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

    pic = Picture(save_path, process, numneu)
    pic.run()

    d = datetime.datetime.today()
    print("{0}/{1}/{2}/{3}:{4}:{5}".format(d.year, d.month, d.day, d.hour, d.minute, d.second))
    elapsed_time = time.time() - starttime
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")


if __name__ == '__main__':
     main()
