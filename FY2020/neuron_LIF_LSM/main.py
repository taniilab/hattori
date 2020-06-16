"""
***Unit of parameters***
Brunel & Wang 2001 integrate and fire neuron
membrane potential -> mV
time -> ms
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
import numpy as np
from lsm import LSM
import matplotlib.pyplot as plt
import subprocess
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec


starttime = time.time()
elapsed_time = 0
save_path = "Z:/simulation/test"
process = 6 #number of processors

#parameters#
numneu = 1
simtime = 1000
lump = 500
num_lump = int(simtime/lump)
dt = 0.04

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
                                            'Cm': 0.5e-3,
                                            'G_L': 25e-6,
                                            'Vreset': -80,
                                            'Vth': -50,
                                            'erest': -70,
                                            #'Iext_amp': round(2e-4+3e-4*i, 6),
                                            'Iext_amp': 1e-3,
                                            'syn_type': 3,
                                            'Pmax_AMPA': round(0.00005+i*0.00005, 8),
                                            #'Pmax_AMPA': 0.00003,
                                            'Pmax_NMDA': round(0.0001+j*0.0001, 8),
                                            #'Pmax_NMDA': 0.00005,
                                            #'Pmax_NMDA': 0,
                                            'tau_syn': 5.26,
                                            'noise_type': 1,
                                            'D': 0}
            self.parm_counter += 1
            self.overall_steps = int(self.i*self.j*self.k*self.l*simtime/(dt*process))


    def input_generator_sin(self):
        # sin wave
        t = np.arange(self.lump_counter * lump, (self.lump_counter + 1) * lump + dt, dt)
        self.neuron.Iext[0, :] =  np.sin(t * 0.02)+1
        #self.neuron.Iext[3, :] = np.sin(t * 0.02) + 1
        if self.lump_counter == 0:
            self.neuron.Iext[0, :2500] = 0
            #self.neuron.Iext[3, :2500] = 0

    def input_generator_mackey_glass(self, beta=2, gamma=1, tau=2, n=9.65, expand=False):
        index = np.arange(1+simtime/dt) #including buffer
        x = index * 0 + 0.5
        tau = int(tau / dt)
        for i in range(tau, len(index) - 1):
            x[i + 1] = x[i] + dt * (beta * x[i - tau] / (1 + x[i - tau] ** n) - gamma * x[i])

        self.neuron.Iext[0, :] = x[int(self.lump_counter * lump/dt):
                                            int(((self.lump_counter + 1) * lump/dt)+1)]

        #scaling
        if expand == True:
            x_expand = np.ones(len(x)*2)*0 + 0.5
            for i in range(len(x)):
                x_expand[2 * i] = x[i]
            for i in range(len(x) - 2):
                x_expand[2 * i + 1] = (x_expand[2 * i + 2] + x_expand[2 * i]) / 2
            self.neuron.Iext[0, :] = x_expand[int(self.lump_counter * lump/dt):
                                              int(((self.lump_counter + 1) * lump/dt)+1)]

        if self.lump_counter == 0:
            self.neuron.Iext[0, :2500] = 0


    def simulate(self, process):
        # parallel processing on each setting value
        self.pid = os.getpid()
        self.neuron = Neuron(**self.parm[process+self.process_counter])
        self.neuron.parm_dict = self.parm[process+self.process_counter]
        self.progress_counter = self.now_cycle_multiproc*self.neuron.allsteps

        # record
        d = datetime.datetime.today()
        filename = "{0}_{1}_{2}_{3}_{4}_{5}_" \
                   "Iext_amp{6}_Pmax_AMPA{7}_Pmax_NMDA{8}_LIF".format(d.year,
                                                                          d.month,
                                                                          d.day,
                                                                          d.hour,
                                                                          d.minute,
                                                                          d.second,
                                                                          self.neuron.Iext_amp,
                                                                          self.neuron.Pmax_AMPA,
                                                                          self.neuron.Pmax_NMDA)
        df = pd.DataFrame(columns=[filename])
        df.to_csv(save_path + '/' + filename + '.csv')
        df = pd.DataFrame()
        for k in range(numneu):
            df['T_{} [ms]'.format(k)] = ""
            df['V_{} [mV]'.format(k)] = ""
            df['fire_{}'.format(k)] = ""
            df['I_syn_{} [uA]'.format(k)] = ""
            df['I_AMPA_{} [uA]'.format(k)] = ""
            df['I_NMDA_{} [uA]'.format(k)] = ""
            df['Iext_{} [uA]'.format(k)] = ""
            df['I_noise_{} [uA]'.format(k)] = ""

        df.to_csv(save_path + '/' + filename + '.csv', mode='a')

        ####### MAIN PROCESS #######
        for j in range(num_lump):
            self.input_generator_sin()
            #self.input_generator_mackey_glass()

            ####### MAIN CYCLE #######
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
                df['fire_{}'.format(k)] = self.neuron.t_fire_list[k]
                df['I_syn_{} [uA]'.format(k)] = self.neuron.Isyn[k]
                df['I_AMPA_{} [uA]'.format(k)] = self.neuron.IAMPA[k]
                df['I_NMDA_{} [uA]'.format(k)] = self.neuron.INMDA[k]
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


        # Visualization of connection structure
        # graphbiz must be installed
        if not os.path.isdir(save_path + '/dot'):
            os.mkdir(save_path + '/dot')
        if not os.path.isdir(save_path + '/structure'):
            os.mkdir(save_path + '/structure')
        dot_txt = 'digraph g{\n'
        dot_txt += 'graph [ dpi = 300, ratio = 1.0];\n'
        for i in range(numneu):
            dot_txt += '{} [label="{}", color=lightseagreen, fontcolor=white, style=filled]\n'.format(i, 'N'+str(i+1))
        for i, j in itertools.product(range(numneu), range(numneu)):
            if self.neuron.Syn_weight[i, j] != 0:
                dot_txt += '{}->{}\n'.format(i, j)
        dot_txt += "}\n"

        with open(save_path + '/dot/' + filename + '.dot', 'w') as f:
            f.write(dot_txt)
        self.cmd = 'dot {} -T png -o {}'.format(save_path + '/dot/' + filename + '.dot', save_path + '/structure/' + filename + '.png')
        subprocess.run(self.cmd, shell=True)


        # Rastergram
        plt.rcParams["font.size"] = 28
        if not os.path.isdir(save_path + '/rastergram'):
            os.mkdir(save_path + '/rastergram')
        num_read_nodes = numneu
        raster_line_length = 1
        raster_line_width = 0.5
        read_cols = ['T_0 [ms]']
        ytick_list = []
        for i in range(num_read_nodes):
            ytick_list.append(i + 1)
            read_cols.append('fire_{}'.format(i))

        df = pd.read_csv(save_path + '/' + filename + '.csv', usecols=read_cols, skiprows=1)[read_cols]
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111)
        ax.set_ylim(0, num_read_nodes + 1)
        ax.set_yticks(ytick_list)
        ax.set_xlabel("Time [ms]")
        ax.set_ylabel("Neuron number")
        for i in range(num_read_nodes):
            for j in range(len(df.values[:, 0])):
                if df.values[j, i + 1] != 0:
                    x = df.values[j, 0]
                    ax.plot([x, x], [i + 1 - (raster_line_length / 2), i + 1 + (raster_line_length / 2)],
                            linestyle="solid",
                            linewidth=raster_line_width,
                            color="black")
        plt.tight_layout()
        plt.savefig(save_path + '/rastergram/' + filename + '.png')
        plt.close(fig)


        ###### LEARNING AND PREDICTION PROCESS ######
        plt.rcParams["font.size"] = 14
        if not os.path.isdir(save_path + '/RC'):
            os.mkdir(save_path + '/RC')
        num_read_nodes = numneu
        read_cols = ['T_0 [ms]']
        for i in range(num_read_nodes):
            read_cols.append('V_{} [mV]'.format(i))
            read_cols.append('I_syn_{} [uA]'.format(i))
            read_cols.append('I_AMPA_{} [uA]'.format(i))
            read_cols.append('I_NMDA_{} [uA]'.format(i))
        read_cols.append('Iext_{} [uA]'.format(0))
        print(read_cols)

        df = pd.read_csv(save_path + '/' + filename + '.csv', usecols=read_cols, skiprows=1)[read_cols]
        train_ratio = 0.5
        border = int(len(df.values[:, 0]) * train_ratio)

        # time
        times = df.values[:, 0].reshape((len(df.values[:, 0]), 1))
        times_bef = df.values[:border, 0].reshape((len(df.values[:border, 0]), 1))
        times_af = df.values[border:, 0].reshape((len(df.values[border:, 0]), 1))

        # Iext
        index_tmp = []
        index_tmp.append(int(4 * num_read_nodes + 1))
        input = df.values[:, index_tmp].reshape((len(df.values[:, index_tmp]), len(index_tmp)))
        target = input[:border]

        # V
        index_tmp = []
        for i in range(num_read_nodes):
            index_tmp.append(i * 4 + 1)
        output = df.values[:, index_tmp].reshape((len(df.values[:, index_tmp]), len(index_tmp)))
        output_train = df.values[:border, index_tmp].reshape((len(df.values[:border, index_tmp]), len(index_tmp)))
        output_predict = df.values[border:, index_tmp].reshape((len(df.values[border:, index_tmp]), len(index_tmp)))

        # Isyn, Iampa, Inmda
        index_tmp = []
        for i in range(num_read_nodes):
            index_tmp.append(i * 4 + 2)
        Isyn = df.values[:, index_tmp].reshape((len(df.values[:, index_tmp]), len(index_tmp)))
        index_tmp = []
        for i in range(num_read_nodes):
            index_tmp.append(i * 4 + 3)
        IAMPA = df.values[:, index_tmp].reshape((len(df.values[:, index_tmp]), len(index_tmp)))
        index_tmp = []
        for i in range(num_read_nodes):
            index_tmp.append(i * 4 + 4)
        INMDA = df.values[:, index_tmp].reshape((len(df.values[:, index_tmp]), len(index_tmp)))

        lsm = LSM()
        lsm.train(output_train, target)
        predict_res = (output_predict @ lsm.output_w).T

        # layout
        fig = plt.figure(figsize=(20, 15))
        fig.suptitle(filename)
        fig.subplots_adjust(left=0.075, bottom=0.05, right=0.95, top=0.95, wspace=0.15, hspace=0.15)
        gs_master = GridSpec(nrows=num_read_nodes + 1, ncols=2)
        gs_rc = GridSpecFromSubplotSpec(nrows=1, ncols=2, subplot_spec=gs_master[0, 0:2])
        ax_rc = fig.add_subplot(gs_rc[:, :])
        gs_status = GridSpecFromSubplotSpec(nrows=num_read_nodes, ncols=2, subplot_spec=gs_master[1:, :], hspace=0.4,
                                            wspace=0.15)
        ax_status_v = []
        ax_status_i = []

        # Firing pattern of individual neurons
        for i in range(num_read_nodes):
            ax_status_v.append(fig.add_subplot(gs_status[i, 0]))
            ax_status_i.append(fig.add_subplot(gs_status[i, 1]))
            if i == 0:
                ax_rc.plot(times_bef, output_train[:, i], label="train_output_n{}".format(i))
                ax_rc.plot(times, input[:, 0], label="input(target)_Iext0")
                ax_rc.plot(times_af, predict_res[0], label="after training")
            ax_status_v[i].plot(times, output[:, i], label="output_n{}".format(i))
            ax_status_i[i].plot(times, Isyn[:, i], label="Isyn")
            ax_status_i[i].plot(times, IAMPA[:, i], label="IAMPA")
            ax_status_i[i].plot(times, INMDA[:, i], label="INMDA")
            ax_status_v[i].legend()
            ax_status_i[i].legend()

        print(times.shape)
        print(output_train.shape)
        print(target.shape)
        print(lsm.output_w.shape)
        print((output_train @ lsm.output_w).shape)
        print(output_predict.shape)
        print("W:{}".format(lsm.output_w))
        #plt.show()
        plt.savefig(save_path + '/RC/' + filename + '.png')
        plt.close(fig)
        ###### LEARNING AND PREDICTION PROCESS END######


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