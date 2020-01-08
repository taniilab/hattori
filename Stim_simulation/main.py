"""
date: 20191213
created by: ishida
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class HH_compartment:
    def __init__(self, sim_time_ms, dt_ms, stim_timing_ms, stim_ext_potential):
        allsteps = int(sim_time_ms / dt_ms)
        self.m = 0.5 * np.ones(allsteps)
        self.h = 0.06 * np.ones(allsteps)
        self.n = 0.5 * np.ones(allsteps)
        self.p = 0.5 * np.ones(allsteps)
        self.u = 0.5 * np.ones(allsteps)

        self.alpha_m = 0 * np.ones(allsteps)
        self.beta_m = 0 * np.ones(allsteps)
        self.alpha_h = 0 * np.ones(allsteps)
        self.beta_h = 0 * np.ones(allsteps)
        self.alpha_n = 0 * np.ones(allsteps)
        self.beta_n = 0 * np.ones(allsteps)
        self.p_inf = 0 * np.ones(allsteps)
        self.tau_p = 0 * np.ones(allsteps)
        self.s_inf = 0 * np.ones(allsteps)
        self.u_inf = 0 * np.ones(allsteps)
        self.tau_u = 0 * np.ones(allsteps)

        """
        self.IKCa = np.zeros(allsteps)
        self.ca_conc = 0 * np.ones(allsteps)
        self.ca_step = 100  # [nM]
        self.tau_ca = 2700  # [ms]
        """

        self.Ileak = np.zeros(allsteps)
        self.INa = np.zeros(allsteps)
        self.IK = np.zeros(allsteps)
        self.Im = np.zeros(allsteps)
        self.ItCa = np.zeros(allsteps)
        self.intracellular_current = np.zeros(allsteps)

        self.V_intra = -74 * np.ones(allsteps)

        v_stim = PotentialStimulation(sim_time_ms, dt_ms)
        self.stim_timing_step, self.V_extra = v_stim.external_potential(stim_timing_ms, stim_ext_potential)


class Neuron:
    def __init__(self, sim_time_ms=180000, dt_ms=0.04, left_stim_timing_ms=[1000, 1020, 1040],
                 left_stim_ext_potential=[30, -30, 0]):
        self.sim_time_ms = sim_time_ms
        self.dt_ms = dt_ms
        self.allsteps = int(sim_time_ms / dt_ms)
        self.curstep = 0
        self.cell_left = HH_compartment(sim_time_ms=sim_time_ms,
                                        dt_ms=dt_ms,
                                        stim_timing_ms=left_stim_timing_ms,
                                        stim_ext_potential=left_stim_ext_potential)
        self.cell_right = HH_compartment(sim_time_ms=sim_time_ms,
                                         dt_ms=dt_ms,
                                         stim_timing_ms=[0],
                                         stim_ext_potential=[0])
        self.time_seq = np.arange(0, sim_time_ms, dt_ms)

    # activation function
    # a / (1 + exp(b * (x - c))
    def activation_func_sigmoid(self, a, b, c, v):
        return a / (1.0 + np.exp(np.clip(b * (v - c), -500, 500)))

    # a * exp(b * (v - c))
    def activation_func_exp(self, a, b, c, v):
        return a * np.exp(np.clip(b * (v - c), -500, 500))

    # a * (v - b) / (exp(c * (v - d)) - 1)
    def activation_func_ReLUlike(self, a, b, c, d, v):
        return a * (v - b) / (np.exp(np.clip(c * (v - d), -500, 500)) - 1)

    def calc_current(self, cell_compartment, Vth=-56.2, tau_max=608, gleak=0.0205, eleak=-70.3,
                     gNa=56, eNa=50, gK=6, eK=-90, gm=0.075, gtCa=0.4, eCa=120):
        cell_compartment.alpha_m[self.curstep] = self.activation_func_ReLUlike(-0.32, Vth + 13, -1 / 4, Vth + 13,
                                                                               cell_compartment.V_intra[self.curstep] -
                                                                               cell_compartment.V_extra[self.curstep])
        cell_compartment.beta_m[self.curstep] = self.activation_func_ReLUlike(0.28, Vth + 40, 1 / 5, Vth + 40,
                                                                              cell_compartment.V_intra[self.curstep] -
                                                                              cell_compartment.V_extra[self.curstep])
        cell_compartment.alpha_h[self.curstep] = self.activation_func_exp(0.128, -1 / 18, Vth + 17,
                                                                          cell_compartment.V_intra[self.curstep] -
                                                                          cell_compartment.V_extra[self.curstep])
        cell_compartment.beta_h[self.curstep] = self.activation_func_sigmoid(4, -1 / 5, Vth + 40,
                                                                             cell_compartment.V_intra[self.curstep] -
                                                                             cell_compartment.V_extra[self.curstep])
        cell_compartment.alpha_n[self.curstep] = self.activation_func_ReLUlike(-0.032, Vth + 15, -1 / 5, Vth + 15,
                                                                               cell_compartment.V_intra[self.curstep] -
                                                                               cell_compartment.V_extra[self.curstep])
        cell_compartment.beta_n[self.curstep] = self.activation_func_exp(0.5, -1 / 40, Vth + 10,
                                                                         cell_compartment.V_intra[self.curstep] -
                                                                         cell_compartment.V_extra[self.curstep])
        cell_compartment.p_inf[self.curstep] = self.activation_func_sigmoid(1, -1 / 10, -35,
                                                                            cell_compartment.V_intra[self.curstep] -
                                                                            cell_compartment.V_extra[self.curstep])
        cell_compartment.tau_p[self.curstep] = (tau_max / (
                3.3 * np.exp(
            np.clip((cell_compartment.V_intra[self.curstep] - cell_compartment.V_extra[self.curstep] + 35) / 20, -709,
                    10000)) + np.exp(
            - np.clip((cell_compartment.V_intra[self.curstep] - cell_compartment.V_extra[self.curstep] + 35) / 20, -709,
                      10000))))

        cell_compartment.s_inf[self.curstep] = self.activation_func_sigmoid(1, -1 / 6.2, -2 - 57,
                                                                            cell_compartment.V_intra[self.curstep] -
                                                                            cell_compartment.V_extra[self.curstep])
        cell_compartment.u_inf[self.curstep] = self.activation_func_sigmoid(1, 1 / 4, -2 - 81,
                                                                            cell_compartment.V_intra[self.curstep] -
                                                                            cell_compartment.V_extra[self.curstep])
        cell_compartment.tau_u[self.curstep] = 30.8 + (
                211.4 + np.exp(
            np.clip((cell_compartment.V_intra[self.curstep] - cell_compartment.V_extra[self.curstep] + 2 + 113.2) / 5,
                    -709, 10000))) / \
                                               (3.7 * (1 + np.exp(
                                                   np.clip((cell_compartment.V_intra[self.curstep] -
                                                            cell_compartment.V_extra[self.curstep] + 2 + 84) / 3.2,
                                                           -709,
                                                           10000))))

        dm = self.dt_ms * (cell_compartment.alpha_m[self.curstep] * (1 - cell_compartment.m[self.curstep])
                           - cell_compartment.beta_m[self.curstep] * cell_compartment.m[self.curstep])
        dh = self.dt_ms * (cell_compartment.alpha_h[self.curstep] * (1 - cell_compartment.h[self.curstep])
                           - cell_compartment.beta_h[self.curstep] * cell_compartment.h[self.curstep])
        dn = self.dt_ms * (cell_compartment.alpha_n[self.curstep] * (1 - cell_compartment.n[self.curstep])
                           - cell_compartment.beta_n[self.curstep] * cell_compartment.n[self.curstep])
        dp = self.dt_ms * (cell_compartment.p_inf[self.curstep] - cell_compartment.p[self.curstep]) / \
             cell_compartment.tau_p[self.curstep]
        du = self.dt_ms * (cell_compartment.u_inf[self.curstep] - cell_compartment.u[self.curstep]) / \
             cell_compartment.tau_u[self.curstep]

        cell_compartment.m[self.curstep + 1] = cell_compartment.m[self.curstep] + dm
        cell_compartment.h[self.curstep + 1] = cell_compartment.h[self.curstep] + dh
        cell_compartment.n[self.curstep + 1] = cell_compartment.n[self.curstep] + dn
        cell_compartment.p[self.curstep + 1] = cell_compartment.p[self.curstep] + dp
        cell_compartment.u[self.curstep + 1] = cell_compartment.u[self.curstep] + du

        cell_compartment.Ileak[self.curstep] = gleak * (
                    eleak - cell_compartment.V_intra[self.curstep] - cell_compartment.V_extra[self.curstep])
        cell_compartment.INa[self.curstep] = gNa * cell_compartment.m[self.curstep] ** 3 * cell_compartment.h[
            self.curstep] * (eNa - (cell_compartment.V_intra[self.curstep] - cell_compartment.V_extra[self.curstep]))
        cell_compartment.IK[self.curstep] = gK * cell_compartment.n[self.curstep] ** 4 * (
                eK - (cell_compartment.V_intra[self.curstep] - cell_compartment.V_extra[self.curstep]))
        cell_compartment.Im[self.curstep] = gm * cell_compartment.p[self.curstep] * (
                eK - (cell_compartment.V_intra[self.curstep] - cell_compartment.V_extra[self.curstep]))
        cell_compartment.ItCa[self.curstep] = gtCa * cell_compartment.s_inf[self.curstep] ** 2 * cell_compartment.u[
            self.curstep] * (eCa - (cell_compartment.V_intra[self.curstep] - cell_compartment.V_extra[self.curstep]))

    def calc_potential(self, cell_left, cell_right, R=10, Cm=1):
        self.calc_current(cell_left, Vth=-56.2, tau_max=608, gleak=0.0205, eleak=-70.3,
                     gNa=56, eNa=50, gK=6, eK=-90, gm=0.075, gtCa=0, eCa=120)
        if self.curstep * self.dt_ms > 500:
            cell_left.intracellular_current[self.curstep] = - (
                        cell_left.V_intra[self.curstep] - cell_right.V_intra[self.curstep]) / R
        dV_intra_left = (self.dt_ms * (
                cell_left.Ileak[self.curstep] + cell_left.INa[self.curstep] + cell_left.IK[self.curstep] +
                cell_left.Im[self.curstep] + cell_left.intracellular_current[self.curstep]) / Cm)
        cell_left.V_intra[self.curstep + 1] = cell_left.V_intra[self.curstep] + dV_intra_left

        self.calc_current(cell_right, Vth=-56.2, tau_max=608, gleak=0.0205, eleak=-70.3,
                     gNa=56, eNa=50, gK=6, eK=-90, gm=0.075, gtCa=0, eCa=120)
        if self.curstep * self.dt_ms > 500:
            cell_right.intracellular_current[self.curstep] = - (
                    cell_right.V_intra[self.curstep] - cell_left.V_intra[self.curstep]) / R

        dV_intra_right = (self.dt_ms * (
                cell_right.Ileak[self.curstep] + cell_right.INa[self.curstep] + cell_right.IK[self.curstep] +
                cell_right.Im[self.curstep] + cell_right.intracellular_current[self.curstep]) / Cm)
        cell_right.V_intra[self.curstep + 1] = cell_right.V_intra[self.curstep] + dV_intra_right


class CurrentStimulation:
    def __init__(self, sim_time_ms, dt_ms):
        self.dt_ms = dt_ms
        self.allsteps = int(sim_time_ms / dt_ms)
        self.Iext = np.zeros(self.allsteps)
        # self.arbitorary_current()
        self.RC_transient(stim_timing_ms=2000, V=5000, R=100, C=1)

    def arbitorary_current(self):
        self.Iext[int(2000/0.04):int(2100/0.04)] = 50

    def RC_transient(self, stim_timing_ms, V, R, C):
        stim_timing_step = int(stim_timing_ms / self.dt_ms)
        for step_i in range(stim_timing_step, self.allsteps):
            self.Iext[step_i] = (V / R) * np.exp(- (step_i - stim_timing_step) * self.dt_ms / (R * C))


class PotentialStimulation:
    def __init__(self, sim_time_ms, dt_ms):
        self.dt_ms = dt_ms
        self.allsteps = int(sim_time_ms / dt_ms)

    def external_potential(self, stim_timing_ms, potential_mv):
        V_extra = np.zeros(self.allsteps)
        stim_timing_step = np.zeros(len(stim_timing_ms))

        if len(stim_timing_ms) == 1:
            V_extra[int(stim_timing_ms[0] / self.dt_ms):] = potential_mv[0]
            return [int(stim_timing_ms[0] / self.dt_ms)], V_extra
        else:
            for stim_i in range(0, len(stim_timing_ms)):
                stim_timing_step[stim_i] = int(stim_timing_ms[stim_i] / self.dt_ms)
            for stim_i in range(0, len(stim_timing_step)-1):
                V_extra[int(stim_timing_step[stim_i]):int(stim_timing_step[stim_i + 1])] = potential_mv[stim_i]
            V_extra[int(stim_timing_step[len(stim_timing_step) - 1]):] = potential_mv[len(stim_timing_step) - 1]

            return stim_timing_step, V_extra


def main():
    left_stim_timing_ms = [500, 520, 540]
    left_stim_ext_potential = [-10, 10, 0]
    neuron = Neuron(sim_time_ms=1000, dt_ms=0.04, left_stim_timing_ms=left_stim_timing_ms,
                    left_stim_ext_potential=left_stim_ext_potential)
    R = 1  # [M ohm]

    for step_i in range(0, neuron.allsteps-1):
        neuron.calc_potential(neuron.cell_left, neuron.cell_right, R)
        neuron.curstep += 1

    df_left = pd.DataFrame({'time [ms]': neuron.time_seq,
                            'I_Na [pA]': neuron.cell_left.INa,
                            'I_leak [pA]': neuron.cell_left.Ileak,
                            'I_m [pA]': neuron.cell_left.Im,
                            'I_K [pA]': neuron.cell_left.IK,
                            'intracellular_current [pA]': neuron.cell_left.intracellular_current,
                            'V_intra [mV]': neuron.cell_left.V_intra,
                            'V_extra [mV]': neuron.cell_left.V_extra,
                            'V_membrane [mV]': neuron.cell_left.V_intra - neuron.cell_left.V_extra})
    df_right = pd.DataFrame({'time [ms]': neuron.time_seq,
                             'I_Na [pA]': neuron.cell_right.INa,
                             'I_leak [pA]': neuron.cell_right.Ileak,
                             'I_m [pA]': neuron.cell_right.Im,
                             'I_K [pA]': neuron.cell_right.IK,
                             'intracellular_current [pA]': neuron.cell_right.intracellular_current,
                             'V_intra [mV]': neuron.cell_right.V_intra,
                             'V_extra [mV]': neuron.cell_right.V_extra,
                             'V_membrane [mV]': neuron.cell_right.V_intra - neuron.cell_right.V_extra})
    df_left.to_csv('C:/Users/Tanii_Lab/Downloads/cell_left_{0}_{1}.csv'.format(left_stim_ext_potential[0],
                                                                                          left_stim_ext_potential[1]))
    df_right.to_csv('C:/Users/Tanii_Lab/Downloads/cell_right_{0}_{1}.csv'.format(left_stim_ext_potential[0],
                                                                                            left_stim_ext_potential[1]))

    fsize = 15
    fig = plt.figure(figsize=(20, 8))
    ax1 = fig.add_subplot(111)
    ax1.plot(neuron.time_seq / 1000, neuron.cell_left.V_intra, label='cell_left', color='darkcyan')
    ax1.plot(neuron.time_seq / 1000, neuron.cell_right.V_intra, label='cell_right', color='darkgreen')
    #ax1.set_xlim([0.95, 2.0])
    ax1.set_xlabel('time [sec]', fontsize=fsize)
    ax1.set_ylabel('potential [mV]', fontsize=fsize)
    ax1.legend(fontsize=fsize)
    ax2 = ax1.twinx()
    ax2.plot(neuron.time_seq / 1000, neuron.cell_left.V_extra, label='cell_right', color='crimson')
    plt.tick_params(labelsize=fsize)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
