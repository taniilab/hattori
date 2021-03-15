"""
date: 20191211
created by: ishida
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class HH_compartment:
    def __init__(self, allsteps):
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

        self.Ileak = np.zeros(allsteps)
        self.INa = np.zeros(allsteps)
        self.IK = np.zeros(allsteps)
        self.Im = np.zeros(allsteps)
        self.ItCa = np.zeros(allsteps)
        self.intracellular_current = np.zeros(allsteps)

        self.V = -74 * np.ones(allsteps)


class Neuron:
    def __init__(self, sim_time_ms=180000, dt_ms=0.04):
        self.sim_time_ms = sim_time_ms
        self.dt_ms = dt_ms
        self.allsteps = int(sim_time_ms / dt_ms)
        self.curstep = 0
        self.cell_left = HH_compartment(self.allsteps)
        self.cell_right = HH_compartment(self.allsteps)
        self.time_seq = np.arange(0, sim_time_ms, dt_ms)
        stim = Stimulation(sim_time_ms, dt_ms)
        self.stim_current = stim.Iext

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
                                                                               cell_compartment.V[self.curstep])
        cell_compartment.beta_m[self.curstep] = self.activation_func_ReLUlike(0.28, Vth + 40, 1 / 5, Vth + 40,
                                                                              cell_compartment.V[self.curstep])
        cell_compartment.alpha_h[self.curstep] = self.activation_func_exp(0.128, -1 / 18, Vth + 17,
                                                                          cell_compartment.V[self.curstep])
        cell_compartment.beta_h[self.curstep] = self.activation_func_sigmoid(4, -1 / 5, Vth + 40,
                                                                             cell_compartment.V[self.curstep])
        cell_compartment.alpha_n[self.curstep] = self.activation_func_ReLUlike(-0.032, Vth + 15, -1 / 5, Vth + 15,
                                                                               cell_compartment.V[self.curstep])
        cell_compartment.beta_n[self.curstep] = self.activation_func_exp(0.5, -1 / 40, Vth + 10,
                                                                         cell_compartment.V[self.curstep])
        cell_compartment.p_inf[self.curstep] = self.activation_func_sigmoid(1, -1 / 10, -35,
                                                                            cell_compartment.V[self.curstep])
        cell_compartment.tau_p[self.curstep] = (tau_max / (
                3.3 * np.exp(np.clip((cell_compartment.V[self.curstep] + 35) / 20, -709, 10000)) + np.exp(
            - np.clip((cell_compartment.V[self.curstep] + 35) / 20, -709, 10000))))

        cell_compartment.s_inf[self.curstep] = self.activation_func_sigmoid(1, -1 / 6.2, -2 - 57,
                                                                            cell_compartment.V[self.curstep])
        cell_compartment.u_inf[self.curstep] = self.activation_func_sigmoid(1, 1 / 4, -2 - 81,
                                                                            cell_compartment.V[self.curstep])
        cell_compartment.tau_u[self.curstep] = 30.8 + (
                211.4 + np.exp(np.clip((cell_compartment.V[self.curstep] + 2 + 113.2) / 5, -709, 10000))) / \
                                               (3.7 * (1 + np.exp(
                                                   np.clip((cell_compartment.V[self.curstep] + 2 + 84) / 3.2, -709,
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

        cell_compartment.Ileak[self.curstep] = gleak * (eleak - cell_compartment.V[self.curstep])
        cell_compartment.INa[self.curstep] = gNa * cell_compartment.m[self.curstep] ** 3 * cell_compartment.h[
            self.curstep] * (eNa - cell_compartment.V[self.curstep])
        cell_compartment.IK[self.curstep] = gK * cell_compartment.n[self.curstep] ** 4 * (
                    eK - cell_compartment.V[self.curstep])
        cell_compartment.Im[self.curstep] = gm * cell_compartment.p[self.curstep] * (
                    eK - cell_compartment.V[self.curstep])
        cell_compartment.ItCa[self.curstep] = gtCa * cell_compartment.s_inf[self.curstep] ** 2 * cell_compartment.u[
            self.curstep] * (eCa - cell_compartment.V[self.curstep])

    def calc_potential(self, cell_left, cell_right, R=10, Cm=1):
        self.calc_current(cell_left)
        self.calc_current(cell_right)
        if self.curstep * self.dt_ms > 1000:
            cell_left.intracellular_current[self.curstep] = - (cell_left.V[self.curstep] - cell_right.V[self.curstep]) / R
            cell_right.intracellular_current[self.curstep] = - (cell_right.V[self.curstep] - cell_left.V[self.curstep]) / R
        dV_left = (self.dt_ms * (
                cell_left.Ileak[self.curstep] + cell_left.INa[self.curstep] + cell_left.IK[self.curstep] +
                cell_left.Im[self.curstep] + self.stim_current[self.curstep] + cell_left.intracellular_current[
                    self.curstep]) / Cm)
        dV_right = (self.dt_ms * (
                cell_right.Ileak[self.curstep] + cell_right.INa[self.curstep] + cell_right.IK[self.curstep] +
                cell_right.Im[self.curstep] - self.stim_current[self.curstep] + cell_right.intracellular_current[
                    self.curstep]) / Cm)
        cell_left.V[self.curstep + 1] = cell_left.V[self.curstep] + dV_left
        cell_right.V[self.curstep + 1] = cell_right.V[self.curstep] + dV_right


class Stimulation:
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


def main():
    neuron = Neuron(sim_time_ms=5000, dt_ms=0.04)
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
                            'V [mV]': neuron.cell_left.V,
                            'I_ext [pA]': neuron.stim_current})
    df_right = pd.DataFrame({'time [ms]': neuron.time_seq,
                             'I_Na [pA]': neuron.cell_right.INa,
                             'I_leak [pA]': neuron.cell_right.Ileak,
                             'I_m [pA]': neuron.cell_right.Im,
                             'I_K [pA]': neuron.cell_right.IK,
                             'intracellular_current [pA]': neuron.cell_right.intracellular_current,
                             'V [mV]': neuron.cell_right.V,
                             'I_ext [pA]': - neuron.stim_current})
    df_left.to_csv('C:/Users/ishida/Desktop/cell_left.csv')
    df_right.to_csv('C:/Users/ishida/Desktop/cell_right.csv')

    fsize = 15
    fig = plt.figure(figsize=(12, 12))
    ax1 = fig.add_subplot(211)
    ax1.plot(neuron.time_seq / 1000, neuron.cell_left.V, label='cell_left', color='darkcyan')
    ax1.set_xlabel('time [sec]', fontsize=fsize)
    ax1.set_ylabel('potential [mV]', fontsize=fsize)
    ax1_2 = ax1.twinx()
    ax1_2.plot(neuron.time_seq / 1000, neuron.stim_current, label='stim_current', color='purple', alpha=0.8)
    ax1_2.set_ylabel('intracellular current [pA]', fontsize=fsize)
    """
    ax1_2.plot(neuron.time_seq / 1000, neuron.cell_left.Ileak, label='I_leak')
    ax1_2.plot(neuron.time_seq / 1000, neuron.cell_left.INa, label='I_Na')
    ax1_2.plot(neuron.time_seq / 1000, neuron.cell_left.IK, label='I_K')
    ax1_2.plot(neuron.time_seq / 1000, neuron.cell_left.Im, label='I_m')
    ax1_2.plot(neuron.time_seq / 1000, -(neuron.cell_left.V - neuron.cell_right.V) / R)
    """
    ax1.legend(fontsize=fsize)
    ax1_2.legend(fontsize=fsize)
    plt.tick_params(labelsize=fsize)
    plt.tight_layout()

    ax2 = fig.add_subplot(212)
    ax2.plot(neuron.time_seq / 1000, neuron.cell_right.V, label='cell_right', color='darkgreen')
    ax2.set_xlabel('time [sec]', fontsize=fsize)
    ax2.set_ylabel('potential [mV]', fontsize=fsize)
    ax2.legend(fontsize=fsize)
    ax2_2 = ax2.twinx()
    ax2_2.plot(neuron.time_seq / 1000, - neuron.stim_current, label='stim_current', color='purple', alpha=0.8)
    ax2_2.set_ylabel('intracellular current [pA]', fontsize=fsize)
    """
    ax2_2.plot(neuron.time_seq / 1000, neuron.cell_right.Ileak, label='I_leak')
    ax2_2.plot(neuron.time_seq / 1000, neuron.cell_right.INa, label='I_Na')
    ax2_2.plot(neuron.time_seq / 1000, neuron.cell_right.IK, label='I_K')
    ax2_2.plot(neuron.time_seq / 1000, neuron.cell_right.Im, label='I_m')
    ax2_2.plot(neuron.time_seq / 1000, -(neuron.cell_right.V - neuron.cell_left.V) / R)
    """
    ax2_2.legend(fontsize=fsize)

    plt.tick_params(labelsize=fsize)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
