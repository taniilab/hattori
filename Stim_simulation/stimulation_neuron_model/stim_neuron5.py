"""
date: 20200122
created by: ishida
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class HH_compartment:
    def __init__(self, sim_time_ms, dt_ms, Cm=1):
        allsteps = int(sim_time_ms / dt_ms)
        self.Cm = Cm  # [micro F / cm^2]
        self.m = 0.00045 * np.ones(allsteps)
        self.h = 0.999 * np.ones(allsteps)
        self.n = 0.00222 * np.ones(allsteps)
        self.p = 0.0243 * np.ones(allsteps)
        self.u = 0.0588 * np.ones(allsteps)

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
        self.gleak = np.zeros(allsteps)
        self.gNa = np.zeros(allsteps)
        self.gK = np.zeros(allsteps)
        self.gm = np.zeros(allsteps)
        self.Ileak = np.zeros(allsteps)
        self.INa = np.zeros(allsteps)
        self.IK = np.zeros(allsteps)
        self.Im = np.zeros(allsteps)
        self.ItCa = np.zeros(allsteps)

        self.V_m = -72 * np.ones(allsteps)


class Neuron:
    def __init__(self, sim_time_ms=180000, dt_ms=0.04):
        self.sim_time_ms = sim_time_ms
        self.dt_ms = dt_ms
        self.allsteps = int(sim_time_ms / dt_ms)
        self.curstep = 0
        self.cell_left = HH_compartment(sim_time_ms=sim_time_ms,
                                        dt_ms=dt_ms)
        self.cell_right = HH_compartment(sim_time_ms=sim_time_ms,
                                         dt_ms=dt_ms)

        self.V_left_extra = np.zeros(self.allsteps)
        self.V_left_intra = -74 * np.ones(self.allsteps)
        self.V_right_extra = np.zeros(self.allsteps)
        self.V_right_intra = -74 * np.ones(self.allsteps)

        self.V_left_extra_ext = np.zeros(self.allsteps)
        self.V_left_extra_left = np.zeros(self.allsteps)
        self.V_left_extra_right = np.zeros(self.allsteps)
        self.V_left_intra_ext = np.zeros(self.allsteps)
        self.V_left_intra_left = np.zeros(self.allsteps)
        self.V_left_intra_right = np.zeros(self.allsteps)
        self.V_right_extra_ext = np.zeros(self.allsteps)
        self.V_right_extra_left = np.zeros(self.allsteps)
        self.V_right_extra_right = np.zeros(self.allsteps)
        self.V_right_intra_ext = np.zeros(self.allsteps)
        self.V_right_intra_left = np.zeros(self.allsteps)
        self.V_right_intra_right = np.zeros(self.allsteps)

        self.I_total_left = np.zeros(self.allsteps)
        self.I_total_right = np.zeros(self.allsteps)

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
                                                                               cell_compartment.V_m[self.curstep])
        cell_compartment.beta_m[self.curstep] = self.activation_func_ReLUlike(0.28, Vth + 40, 1 / 5, Vth + 40,
                                                                              cell_compartment.V_m[self.curstep])
        cell_compartment.alpha_h[self.curstep] = self.activation_func_exp(0.128, -1 / 18, Vth + 17,
                                                                          cell_compartment.V_m[self.curstep])
        cell_compartment.beta_h[self.curstep] = self.activation_func_sigmoid(4, -1 / 5, Vth + 40,
                                                                             cell_compartment.V_m[self.curstep])
        cell_compartment.alpha_n[self.curstep] = self.activation_func_ReLUlike(-0.032, Vth + 15, -1 / 5, Vth + 15,
                                                                               cell_compartment.V_m[self.curstep])
        cell_compartment.beta_n[self.curstep] = self.activation_func_exp(0.5, -1 / 40, Vth + 10,
                                                                         cell_compartment.V_m[self.curstep])
        cell_compartment.p_inf[self.curstep] = self.activation_func_sigmoid(1, -1 / 10, -35,
                                                                            cell_compartment.V_m[self.curstep])
        cell_compartment.tau_p[self.curstep] = (tau_max / (
                3.3 * np.exp(
            np.clip((cell_compartment.V_m[self.curstep] + 35) / 20, -709,
                    10000)) + np.exp(
            - np.clip((cell_compartment.V_m[self.curstep] + 35) / 20, -709,
                      10000))))

        cell_compartment.s_inf[self.curstep] = self.activation_func_sigmoid(1, -1 / 6.2, -2 - 57,
                                                                            cell_compartment.V_m[self.curstep])
        cell_compartment.u_inf[self.curstep] = self.activation_func_sigmoid(1, 1 / 4, -2 - 81,
                                                                            cell_compartment.V_m[self.curstep])
        cell_compartment.tau_u[self.curstep] = 30.8 + (
                211.4 + np.exp(np.clip((cell_compartment.V_m[self.curstep] + 2 + 113.2) / 5, -709, 10000))) / (3.7 * (
                    1 + np.exp(np.clip((cell_compartment.V_m[self.curstep] + 2 + 84) / 3.2, -709, 10000))))

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

        cell_compartment.gleak[self.curstep] = gleak
        cell_compartment.gNa[self.curstep] = gNa * cell_compartment.m[self.curstep] ** 3 * cell_compartment.h[
            self.curstep]
        cell_compartment.gK[self.curstep] = gK * cell_compartment.n[self.curstep] ** 4
        cell_compartment.gm[self.curstep] = gm * cell_compartment.p[self.curstep]

        cell_compartment.Ileak[self.curstep] = gleak * (
                eleak - cell_compartment.V_m[self.curstep])
        cell_compartment.INa[self.curstep] = cell_compartment.gNa[self.curstep] * (
                eNa - (cell_compartment.V_m[self.curstep]))
        cell_compartment.IK[self.curstep] = cell_compartment.gK[self.curstep] * (
                eK - (cell_compartment.V_m[self.curstep]))
        cell_compartment.Im[self.curstep] = cell_compartment.gm[self.curstep] * (
                eK - (cell_compartment.V_m[self.curstep]))
        cell_compartment.ItCa[self.curstep] = gtCa * cell_compartment.s_inf[self.curstep] ** 2 * cell_compartment.u[
            self.curstep] * (eCa - (cell_compartment.V_m[self.curstep]))

    def calc_membrane_impedance(self, cell_component, omega):
        return (1 / (1j * omega * cell_component.Cm + cell_component.gleak[self.curstep] + cell_component.gNa[
            self.curstep] + cell_component.gK[self.curstep] + cell_component.gm[self.curstep])).real

    def calc_potential(self, cell_left, cell_right, ext_E, R1=10, R2=10, R3=10):
        self.calc_current(cell_left)
        self.calc_current(cell_right)
        # Z_left = self.calc_membrane_impedance(self.cell_left, 2 * np.pi)
        Z_left = self.calc_membrane_impedance(self.cell_left, 0)
        I_total_left = cell_left.Ileak[self.curstep] + cell_left.INa[self.curstep] + cell_left.IK[
            self.curstep] + cell_left.Im[self.curstep]
        Z_right = self.calc_membrane_impedance(cell_right, 2 * np.pi)
        I_total_right = cell_right.Ileak[self.curstep] + cell_right.INa[self.curstep] + cell_right.IK[
            self.curstep] + cell_right.Im[self.curstep]
        sum_R = R1 + R2 + R3 + Z_left + Z_right
        self.I_total_left[self.curstep] = I_total_left
        self.I_total_right[self.curstep] = I_total_right

        self.V_left_extra_ext[self.curstep + 1] = ext_E[self.curstep] * (Z_left + R3 + Z_right + R2) / sum_R
        self.V_left_extra_left[self.curstep + 1] = - I_total_left * R1 * Z_left / sum_R 
        self.V_left_extra_right[self.curstep + 1] = - I_total_right * R1 * Z_right / sum_R

        self.V_right_extra_ext[self.curstep + 1] = ext_E[self.curstep] * R2 / sum_R
        self.V_right_extra_left[self.curstep + 1] = I_total_left * R2 * Z_left / sum_R
        self.V_right_extra_right[self.curstep + 1] = I_total_right * R2 * Z_right / sum_R

        self.V_left_intra_ext[self.curstep + 1] = ext_E[self.curstep] * (R3 + Z_right + R2) / sum_R
        self.V_left_intra_left[self.curstep + 1] = I_total_left * (R2 + Z_right + R3) * Z_left / sum_R
        self.V_left_intra_right[self.curstep + 1] = - I_total_right * (R1 + Z_left) * Z_right / sum_R

        self.V_right_intra_ext[self.curstep + 1] = ext_E[self.curstep] * (Z_right + R2) / sum_R
        self.V_right_intra_left[self.curstep + 1] = I_total_left * (R2 + Z_right) * Z_left / sum_R
        self.V_right_intra_right[self.curstep + 1] = - I_total_right * (R1 + Z_left + R3) * Z_right / sum_R

        self.V_left_extra[self.curstep + 1] = ext_E[self.curstep] * (Z_left + R3 + Z_right + R2) / sum_R \
                                          - I_total_left * R1 * Z_left / sum_R - I_total_right * R1 * Z_right / sum_R
        self.V_right_extra[self.curstep + 1] = ext_E[self.curstep] * R2 / sum_R + I_total_left * R2 * Z_left / sum_R \
                                          + I_total_right * R2 * Z_right / sum_R
        self.V_left_intra[self.curstep + 1] = ext_E[self.curstep] * (R3 + Z_right + R2) / sum_R + I_total_left * (
                    R2 + Z_right + R3) * Z_left / sum_R - I_total_right * (R1 + Z_left) * Z_right / sum_R
        self.V_right_intra[self.curstep + 1] = ext_E[self.curstep] * (Z_right + R2) / sum_R + I_total_left * (
                    R2 + Z_right) * Z_left / sum_R - I_total_right * (R1 + Z_left + R3) * Z_right / sum_R

        cell_left.V_m[self.curstep + 1] = self.V_left_intra[self.curstep + 1] - self.V_left_extra[self.curstep + 1]
        cell_right.V_m[self.curstep + 1] = self.V_right_intra[self.curstep + 1] - self.V_right_extra[self.curstep + 1]

        self.curstep += 1

def main():
    neuron = Neuron(sim_time_ms=5000, dt_ms=0.04)
    # ext_E = np.sin(np.arange(0, 5000, 0.04) * 2 * np.pi / 1000) * 5
    # ext_E[0:int(1000/0.04)] = 0
    ext_E = np.zeros(neuron.allsteps)

    for step_i in range(0, neuron.allsteps-1):
        neuron.calc_potential(neuron.cell_left, neuron.cell_right, ext_E, R1=1000, R2=1000, R3=1000)

    df_left = pd.DataFrame({'time [ms]': neuron.time_seq,
                            'I_Na [pA]': neuron.cell_left.INa,
                            'I_leak [pA]': neuron.cell_left.Ileak,
                            'I_m [pA]': neuron.cell_left.Im,
                            'I_K [pA]': neuron.cell_left.IK,
                            'V_intra [mV]': neuron.V_left_intra,
                            'V_extra [mV]': neuron.V_left_extra,
                            'V_membrane [mV]': neuron.cell_left.V_m})
    df_right = pd.DataFrame({'time [ms]': neuron.time_seq,
                             'I_Na [pA]': neuron.cell_right.INa,
                             'I_leak [pA]': neuron.cell_right.Ileak,
                             'I_m [pA]': neuron.cell_right.Im,
                             'I_K [pA]': neuron.cell_right.IK,
                             'V_intra [mV]': neuron.V_right_intra,
                             'V_extra [mV]': neuron.V_right_extra,
                             'V_membrane [mV]': neuron.cell_right.V_m})
    df_left.to_csv('G:/Box Sync/Personal/xxx/cell_left.csv')
    df_right.to_csv('G:/Box Sync/Personal/xxx/cell_right.csv')
    df_potential = pd.DataFrame({'V_left_extra_ext': neuron.V_left_extra_ext,
                                 'V_left_extra_left': neuron.V_left_extra_left,
                                 'V_left_extra_right': neuron.V_left_extra_right,
                                 'V_left_intra_ext': neuron.V_left_intra_ext,
                                 'V_left_intra_left': neuron.V_left_intra_left,
                                 'V_left_intra_right': neuron.V_left_intra_right,
                                 'I_total_left': neuron.I_total_left,
                                 'V_right_extra_ext': neuron.V_right_extra_ext,
                                 'V_right_extra_left': neuron.V_right_extra_left,
                                 'V_right_extra_right': neuron.V_right_extra_right,
                                 'V_right_intra_ext': neuron.V_right_intra_ext,
                                 'V_right_intra_left': neuron.V_right_intra_left,
                                 'V_right_intra_right': neuron.V_right_intra_right,
                                 'I_total_right': neuron.I_total_right})
    df_potential.to_csv(
        'G:/Box Sync/Personal/xxx/cell_potential.csv')


    fsize = 15
    fig = plt.figure(figsize=(12, 9))
    ax1 = fig.add_subplot(111)

    ax1.plot(neuron.time_seq / 1000, neuron.cell_left.V_m, label='cell_left',
             color='darkcyan')
    ax1.plot(neuron.time_seq / 1000, neuron.cell_right.V_m, label='cell_right',
             color='darkgreen')

    ax1.plot(neuron.time_seq / 1000, ext_E, label='ext_E',
             color='darkmagenta')

    ax1.set_xlabel('time [sec]', fontsize=fsize)
    ax1.set_ylabel('potential [mV]', fontsize=fsize)

    ax1.legend(fontsize=fsize)
    plt.tick_params(labelsize=fsize)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
