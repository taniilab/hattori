import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


save_path = "Z:/simulation/test"
filename = "2020_6_5_14_41_28_Iext_amp0.001_Pmax_AMPA3e-05_Pmax_NMDA3e-05_LIF.csv"

class LSM():
    def __init__(self):
        pass

    def train(self, train_data, target_data, lamda=1, regularization=1):
        if regularization == 0:
            pass
        else:
            self.output_w = (np.linalg.inv(train_data.T @ train_data + \
                                           lamda * np.identity(np.size(train_data, 1))) @ train_data.T) @ target_data
    def predict(self):
        pass


def main():
    num_read_nodes = 5
    read_cols = ['T_0 [ms]']
    for i in range(num_read_nodes):
        read_cols.append('V_{} [mV]'.format(i))
        read_cols.append('I_syn_{} [uA]'.format(i))

    read_cols.append('I_AMPA_{} [uA]'.format(0))
    read_cols.append('I_NMDA_{} [uA]'.format(0))
    read_cols.append('Iext_{} [uA]'.format(0))
    print(read_cols)
    """
    read_cols = ['T_0 [ms]',  # 0
                 'V_0 [mV]',  # 1
                 'I_syn_0 [uA]',  # 2
                 'I_AMPA_0 [uA]',  # 3
                 'I_NMDA_0 [uA]',  # 4
                 'V_1 [mV]',  # 5
                 'Iext_0 [uA]', #6
                 'V_2 [mV]',
                 'V_3 [mV]',
                 'V_4 [mV]',
                 'V_5 [mV]',
                 'V_6 [mV]',
                 'V_7 [mV]',
                 'V_8 [mV]',
                 'V_9 [mV]']
    """
    df = pd.read_csv(save_path + '/' + filename, usecols=read_cols, skiprows=1)[read_cols]
    train_ratio = 0.5
    border = int(len(df.values[:, 0])*train_ratio)
    print(border)
    print(df)

    times = df.values[:, 0].reshape((len(df.values[:, 0]), 1))
    times_bef = df.values[:border, 0].reshape((len(df.values[:border, 0]), 1))
    times_af = df.values[border:, 0].reshape((len(df.values[border:, 0]), 1))

    index_tmp = []
    index_tmp.append(int(2*num_read_nodes+3))
    input = df.values[:, index_tmp].reshape((len(df.values[:, index_tmp]), len(index_tmp))) # ampa, nmda, "Iext"
    target = input[:border]
    index_tmp = []
    for i in range(num_read_nodes):
        index_tmp.append(i*2+1)
    output_train = df.values[:border, index_tmp].reshape((len(df.values[:border, index_tmp]), len(index_tmp)))
    output_predict = df.values[border:, index_tmp].reshape((len(df.values[border:, index_tmp]), len(index_tmp)))

    Isyn = df.values[:, 2].reshape((len(df.values[:, 2]), 1))
    IAMPA = df.values[:, 3].reshape((len(df.values[:, 3]), 1))
    INMDA = df.values[:, 4].reshape((len(df.values[:, 4]), 1))

    lsm = LSM()
    lsm.train(output_train, target)
    predict_res = (output_predict @ lsm.output_w).T

    fig = plt.figure(figsize=(12, 12))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax1.plot(times_bef, output_train[:, 0], label="train_output_n0")
    ax1.plot(times_bef, output_train[:, 1], label="train_output_n1")
    ax1.plot(times, input[:, 0], label="input(target)_Iext0")
    ax1.plot(times_af, predict_res[0], label="after training")
    ax1.legend()
    ax2.plot(times, Isyn[:, 0], label="Isyn")
    ax2.plot(times, IAMPA[:, 0], label="IAMPA")
    ax2.plot(times, INMDA[:, 0], label="INMDA")
    ax2.legend()
    print(times.shape)
    print(output_train.shape)
    print(target.shape)
    print(lsm.output_w.shape)
    print((output_train @ lsm.output_w).shape)
    print(output_predict.shape)
    print("W:{0}".format(lsm.output_w))
    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
     main()