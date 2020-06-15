import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec


save_path = "Z:/simulation/test"
filename = "2020_6_11_14_47_51_Iext_amp0.001_Pmax_AMPA0.0001_Pmax_NMDA0.0001_LIF"

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
    plt.rcParams["font.size"] = 14
    if not os.path.isdir(save_path + '/RC'):
        os.mkdir(save_path + '/RC')
    num_read_nodes = 5
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
    print(index_tmp)
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
    #plt.title(filename)
    gs_master = GridSpec(nrows=num_read_nodes + 1, ncols=2)
    gs_rc = GridSpecFromSubplotSpec(nrows=1, ncols=2, subplot_spec=gs_master[0, 0:2])
    ax_rc = fig.add_subplot(gs_rc[:, :])
    gs_status = GridSpecFromSubplotSpec(nrows=num_read_nodes, ncols=2, subplot_spec=gs_master[1:, :], hspace=0.4, wspace=0.15)
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
    fig.tight_layout()
    plt.show()
    #plt.savefig(save_path + '/RC/' + filename + '.png')
    plt.close(fig)

if __name__ == '__main__':
     main()