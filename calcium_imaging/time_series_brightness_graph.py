import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


class calcium_time_series_graph():
    def __init__(self, data_file, save_path, save_name, dt):
        self.data_file = data_file
        self.save_path = save_path
        self.save_name = save_name
        self.dt = dt

    def run(self):
        fsize = 15
        df = pd.read_csv(self.data_file)
        soma = np.array(df['Mean1'])
        background = np.array(df['Mean2'])
        T = np.arange(0, self.dt*len(soma), self.dt)

        data = np.zeros(len(soma))

        for i in range(0, len(data)):
            data[i] = soma[i] - background[i]

        fig = plt.figure(figsize=(18, 9))
        ax1 = fig.add_subplot(111)
        ax1.set_title(os.path.basename(self.data_file))
        ax1.plot(T, data, color='darkcyan', label='soma - background')
        ax1.plot(T, soma, color='purple', label='soma')
        ax1.plot(T, background, color='crimson', label='background')
        ax1.set_xlabel('Time [ms]', fontsize=fsize)
        ax1.set_ylabel('brightness', fontsize=fsize)
        ax1.tick_params(labelsize=fsize)
        fig.tight_layout()
        fig.legend()

        fig.savefig(self.save_path+'/'+self.save_name)
        plt.close()


def main():
    data_file = 'C:/Users/ishida/Desktop/' + 'Data269.csv'
    save_path = 'C:/Users/ishida/Desktop'
    save_name = 'data1.jpg'
    dt = 25  # [ms]

    c = calcium_time_series_graph(data_file, save_path, save_name, dt)
    c.run()


if __name__ == '__main__':
    main()
