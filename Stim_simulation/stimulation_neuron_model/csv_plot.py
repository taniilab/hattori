"""
date: 20200107
created by: ishida
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import glob


class Plotter:
    def __init__(self, path, font_size=15, window_width=15, window_hight=10, line_width=1.5):
        self.path = path
        self.save_path = path
        self.csvs = path + '/cell_left_*.csv'
        self.files = glob.glob(self.csvs)
        self.fsize = font_size
        self.window_width = window_width
        self.window_hight = window_hight
        self.lw = line_width

    def plot(self):
        for _file in self.files:
            print(_file)
            df_data = pd.read_csv(_file)
            fig = plt.figure(figsize=(self.window_width, self.window_hight))
            ax1 = fig.add_subplot(111)
            ax1.plot(df_data['time [ms]'], df_data['V_intra [mV]'], color='darkgreen', lw=self.lw)
            ax1.set_xlabel('Time [ms]', fontsize=self.fsize)
            ax1.set_ylabel('Intracellular potential [mV]', fontsize=self.fsize, color='darkgreen')
            ax1.set_ylim((-139, 59))
            ax1.spines['right'].set_color('none')
            ax1.spines['top'].set_color('none')
            ax1.tick_params(labelsize=self.fsize)

            ax2 = ax1.twinx()
            ax2.plot(df_data['time [ms]'], df_data['V_extra [mV]'], color='purple', label='external potential', alpha=0.7, lw=self.lw)
            ax2.set_ylabel('$\mathrm{V_{DD}}$ [mV]', fontsize=self.fsize, color='purple')
            ax2.set_ylim((-39, 39))
            ax2.tick_params(labelsize=self.fsize)

            plt.rcParams['font.family'] = 'Ariel'
            plt.rcParams['axes.linewidth'] = 1.5
            plt.rcParams['xtick.major.width'] = 1.5
            plt.rcParams['ytick.major.width'] = 1.5
            """
            plt.tick_params(labelbottom=True,
                            labelleft=True,
                            labelright=False,
                            labeltop=False)
            """

            plt.tight_layout()
            plt.show()
            # plt.savefig(self.path + '/picture/' + os.path.basename(_file).replace('.csv', '') + '.png', dpi=600)
            # plt.close()


class Plotter2:
    def __init__(self, cell_left_csv, cell_right_csv, font_size=15, window_width=15, window_hight=10, line_width=1.5):
        self.fsize = font_size
        self.window_width = window_width
        self.window_hight = window_hight
        self.lw = line_width
        self.df_left = pd.read_csv(cell_left_csv)
        self.df_right = pd.read_csv(cell_right_csv)
        print('cell_left csv file: ' + cell_left_csv)
        print('cell_right csv file: ' + cell_right_csv)

    def plot2(self):
        fig = plt.figure(figsize=(self.window_width, self.window_hight))
        ax1 = fig.add_subplot(111)
        ax1.plot(self.df_left['time [ms]'], self.df_left['V_intra [mV]'], color='darkgreen', label='cell_left', lw=self.lw)
        ax1.plot(self.df_right['time [ms]'], self.df_right['V_intra [mV]'], color='darkcyan', label='cell_right', lw=self.lw, alpha=0.7)
        ax1.set_xlabel('Time [ms]', fontsize=self.fsize)
        ax1.set_ylabel('Intracellular potential [mV]', fontsize=self.fsize, color='darkgreen')
        ax1.set_ylim((-139, 59))
        ax1.spines['top'].set_color('none')
        ax1.tick_params(labelsize=self.fsize)
        # ax1.legend(fontsize=self.fsize-10)

        ax2 = ax1.twinx()
        ax2.plot(self.df_left['time [ms]'], self.df_left['V_extra [mV]'], color='purple', label='external potential', alpha=0.7,
                 lw=self.lw)
        ax2.set_ylabel('$\mathrm{V_{DD}}$ [mV]', fontsize=self.fsize, color='purple')
        ax2.set_ylim((-39, 39))
        ax2.tick_params(labelsize=self.fsize)

        plt.rcParams['font.family'] = 'Ariel'
        plt.rcParams['axes.linewidth'] = 1.5
        plt.rcParams['xtick.major.width'] = 1.5
        plt.rcParams['ytick.major.width'] = 1.5

        plt.tight_layout()
        plt.show()


def main():

    csv_folder_path = 'C:/Users/ishida/Desktop/stim_neuron_sim/for_plot'
    c = Plotter(path=csv_folder_path, font_size=45, line_width=3)
    c.plot()
    """
    path = 'C:/Users/ishida/Desktop/stim_neuron_sim/for_plot'
    cell_left_csv = path + '/' + 'cell_left_' + '2Mohm_-20_DC.csv'
    cell_right_csv = path + '/' + 'cell_right_' + '2Mohm_-20_DC.csv'

    d = Plotter2(cell_left_csv=cell_left_csv, cell_right_csv=cell_right_csv,
                 font_size=45, line_width=3)
    d.plot2()
    """


if __name__ == '__main__':
    main()



