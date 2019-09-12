"""
date: 20190628
created by: ishida
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class Plot:
    def __init__(self, roi_csv_file, dt):
        self.file = roi_csv_file
        self.dt = dt

    def mean_plot(self):
        df = pd.read_csv(self.file)
        cell_num = int(len(df.columns) / 4)
        time = np.arange(0, len(df) * self.dt, self.dt)

        fsize = 15
        fig = plt.figure(figsize=(18, 9))
        ax = fig.add_subplot(111)
        for i in range(0, cell_num):
            ax.plot(time, df['Mean{}'.format(i + 1)], label='Cell {}'.format(i + 1))
        ax.set_xlabel('time [sec]', fontsize=fsize)
        ax.set_ylabel('intensity', fontsize=fsize)
        ax.tick_params(labelsize=fsize)

        plt.tight_layout()
        plt.show()


def main():
    roi_csv_file = 'C:'
    dt = 0.025  # [sec]
    c = Plot(roi_csv_file, dt)
    c.mean_plot()


if __name__ == '__main__':
    main()