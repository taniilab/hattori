# coding=utf-8
"""
date:190915
created by takahashi & ishida

Extended by @Ittan_moment
update:20191021
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import statistics


class AnalysisFiring:
    def __init__(self, roi_intensity_csv_file, firing_result_csv_file, sigma_mean_csv_file, graph_save_path, rec_time):
        self.roi_intensity_csv_file = roi_intensity_csv_file
        self.firing_result_csv_file = firing_result_csv_file
        self.sigma_mean_csv_file = sigma_mean_csv_file
        self.graph_save_path = graph_save_path
        self.rec_time = rec_time
        self.df = pd.read_csv(self.roi_intensity_csv_file)
        self.dt_sec = self.rec_time / len(self.df)

    def plot_mod_mean_and_threshold(self, z):
        """
        ベースラインを補正した平均輝度データであるmod_meanと閾値であるthresholdのグラフを生成する関数
        :return: グラフが生成・保存される
        """
        print('\n########### write mod_mean plot ###########\n')
        df = pd.read_csv(self.firing_result_csv_file)

        fsize = 30
        for i in range(0, int((len(df.columns) - 1) / 3)):
            print('max(mod_mean[{0}]) = {1}'.format(i, max(df['mod_mean_{}'.format(i + 1)])))
            fig = plt.figure(figsize=(15, 9))
            ax0 = fig.add_subplot(111)
            ax0.plot(df['#Time (sec)'], df['mod_mean_{}'.format(i + 1)], label='mod_mean_{}'.format(i + 1))

            ax0.plot(df['#Time (sec)'], df['threshold_{}'.format(i + 1)], label='threshold_{}'.format(i + 1),
                     lw=3)

            ax0.set_xlabel('#Time (sec)', fontsize=fsize)
            ax0.set_ylabel('Mean fluorescence intensity', fontsize=fsize)
            ax0.vlines(z, -500, 2500, "green", linestyle="dashed")
            # ax0.set_ylim(-500, 2500)
            plt.tick_params(labelsize=fsize)
            plt.legend(fontsize=fsize)
            plt.tight_layout()
            plt.savefig(fname=self.graph_save_path + '/' + os.path.basename(self.firing_result_csv_file)
                        .replace('_firing_detect.csv', '') + '_N{}_mod_mean_plot.png'.format(i + 1), dpi=350)
            plt.close()
            # plt.show()
            print(
                self.graph_save_path + '/' + os.path.basename(self.firing_result_csv_file)
                .replace('_firing_detect.csv', '') + '_N{}_mod_mean_plot.png'.format(i + 1))
            print()
            """
            fig = plt.figure(figsize=(15, 9))
            ax0 = fig.add_subplot(111)
            ax0.plot(df['#Time (sec)'], df['mod_mean_{}'.format(i + 1)], label='mod_mean_{}'.format(i + 1))
            ax0.set_xlabel('#Time (sec)', fontsize=fsize)
            ax0.set_ylabel('Mean fluorescence intensity', fontsize=fsize)
            ax0.set_ylim(-500, 2500)
            plt.tick_params(labelsize=fsize)
            # plt.legend(fontsize=fsize)
            plt.tight_layout()
            plt.savefig(fname=self.graph_save_path + '/' + os.path.basename(self.firing_result_csv_file)
                        .replace('_firing_detect.csv', '') + '_N{}_mod_mean_plot.png'.format(i + 1),
                        dpi=350)
            plt.close()
            # plt.show()
            print(
                self.graph_save_path + '/' + os.path.basename(self.firing_result_csv_file)
                .replace('_firing_detect.csv', '') + '_N{}_mod_mean_plot.png'.format(i + 1))
            print()
            """

    def raster_plot(self, z):
        """
        ラスタープロットを生成する関数
        :return: ラスタープロットが生成・保存される
        """
        print('\n########### write raster plot ###########\n')
        df = pd.read_csv(self.firing_result_csv_file)
        fsize = 30
        fig_raster = plt.figure(figsize=(18, 9))
        ax_raster = fig_raster.add_subplot(111)
        for j in range(1, int((len(df.columns) - 1) / 3) + 1):
            tmp = np.where(np.array(df['firing_{}'.format(j)]) == 0, -10, np.array(df['firing_{}'.format(j)]))
            ax_raster.scatter(df['#Time (sec)'], tmp, marker='|', s=500, color='red')
        ax_raster.set_yticks(np.arange(1, int((len(df.columns) - 1) / 3) + 1))
        ax_raster.set_ylim(0.1, int((len(df.columns) - 1) / 3) + 0.9)
        ax_raster.set_ylabel('Cell No.', fontsize=fsize)
        ax_raster.set_xlabel('#Time (sec)', fontsize=fsize)
        if z[0] != -1:
            ax_raster.vlines(z, 1, int((len(df.columns) - 1) / 3) + 1, "blue", linestyles='dashed')
        plt.tick_params(labelsize=fsize)
        plt.savefig(fname=self.graph_save_path + '/' + os.path.basename(self.firing_result_csv_file)
                    .replace('_firing_detect.csv', '') + '_raster_plot.png',
                    dpi=350)
        # plt.show()
        print(self.graph_save_path + '/' + os.path.basename(self.firing_result_csv_file)
              .replace('_firing_detect.csv', '') + '_raster_plot.png')

    def network_analysis(self, CellPercent, thres_burst_time):
        """
        ネットワークバーストの頻度・周波数・IBIを計算する関数
        :param CellPercent: 存在する細胞の内、cell_percent以上の割合の細胞が同時に発火したときに
                              バースト発火の条件を満たしうる
        :param thres_burst_time: 存在する細胞の内、cell_percent以上の割合の細胞が同時に発火し、かつ、
                                　thres_burst_time [sec]以上発火するとバースト発火としてみなす
        :return: 計算したネットワークバーストの頻度・周波数・IBIを表示する
        """
        print('\n########### network burst analysis ###########\n')
        df = pd.read_csv(self.firing_result_csv_file)
        cell_num = int((len(df.columns)-1) / 3)
        burst_duration = []
        burst_event = 0
        prev_time = 0
        IBI = []
        cnt = {}

        firing = np.zeros((cell_num, len(df)))
        for i in range(0, cell_num):
            firing[i] = np.array(df['firing_{}'.format(i + 1)])

        for step in range(0, len(df)):
            count_cell = 0
            for i in range(0, cell_num):
                if firing[i, step] != 0:
                    count_cell += 1
            cnt[step * self.dt_sec] = count_cell  # 同じ時刻に発火している細胞数

        # print('time [sec]\tsum(firing cell)')
        for time_i in sorted(cnt.keys()):
            if cnt[time_i] >= cell_num * CellPercent:
                if time_i - prev_time > self.dt_sec:
                    # 連続しているバーストは1つとして数える
                    # print('{:.3f}\t\t{}'.format(time_i, cnt[time_i]))
                    burst_event += 1
                    if prev_time != 0:
                        IBI.append(time_i - prev_time)
                prev_time = time_i

        for cell_i in range(0, cell_num):
            burst_flag = 0
            for step in range(0, len(df)):
                if firing[cell_i, step] != 0:  # バーストの判定方法が山本先生のプログラムとは異なる
                    burst_flag += 1
                elif burst_flag != 0:  # 発火が止まったとき
                    if burst_flag * self.dt_sec >= thres_burst_time:
                        burst_duration.append(burst_flag * self.dt_sec)
                    burst_flag = 0

        print('ネットワークバーストの頻度 : {:.3f}秒間に{}回'.format(self.dt_sec * (len(df) - 1), burst_event))
        print('ネットワークバーストの周波数 : {:.2f} Hz'.format(burst_event / (self.dt_sec * (len(df) - 1))))

        if len(burst_duration) >= 1:
            print('個別のバースト発火の平均持続時間(±標準偏差) : {:.3f}秒(±{:.3f}秒) ;'
                  ' n_burst = {}'.format(np.mean(burst_duration), statistics.pstdev(burst_duration),
                                           len(burst_duration)))

        if len(IBI) >= 1:
            print('ネットワークバーストの平均間隔(IBI)(±標準偏差) : {:.3f}秒(±{:.3f}秒) ;'
                  ' n_IBI = {}'.format(np.mean(IBI), statistics.pstdev(IBI), len(IBI)))

    def burst_analysis(self):
        print('\n########### burst analysis ###########\n')
        df = pd.read_csv(self.firing_result_csv_file)
        df_sigma_mean = pd.read_csv(self.sigma_mean_csv_file)
        df_roi_intensity = pd.read_csv(self.roi_intensity_csv_file)
        cell_num = int((len(df.columns) - 1) / 3)

        for i in range(0, cell_num):
            burst_duration = []
            IBI = []
            start_time = []
            end_time = []
            signal_intensity = []
            prev_time = 0
            firing = np.array(df['firing_{}'.format(i + 1)])
            burst_flag = 0
            for tstep_i in range(0, len(firing)):
                if firing[tstep_i] != 0:
                    if burst_flag == 0:
                        start_time.append(tstep_i * self.dt_sec)
                        if prev_time != 0:
                            IBI.append(tstep_i * self.dt_sec - prev_time - self.dt_sec)
                    prev_time = tstep_i * self.dt_sec
                    burst_flag += 1
                elif burst_flag != 0:  # 発火が止まったとき
                    end_time.append(tstep_i * self.dt_sec)
                    signal_intensity.append(df_roi_intensity['Mean{}'.format(i + 1)][tstep_i-1])
                    # print(tstep_i)
                    burst_duration.append(burst_flag * self.dt_sec)
                    burst_flag = 0

            if len(burst_duration) >= 1:
                print('Cell_{}のバースト発火の平均持続時間(±標準偏差) : {:.3f}秒(±{:.3f}秒) ;'
                      ' n_burst = {}'.format(i+1, np.mean(burst_duration), statistics.pstdev(burst_duration),
                                             len(burst_duration)))
                df_res = pd.DataFrame({'firingNo.': np.arange(0, len(burst_duration), 1),
                                       'duration [sec]': burst_duration,
                                       'ave_duration [sec]': np.mean(np.array(burst_duration)),
                                       'stdev_duration [sec]': statistics.pstdev(burst_duration),
                                       'event_number': len(burst_duration)},
                                      columns=['firingNo.', 'duration [sec]', 'ave_duration [sec]',
                                               'stdev_duration [sec]', 'event_number'])
                tmp_IBI = [0]
                for j in range(0, len(IBI)):
                    tmp_IBI.append(IBI[j])
                df_res['IBI'] = tmp_IBI

                if len(tmp_IBI) == 1:
                    df_res['ave_IBI'] = ''
                    df_res['stdev_IBI'] = ''
                else:
                    df_res['ave_IBI'] = np.mean(np.array(IBI))
                    df_res['stdev_IBI'] = statistics.pstdev(IBI)
                df_res['burst_start_time'] = start_time
                df_res['burst_end_time'] = end_time
                sigma = df_sigma_mean['sigma'][i]
                mean = df_sigma_mean['mean'][i]
                signal_noise_ratio = (np.array(signal_intensity) - mean)
                # print(signal_noise_ratio)
                signal_noise_ratio = signal_noise_ratio / sigma
                # print(signal_noise_ratio)
                df_res['SN_ratio'] = signal_noise_ratio

                # print(sigma, mean, signal_intensity)


                if not os.path.isdir(os.path.dirname(self.firing_result_csv_file) + '/burst_analysis'):
                    os.mkdir(os.path.dirname(self.firing_result_csv_file) + '/burst_analysis')
                df_res.to_csv(os.path.dirname(self.firing_result_csv_file) + '/burst_analysis/'
                              + os.path.basename(self.firing_result_csv_file).replace('_firing_detect.csv', '')
                              + '_N{}_burst_analysis.csv'.format(i + 1))
