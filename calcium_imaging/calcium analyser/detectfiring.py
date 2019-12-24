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
from scipy.optimize import curve_fit
from scipy import signal
from scipy.signal import argrelmax
import statistics


class DetectFiring:
    def __init__(self, roi_intensity_csv_file, rec_time, graph_save_path, csv_save_path):
        self.roi_intensity_csv_file = roi_intensity_csv_file
        self.rec_time = rec_time
        self.graph_save_path = graph_save_path
        self.csv_save_path = csv_save_path
        # self.mod_mean = self.modify_baseline_using_time_window(window_sec=25)
        self.name_of_firing_result_csv_file = ''
        self.name_of_sigma_mean_csv_file = ''

        self.df = pd.read_csv(self.roi_intensity_csv_file)
        self.dt_sec = self.rec_time / len(self.df)
        self.time_seq = np.arange(0, len(self.df) * self.dt_sec, self.dt_sec)
        self.cell_num = int((len(self.df.columns) - 1) / 4)
        self.threshold = np.zeros(self.cell_num)
        intensity_list = []
        for cell_i in range(0, self.cell_num):
            intensity_list.append(self.df['Mean{}'.format(cell_i+1)])
        self.intensity = np.array(intensity_list)

    # 一次関数フィッティング用。ベースラインの補正に用いる
    def linear(self, x, a, b):
        return a * x + b

    # 数値判定
    def is_num(self, s):
        return s.replace(',', '').replace('.', '').replace('-', '').isnumeric()

    def plot_mean_intensity_from_raw_data(self,z,data_name):
        sigma_array = []
        mean_array = []
        for cell_i in range(0, self.cell_num):
            print(str(cell_i+1)+'個目の座標の解析をしますね')
            print('プロットのベースライン(=平坦な場所)は、どこかな？？')
            is_appropriate_range = False
            while not is_appropriate_range:
                plt.plot(self.time_seq, self.intensity[cell_i])
                # plt.show(block=False)
                plt.show()
                while True:
                    print("ベースラインの開始時間を入力してください [sec]:")
                    calc_baseline_start_sec = input('>>')
                    if self.is_num(calc_baseline_start_sec):
                        calc_baseline_start_sec = float(calc_baseline_start_sec)
                        break
                    else:
                        print("数値じゃないですよ？")
                while True:
                    print("ベースラインの終了時間はどうでしょうか？ [sec]:")
                    calc_baseline_end_sec = input('>>')
                    if self.is_num(calc_baseline_end_sec):
                        calc_baseline_end_sec = float(calc_baseline_end_sec)
                        break
                    else:
                        print("これ、数値じゃないんじゃない？")
                print('\nstart : {0} sec\nend : {1} sec'.format(calc_baseline_start_sec, calc_baseline_end_sec))

                plt.plot(self.time_seq, self.intensity[cell_i])
                plt.plot(self.time_seq[int(calc_baseline_start_sec/self.dt_sec):int(calc_baseline_end_sec/self.dt_sec)],
                         self.intensity[cell_i][int(calc_baseline_start_sec/self.dt_sec):int(calc_baseline_end_sec/self.dt_sec)])
                plt.show()

                print('ここを基準にフィッティングするけど、いいかな？ : [y]/n')
                tmp_input = input('>>')
                if tmp_input == 'n':
                    print('しょうがないな・・・もう一回指定し直し！')
                    pass
                else:
                    selected_range_time = self.time_seq[int(calc_baseline_start_sec / self.dt_sec)
                                                        :int(calc_baseline_end_sec / self.dt_sec)]
                    selected_range_intensity = self.intensity[cell_i][int(calc_baseline_start_sec / self.dt_sec)
                                                                      :int(calc_baseline_end_sec / self.dt_sec)]
                    is_appropriate_range = True

            sigma_from_selected_range = statistics.stdev(selected_range_intensity)
            mean_from_selected_range = np.mean(selected_range_intensity)
            sigma_array.append(sigma_from_selected_range)
            mean_array.append(mean_from_selected_range)
            coefficient_value = 7
            self.threshold[cell_i] = coefficient_value * sigma_from_selected_range + mean_from_selected_range
            print('sigma : {0}, mean : {1}, threshold : {2}'.format(sigma_from_selected_range, mean_from_selected_range,
                                                                    self.threshold[cell_i]))

            # ydata, xdata = np.histogram(selected_range_intensity, bins=30)
            # plt.scatter(xdata[1:], ydata)
            # plt.show()

            fsize = 30
            fig = plt.figure(figsize=(15, 9))
            fig.subplots_adjust(bottom=0.2)
            fig.subplots_adjust(left=0.2)
            ax = fig.add_subplot(111)
            ax.plot(self.time_seq, self.intensity[cell_i], label='intensity', color='dodgerblue')
            ax.plot(self.time_seq, self.threshold[cell_i]*np.ones(len(self.time_seq)), label='threshold', color='crimson')
            ax.set_xlabel('time [sec]', fontsize=fsize)
            ax.set_ylabel('Fluorescence intensity', fontsize=fsize)
            if (z[0] != -1):
                ax.vlines(z, np.amin(self.intensity[cell_i]), np.amax(self.intensity[cell_i]), "green", linestyle="dashed")
            plt.tick_params(labelsize=fsize)
            plt.legend(fontsize=30)
            # plt.title('threshold = {0}*$\sigma$+$\mu$'.format(coefficient_value), fontsize=fsize)
            plt.title(data_name+' Point'+str(cell_i+1), fontsize=fsize)
            plt.savefig(fname=self.graph_save_path + '/' + os.path.basename(self.roi_intensity_csv_file)
                        .replace('_roi_data.csv', '') + '_N{0}_s_{1}_m_{2}_({3},{4}).png'.format(cell_i + 1,
                                                                                                 sigma_from_selected_range,
                                                                                                 mean_from_selected_range,
                                                                                                 round(
                                                                                                     calc_baseline_start_sec),
                                                                                                 round(
                                                                                                     calc_baseline_end_sec)),
                        dpi=350)
            plt.show()

        df_sigma_mean = pd.DataFrame({'sigma': sigma_array,
                                      'mean': mean_array})
        self.name_of_sigma_mean_csv_file = self.csv_save_path + '/' + \
                                           os.path.basename(self.roi_intensity_csv_file).replace('_roi_data.csv', '') + '_sigma_mean.csv'

        df_sigma_mean.to_csv(self.name_of_sigma_mean_csv_file)

    def modify_baseline_using_time_window(self, window_sec):
        """
        ROIの平均輝度データを時間窓windowで区切り、時間窓中の最小値の値をあつめた配列のベースラインを一次関数で
        補正する関数
        :param window_sec: 時間窓 [sec]
        :return: 平均輝度データのベースラインを一次関数で補正した結果であるmod_meanを返す
        """
        df = pd.read_csv(self.roi_intensity_csv_file)
        time_seq = np.arange(0, len(df) * self.dt_sec, self.dt_sec)

        cell_num = int((len(df.columns) - 1) / 4)
        mod_mean = np.zeros((cell_num, len(df)))

        for i in range(0, cell_num):
            mean = df['Mean{}'.format(i + 1)]
            min_mean = mean.rolling(int(window_sec/self.dt_sec), center=True, min_periods=1).min()
            parameter_initial = [1., min(mean)]  # TODO:汎用性が低い書き方。fittingがうまくいかない可能性あり
            res = curve_fit(self.linear, time_seq, min_mean, p0=parameter_initial)
            a = res[0][0]
            b = res[0][1]

            print('baseline : y = {0} * x + {1}'.format(round(a, 4), round(b, 4)))
            baseline = np.zeros(len(mean))
            for t in range(0, len(mean)):
                baseline[t] = self.linear(t * self.dt_sec, a, b)

            mod_mean[i] = mean - baseline

            fsize = 30
            fig = plt.figure(figsize=(15, 9))
            ax = fig.add_subplot(111)
            ax.plot(time_seq, mean, label='mean', color='darkcyan')
            ax.plot(time_seq, min_mean, label='min_mean', color='purple')
            ax.plot(time_seq, mod_mean[i], label='mod_mean', color='green')
            ax.plot(time_seq, baseline, label='baseline', color='pink')
            ax.set_xlabel('time [sec]', fontsize=fsize)
            ax.set_ylabel('intensity', fontsize=fsize)
            ax.tick_params(labelsize=fsize)
            ax.legend(fontsize=fsize)
            ax.set_title('time window = {} sec'.format(window_sec))
            plt.savefig(fname=self.graph_save_path + '/' + os.path.basename(self.roi_intensity_csv_file)
                        .replace('_roi_data.csv', '') + '_N{}_baseline.png'.format(i + 1),
                        dpi=350)
            plt.close()
            # plt.show()

        return mod_mean

    def calc_sigma_subtract_moveing_ave(self, data, ave_window_sec, bins):
        moving_ave = pd.DataFrame(data).rolling(window=int(ave_window_sec / self.dt_sec),
                                                center=True,
                                                min_periods=1).mean()[0]
        fluctuation = data - moving_ave

        ydata, xdata = np.histogram(fluctuation, bins=bins)
        parameter_initial = np.array([5000., 50., xdata[np.argmax(ydata)]])  # TODO:汎用性が低い書き方。fittingがうまくいかない可能性あり
        res = curve_fit(self.gaussian, xdata[1:], ydata, p0=parameter_initial)

        a1 = res[0][0]
        sigma1 = res[0][1]
        mean1 = res[0][2]

        print('a1:{0}, sigma1:{1}, mean1:{2}'.format(round(a1, 4), round(sigma1, 4), round(mean1, 4)))
        print('3 * sigma1 = {}'.format(round(3 * sigma1, 4)))

        return a1, sigma1, mean1

    def write_hist_sub_moving_ave(self, ave_window_sec, bins):
        print('\n########### write histograms ###########\n')
        for cell_i in range(0, len(self.mod_mean)):
            moving_ave = pd.DataFrame(self.mod_mean[cell_i]).rolling(window=int(ave_window_sec / self.dt_sec),
                                                                     center=True,
                                                                     min_periods=1).mean()[0]
            fluctuation = self.mod_mean[cell_i] - moving_ave

            ydata, xdata = np.histogram(fluctuation, bins=bins)
            a, sigma, mean = self.calc_sigma_subtract_moveing_ave(data=self.mod_mean[cell_i],
                                                                  ave_window_sec=ave_window_sec, bins=bins)

            list_y = []
            x_linspace = np.linspace(min(xdata), max(xdata), len(xdata) * 10)
            for x in x_linspace:
                list_y.append(self.gaussian(x, a, sigma, mean))

            fsize = 30
            fig = plt.figure(figsize=(15, 9))
            ax0 = fig.add_subplot(111)
            ax0.scatter(xdata[1:], ydata, label='mod_mean', color='darkcyan')
            ax0.plot(x_linspace, np.array(list_y), label='fitting', color='purple')
            ax0.set_xlabel('gray value', fontsize=fsize)
            ax0.set_ylabel('num of events', fontsize=fsize)
            ax0.legend(fontsize=fsize)
            ax0.tick_params(labelsize=fsize)
            ax0.set_title('subtract (a, sigma, mean) = ({0}, {1}, {2})'.format(round(a, 4),
                                                                               round(sigma, 4),
                                                                               round(mean, 4)),
                          fontsize=fsize)
            plt.tight_layout()
            plt.savefig(self.graph_save_path + '/{0}_N{1}_hist.jpg'.format(os.path.basename(self.roi_intensity_csv_file)
                                                                           .replace('_roi_data.csv', ''), cell_i + 1),
                        dpi=350)
            plt.close()
            print(self.graph_save_path + '/{0}_N{1}_hist.png'.format(os.path.basename(self.roi_intensity_csv_file)
                                                                     .replace('_roi_data.csv', ''), cell_i + 1))

            time_seq = np.arange(0, self.dt_sec * len(self.mod_mean[cell_i]), self.dt_sec)

            fig = plt.figure(figsize=(15, 9))
            ax0 = fig.add_subplot(111)
            ax0.plot(time_seq, self.mod_mean[cell_i], label='mod_mean_N{}'.format(cell_i + 1), color='green')
            ax0.plot(time_seq, moving_ave, label='moving_ave_N{}'.format(cell_i + 1), color='darkmagenta')
            ax0.plot(time_seq, fluctuation, label='fluctuation', color='darkcyan')
            ax0.set_ylabel('intensity', fontsize=fsize)
            ax0.set_xlabel('time [sec]', fontsize=fsize)
            ax0.tick_params(labelsize=fsize)
            ax0.legend(fontsize=fsize)
            plt.tight_layout()
            plt.savefig(self.graph_save_path + '/{0}_N{1}_fluctuation.jpg'
                        .format(os.path.basename(self.roi_intensity_csv_file).replace('_roi_data.csv', ''), cell_i + 1),
                        dpi=350)
            plt.close()
            print(self.graph_save_path + '/{0}_N{1}_fluctuation.jpg'
                  .format(os.path.basename(self.roi_intensity_csv_file).replace('_roi_data.csv', ''), cell_i + 1))
        print()

    # ガウシアンフィッティング用
    def gaussian(self, x, a, sigma, mean):
        return a / np.sqrt(2.0*np.pi) / sigma * np.exp(-((x-mean)/sigma)**2/2)

    def detect_firing_only_raw_data(self, result_csv_save_path, standard_max_rise_time_steps, peak_detect_sensitivity):
        print('\n########### make firing detect csv file ###########\n')
        df_result = pd.DataFrame({'#Time (sec)': self.time_seq})
        firing = np.zeros((self.cell_num, len(self.intensity[0])))

        for cell_i in range(0, self.cell_num):
            # print('Cell {}:'.format(cell_i + 1))

            peak_time_step = argrelmax(self.intensity[cell_i], order=peak_detect_sensitivity)[0]
            # print(peak_time_step)
            prev_firing_cnt = -1

            for cnt in range(0, len(peak_time_step)):
                above_thres_index = -1
                # 輝度のピークが閾値以上のとき発火
                if self.intensity[cell_i][peak_time_step[cnt]] >= self.threshold[cell_i]:
                    # 最大バースト持続時間ステップは特に制限がない限り、standard_max_rise_time_steps
                    max_rise_time_steps = standard_max_rise_time_steps
                    # ピーク発生時刻ステップが測定開始(tstep=0)からstandard_max_rise_time_steps未満の場合、
                    # 最大バースト持続時間ステップはピーク発生時刻ステップ
                    if peak_time_step[cnt] < standard_max_rise_time_steps:
                        max_rise_time_steps = peak_time_step[cnt]
                    # 今回のバースト発火は必ず前回の発火によるピーク発生時刻ステップ以降に発生する
                    elif prev_firing_cnt >= 0 \
                            and peak_time_step[cnt] - peak_time_step[prev_firing_cnt] < standard_max_rise_time_steps:
                        max_rise_time_steps = peak_time_step[cnt] - peak_time_step[prev_firing_cnt]

                    above_thres_index = peak_time_step[cnt] - max_rise_time_steps
                    # ピーク発生時刻ステップから最大でmax_rise_time_stepsだけ遡りながら、
                    # 輝度mod_meanが閾値thresholdを下回らない最も前の時間ステップabove_thres_indexを求める
                    for tstep_i in range(1, max_rise_time_steps+1):
                        if self.intensity[cell_i][peak_time_step[cnt] - tstep_i] < self.threshold[cell_i]:
                            above_thres_index = peak_time_step[cnt] - tstep_i + 1
                            break

                # 得られたabove_thres_indexが初期値-1から更新され、かつ、
                # ピーク発生時刻ステップより時間的に前であるとき、above_thres_indexからpeak_time_step[cnt]の間の
                # 輝度mod_meanをabove_thres_mod_meanとして取り出す
                # above_thres_mod_meanの中の最小輝度をとる時間ステップがバースト発火開始時刻を表す
                # print(peak_time_step[cnt] * self.dt_sec, above_thres_index)
                if above_thres_index > -1 and above_thres_index != peak_time_step[cnt]:
                    above_thres_intensity = self.intensity[cell_i][above_thres_index:peak_time_step[cnt]]
                    burst_start_time_step = above_thres_index + np.argmin(above_thres_intensity)
                    # print(peak_time_step[cnt] * self.dt_sec, above_thres_index, burst_start_time_step * self.dt_sec)

                    for firing_step_i in range(burst_start_time_step, peak_time_step[cnt] + 1):
                        firing[cell_i][firing_step_i] = cell_i + 1
                    prev_firing_cnt = cnt

            df_result['mean_intensity_{}'.format(cell_i + 1)] = self.intensity[cell_i]
            df_result['threshold_{}'.format(cell_i + 1)] = self.threshold[cell_i] * np.ones(len(self.intensity[cell_i]))
            df_result['firing_{}'.format(cell_i + 1)] = firing[cell_i]

        self.name_of_firing_result_csv_file = result_csv_save_path + '/' + os.path.basename(self.roi_intensity_csv_file)\
            .replace('_roi_data.csv', '') + '_firing_detect.csv'
        df_result.to_csv(self.name_of_firing_result_csv_file)
        print('firing detect data is saved as :')
        print(self.name_of_firing_result_csv_file)


    def detect_firing_sub_moving_ave(self, result_csv_save_path, bins, standard_max_rise_time_steps,
                                     peak_detect_sensitivity, ave_window_sec, min_bursting_time_sec):
        print('\n########### make firing detect csv file ###########\n')
        time_seq = np.arange(0, len(self.mod_mean[0]) * self.dt_sec, self.dt_sec)

        df_result = pd.DataFrame({'#Time (sec)': time_seq})

        cell_num = len(self.mod_mean)
        firing = np.zeros((cell_num, len(self.mod_mean[0])))

        for cell_i in range(0, cell_num):
            print('Cell {}:'.format(cell_i + 1))
            sigma = abs(self.calc_sigma_subtract_moveing_ave(data=self.mod_mean[cell_i],
                                                             ave_window_sec=ave_window_sec,
                                                             bins=bins)[1])
            mean = self.calc_sigma_subtract_moveing_ave(data=self.mod_mean[cell_i],
                                                        ave_window_sec=ave_window_sec,
                                                        bins=bins)[2]
            mean_moving_ave = np.mean(
                pd.DataFrame(self.mod_mean[cell_i]).rolling(window=int(ave_window_sec / self.dt_sec),
                                                            center=True,
                                                            min_periods=1).mean()[0])

            threshold = 3 * sigma + mean

            peak_time_step = argrelmax(self.mod_mean[cell_i], order=peak_detect_sensitivity)[0]
            print(peak_time_step)
            prev_firing_cnt = -1

            for cnt in range(0, len(peak_time_step)):
                print(peak_time_step[cnt])
                above_thres_index = -1
                # 輝度のピークが閾値以上のとき発火
                if self.mod_mean[cell_i][peak_time_step[cnt]] >= threshold:
                    # 最大バースト持続時間ステップは特に制限がない限り、standard_max_rise_time_steps
                    max_rise_time_steps = standard_max_rise_time_steps
                    # ピーク発生時刻ステップが測定開始(tstep=0)からstandard_max_rise_time_steps未満の場合、
                    # 最大バースト持続時間ステップはピーク発生時刻ステップ
                    if peak_time_step[cnt] < standard_max_rise_time_steps:
                        max_rise_time_steps = peak_time_step[cnt]
                    # 今回のバースト発火は必ず前回の発火によるピーク発生時刻ステップ以降に発生する
                    elif prev_firing_cnt >= 0 \
                            and peak_time_step[cnt] - peak_time_step[prev_firing_cnt] < standard_max_rise_time_steps:
                        max_rise_time_steps = peak_time_step[cnt] - peak_time_step[prev_firing_cnt]

                    above_thres_index = peak_time_step[cnt] - max_rise_time_steps
                    # ピーク発生時刻ステップから最大でmax_rise_time_stepsだけ遡りながら、
                    # 輝度mod_meanが閾値thresholdを下回らない最も前の時間ステップabove_thres_indexを求める
                    for tstep_i in range(1, max_rise_time_steps+1):
                        if self.mod_mean[cell_i][peak_time_step[cnt] - tstep_i] < threshold:
                            above_thres_index = peak_time_step[cnt] - tstep_i + 1
                            break

                    # 得られたabove_thres_indexが初期値-1から更新され、かつ、
                    # ピーク発生時刻ステップより時間的に前であるとき、above_thres_indexからpeak_time_step[cnt]の間の
                    # 輝度mod_meanをabove_thres_mod_meanとして取り出す
                    # above_thres_mod_meanの中の最小輝度をとる時間ステップがバースト発火開始時刻を表す
                if above_thres_index > -1 and above_thres_index != peak_time_step[cnt]:
                        above_thres_mod_mean = self.mod_mean[cell_i][above_thres_index:peak_time_step[cnt]]
                        burst_start_time_step = above_thres_index + np.argmin(above_thres_mod_mean)

                        # 発火判定に時間制約を持たせる
                        # バースト持続時間がmin_bursting_time_sec以上の時、発火として判定する
                        if peak_time_step[cnt] - burst_start_time_step >= min_bursting_time_sec / self.dt_sec:
                            for firing_step_i in range(burst_start_time_step, peak_time_step[cnt] + 1):
                                firing[cell_i][firing_step_i] = cell_i + 1
                            prev_firing_cnt = cnt

            df_result['mod_mean_{}'.format(cell_i + 1)] = self.mod_mean[cell_i]
            df_result['threshold_{}'.format(cell_i + 1)] = threshold * np.ones(len(self.mod_mean[cell_i]))
            df_result['firing_{}'.format(cell_i + 1)] = firing[cell_i]

        self.name_of_firing_result_csv_file = result_csv_save_path + '/' + os.path.basename(self.roi_intensity_csv_file)\
            .replace('_roi_data.csv', '') + '_firing_detect.csv'
        df_result.to_csv(self.name_of_firing_result_csv_file)
        print('firing detect data is saved as :')
        print(self.name_of_firing_result_csv_file)
        print()