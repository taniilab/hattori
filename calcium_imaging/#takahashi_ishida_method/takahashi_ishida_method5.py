""""
date:190626
created by takahashi & ishida
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import glob
import os
from scipy.optimize import curve_fit
import matplotlib.animation as animation
from PIL import Image
import statistics
from scipy import signal


class Ca_analyze:
    def __init__(self, path, width=336, height=256):
        self.path = path
        self.folders = path + '/Field Data/Field*'
        self.files = {}
        self.files = glob.glob(self.folders)
        self.height = height
        self.width = width

    def plot_heatmap(self, save_path):
        """
        解凍したcxdファイルのデータを用いて、すべてのフレームの輝度データを取得し、ヒートマップとして保存する関数
        gifを作成したいときなどに便利
        :param save_path: ヒートマップを保存するパス
        """
        for f in range(0, len(self.files)):

            print(self.path + '/Field Data/Field {}/i_Image1/Bitmap 1'.format(f + 1))
            bitmap = open(self.path + '/Field Data/Field {}/i_Image1/Bitmap 1'.format(f + 1), 'rb')
            """
            print(self.path + '/Field Data/Field {}/i_Image1/Bitmap 1'.format(int(70 / 0.025)))
            bitmap = open(self.path + '/Field Data/Field {}/i_Image1/Bitmap 1'.format(int(70 / 0.025)), 'rb')
            """
            data = bitmap.read()

            list_1 = []
            j = 0
            for i in range(0, int(len(data) / 2)):
                list_1.append(int.from_bytes([data[j], data[j + 1]], 'little'))
                j += 2

            cnt = 0
            map = np.zeros((self.height, self.width))
            for i in range(0, self.height):
                for j in range(0, self.width):
                    map[i][j] = int(list_1[cnt])
                    cnt += 1

            plt.figure(figsize=(12, 9))
            sns.heatmap(map, square=True, vmax=4095, vmin=200)
            plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
            plt.savefig(save_path + '/Field_{}.png'.format(f + 1), dpi=350)
            plt.close()
            # plt.show()

    def make_gif(self, png_path, gif_save_path, gif_name):
        """
        plot_heatmap()で出力したヒートマップのpngファイルからgifアニメを作成する関数
        imagemagickなどあらかじめPCの環境を用意しておく必要がある
        :param png_path: plot_heatmap()で出力したpngファイルがあるパス
        :param gif_save_path: 作成したgifアニメを保存するパス
        :param gif_name: 作成したgifアニメの名前
        :return: gifアニメができる
        """
        pngs = png_path + '/*.png'
        _pngs = glob.glob(pngs)

        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111)
        ax.spines['right'].set_color('None')
        ax.spines['left'].set_color('None')
        ax.spines['top'].set_color('None')
        ax.spines['bottom'].set_color('None')
        ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
        ax.tick_params(bottom=False, left=False, right=False, top=False)
        ims = []

        k = np.arange(400, 700, 10)
        for f in range(0, len(_pngs)):
            print(png_path + '/Field_{}.png'.format(k[f]))
            tmp = Image.open(png_path + '/Field_{}.png'.format(k[f]))
            ims.append([plt.imshow(tmp, interpolation='spline36')])
            # k += 100

        ani = animation.ArtistAnimation(fig, ims, interval=20)
        ani.save(gif_save_path + '/' + gif_name, writer='imagemagick')

    def input_roi_centers(self, x, y, roi_num=1, heatmap_save_path='C:/Users/admin/Desktop'):
        """
        引数としてROIのcenterとなる座標を与え、そのcenterを基準に指定したピクセル数・相対位置でROIとして
        取得するピクセルの番地をself.roiに格納する関数
        :param x: 指定したROIのcenterのx座標のみが入った配列
        :param y: 指定したROIのcenterのy座標のみが入った配列
        :param roi_num: どのようなROIを取るかを指定する番号
        :param heatmap_save_path: 指定したROIを視覚的に確認できるヒートマップを保存するパス
        :return: この関数を実行した結果、self.roiの中にROIとして指定したピクセルの番地が入る
        """
        centers = np.zeros(len(x))  # 指定したROIのcenterとなる番地を保存する配列
        for i in range(0, len(x)):
            print('center_{0}   x : {1},  y : {2}'.format(i, x[i], y[i]))  # ROIのx座標，y座標をprint
            centers[i] = y[i] * self.width + x[i]

        # roi_num:
        # 0 : 13 pixels
        # 1 : 4 pixels

        """
                ROI image #0:
                　　□
                　□□□
                □□■□□
                　□□□
                　　□
                
                ■:center
        """
        if roi_num == 0:
            self.roi = np.zeros((len(x), 13))
            for _center in range(0, len(x)):
                self.roi[_center] = np.array([centers[_center] - self.width * 2,
                                              centers[_center] - self.width - 1, centers[_center] - self.width, centers[_center] - self.width + 1,
                                              centers[_center] - 2, centers[_center] - 1, centers[_center], centers[_center] + 1, centers[_center] + 2,
                                              centers[_center] + self.width - 1, centers[_center] + self.width, centers[_center] + self.width + 1,
                                              centers[_center] + self.width * 2])

        """
                ROI image #1:
                ■□
                □□
                
                ■:center
        """
        if roi_num == 1:
            self.roi = np.zeros((len(x), 4))
            for _center in range(0, len(x)):
                self.roi[_center] = np.array([centers[_center], centers[_center] + 1,
                                              centers[_center] + self.width, centers[_center] + self.width + 1])

        # ROIをヒートマップに表示して視覚的に確認する
        print(self.path + '/Field Data/Field 1/i_Image1/Bitmap 1')
        bitmap = open(self.path + '/Field Data/Field 1/i_Image1/Bitmap 1', 'rb')
        data = bitmap.read()

        list_1 = []
        j = 0
        for i in range(0, int(len(data) / 2)):
            list_1.append(int.from_bytes([data[j], data[j + 1]], 'little'))
            j += 2

        tmp = np.array(list_1)
        for _center in range(0, len(self.roi)):
            for j in range(0, len(self.roi[0])):
                tmp[int(self.roi[_center, j])] = 0

        cnt = 0
        map_roi = np.zeros((self.height, self.width))
        for i in range(0, self.height):
            for j in range(0, self.width):
                map_roi[i][j] = tmp[cnt]
                cnt += 1

        plt.figure(figsize=(12, 9))
        sns.heatmap(map_roi, square=True, vmax=4095, vmin=0)

        plt.savefig(heatmap_save_path + '/' + os.path.basename(self.path) + '_ROI.png', dpi=350)
        plt.close()
        # plt.show()
        print('ROI check heatmap is saved as :')
        print(heatmap_save_path + '/' + os.path.basename(self.path) + '_ROI.png\n')

    def make_roi_csv(self, csv_save_path):
        """
        input_roi_centers()を実行することでself.roiに格納されたroiのピクセルの番地を用いる
        self.roiの番地からすべてのフレームでROIのピクセルの輝度データを取得する
        取得した輝度データを用いてすべてのROIの
        ・ピクセル数   Area
        ・平均輝度    Mean
        ・最小輝度    Min
        ・最大輝度    Max
        を求め、csvファイルに保存する
        :param csv_save_path: 各ROIのArea,Mean,Min,Maxが記されたcsvファイルを保存するパス
        """
        print('\n########### make roi csv file ###########\n')
        roi_pixel_data = np.zeros((len(self.roi), len(self.roi[0]), len(self.files)))
        roi_mean_data = np.zeros((len(self.roi), len(self.files)))
        roi_min_data = np.zeros((len(self.roi), len(self.files)))
        roi_max_data = np.zeros((len(self.roi), len(self.files)))
        k = 1
        for f in range(0, len(self.files)):
            print(self.path + '/Field Data/Field {}/i_Image1/Bitmap 1'.format(k))
            bitmap = open(self.path + '/Field Data/Field {}/i_Image1/Bitmap 1'.format(k), 'rb')
            data = bitmap.read()

            list_1 = []
            j = 0
            for i in range(0, int(len(data) / 2)):
                list_1.append(int.from_bytes([data[j], data[j + 1]], 'little'))
                j += 2

            for _center in range(0, len(self.roi)):
                for _roi_pixel in range(0, len(self.roi[0])):
                    roi_pixel_data[_center, _roi_pixel, f] = list_1[int(self.roi[_center, _roi_pixel])]

                roi_mean_data[_center, f] = np.mean(roi_pixel_data[_center, :, f])
                roi_min_data[_center, f] = min(roi_pixel_data[_center, :, f])
                roi_max_data[_center, f] = max(roi_pixel_data[_center, :, f])

            k += 1

        df_roi_data = pd.DataFrame({'Area1': len(self.roi[0]) * np.ones(len(roi_mean_data[0])),
                                    'Mean1': roi_mean_data[0],
                                    'Min1': roi_min_data[0],
                                    'Max1': roi_max_data[0]},
                                   columns=['Area1', 'Mean1', 'Min1', 'Max1'])

        for i in range(1, len(self.roi)):
            df_roi_data['Area{}'.format(i + 1)] = len(self.roi[0]) * np.ones(len(roi_mean_data[0]))
            df_roi_data['Mean{}'.format(i + 1)] = roi_mean_data[i]
            df_roi_data['Min{}'.format(i + 1)] = roi_min_data[i]
            df_roi_data['Max{}'.format(i + 1)] = roi_max_data[i]

        df_roi_data.to_csv(csv_save_path + '/' + os.path.basename(self.path) + '_roi_data.csv')
        print('ROI pixels data is saved as :')
        print(csv_save_path + '/' + os.path.basename(self.path) + '_roi_data.csv\n')

    # 一次関数フィッティング用。ベースラインの補正に用いる
    def linear(self, x, a, b):
        return a * x + b

    def modify_baseline(self, roi_csv_file, dt):
        """
        ROIの平均輝度データのベースラインを一次関数で補正する関数
        :param roi_csv_file: make_roi_csv()で作成したcsvファイル
        :param dt: イメージングを実施した際のフレームレートの逆数
        :return: 平均輝度データのベースラインを一次関数で補正した結果であるmod_meanを返す
        """
        df = pd.read_csv(roi_csv_file)
        time = np.arange(0, len(df) * dt, dt)

        cell_num = int((len(df.columns) - 1) / 4)
        mod_mean = np.zeros((cell_num, len(df)))

        for i in range(0, cell_num):
            mean = np.array(df['Mean{}'.format(i + 1)])
            parameter_initial = [2., 200]
            res = curve_fit(self.linear, time, mean, p0=parameter_initial)
            a = res[0][0]
            b = res[0][1]

            print('baseline : y = {0} * x + {1}'.format(round(a, 4), round(b, 4)))
            baseline = np.zeros(len(mean))
            for t in range(0, len(mean)):
                baseline[t] = self.linear(t * dt, a, b)

            mod_mean[i] = mean - baseline

        return mod_mean

    def modify_baseline_window(self, roi_csv_file, baseline_save_path, dt, window):
        """
        ROIの平均輝度データを時間窓windowで区切り、時間窓中の最小値の値をあつめた配列のベースラインを一次関数で
        補正する関数
        :param roi_csv_file: make_roi_csv()で作成したcsvファイル
        :param dt: イメージングを実施した際のフレームレートの逆数
        :param window: 時間窓 [sec]
        :return: 平均輝度データのベースラインを一次関数で補正した結果であるmod_meanを返す
        """
        df = pd.read_csv(roi_csv_file)
        time = np.arange(0, len(df) * dt, dt)

        cell_num = int((len(df.columns) - 1) / 4)
        mod_mean = np.zeros((cell_num, len(df)))

        for i in range(0, cell_num):
            mean = df['Mean{}'.format(i + 1)]
            min_mean = mean.rolling(int(window/dt), center=True, min_periods=1).min()
            parameter_initial = [1., 500]
            res = curve_fit(self.linear, time, min_mean, p0=parameter_initial)
            a = res[0][0]
            b = res[0][1]

            print('baseline : y = {0} * x + {1}'.format(round(a, 4), round(b, 4)))
            baseline = np.zeros(len(mean))
            for t in range(0, len(mean)):
                baseline[t] = self.linear(t * dt, a, b)

            mod_mean[i] = mean - baseline

            fsize = 30
            fig = plt.figure(figsize=(18, 9))
            ax = fig.add_subplot(111)
            ax.plot(time, mean, label='mean', color='darkcyan')
            ax.plot(time, min_mean, label='min_mean', color='purple')
            ax.plot(time, mod_mean[i], label='mod_mean', color='green')
            ax.plot(time, baseline, label='baseline', color='pink')
            ax.set_xlabel('time [sec]', fontsize=fsize)
            ax.set_ylabel('intensity', fontsize=fsize)
            ax.tick_params(labelsize=fsize)
            ax.legend(fontsize=fsize)
            ax.set_title('time window = {}'.format(window))
            plt.savefig(fname=baseline_save_path + '/' + os.path.basename(roi_csv_file).replace('_roi_data.csv', '')
                              + '_N{}_baseline.png'.format(i + 1),
                        dpi=350)
            plt.close()
            # plt.show()

        return mod_mean


    def modify_baseline_signal(self, roi_csv_file, dt):
        """
        scipyのコマンドを用いてROIの平均輝度データのベースラインを補正する関数
        :param roi_csv_file: make_roi_csv()で作成したcsvファイル
        :param dt: イメージングを実施した際のフレームレートの逆数
        :return:
        """
        df = pd.read_csv(roi_csv_file)
        time = np.arange(0, len(df) * dt, dt)

        cell_num = int((len(df.columns) - 1) / 4)

        for i in range(0, cell_num):
            mean = df['Mean{}'.format(i + 1)]
            mod_mean = signal.detrend(mean)

            plt.plot(time, mean, label='mean')
            plt.plot(time, mod_mean, label='mod_mean')
            plt.legend()
            plt.show()

    def calc_sigma(self, data, bin):
        """
        引数dataのヒストグラムを2つのガウシアンの足し合わせであるsum_gaussian()でフィッティングする
        フィッティングした結果のパラメータを返す
        :param data: sum_gaussian()を用いてヒストグラムをフィッティングしたいデータの配列
        :param bin: ヒストグラムを取る時のbinsの値
        return: フィッティングした結果のパラメータa1, sigma1, mean1, a2, sigma2, mean2を返す
        """
        ydata, xdata = np.histogram(data, bins=bin)

        parameter_initial = np.array([5000., 50., 0.1])
        res = curve_fit(self.gaussian, xdata[1:], ydata, p0=parameter_initial)

        a1 = res[0][0]
        sigma1 = res[0][1]
        mean1 = res[0][2]

        print('a1:{0}, sigma1:{1}, mean1:{2}'.format(round(a1, 4), round(sigma1, 4), round(mean1, 4)))

        print('3 * sigma1 = {}'.format(round(3 * sigma1, 4)))

        return a1, sigma1, mean1

    def calc_sigma_sum_gausian(self, data, bin):
        """
        引数dataのヒストグラムを2つのガウシアンの足し合わせであるsum_gaussian()でフィッティングする
        フィッティングした結果のパラメータを返す
        :param data: sum_gaussian()を用いてヒストグラムをフィッティングしたいデータの配列
        :param bin: ヒストグラムを取る時のbinsの値
        return: フィッティングした結果のパラメータa1, sigma1, mean1, a2, sigma2, mean2を返す
        """
        ydata, xdata = np.histogram(data, bins=bin)

        parameter_initial = np.array([5000., 50., 0.1, 1., 1., 1.])
        res = curve_fit(self.sum_gaussian, xdata[1:], ydata, p0=parameter_initial)
        a1 = res[0][0]
        sigma1 = res[0][1]
        mean1 = res[0][2]
        a2 = res[0][3]
        sigma2 = res[0][4]
        mean2 = res[0][5]


        print('a1:{0}, sigma1:{1}, mean1:{2}, \na2:{3}, sigma2:{4}, mean2:{5}'.format(round(a1, 4), round(sigma1, 4),
                                                                                      round(mean1, 4), round(a2, 4),
                                                                                      round(sigma2, 4), round(mean2, 4)))

        print('sigma1 = {}'.format(round(sigma1, 4)))
        print('3 * sigma1 = {}'.format(round(3 * sigma1, 4)))
        return a1, sigma1, mean1, a2, sigma2, mean2


    def write_hist(self, roi_csv_file, data, bin, parameter, histogram_save_path, i):
        """
        calc_sigma(data)で計算したパラメータを使用して、dataのヒストグラムとフィッティング曲線をプロットした
        グラフを生成し保存する関数
        :param roi_csv_file : dataが入っているroiの輝度データが入ったcsvファイル
        :param data: calc_sigma()で与えたフィッティングしたいデータの配列
        :param bin: ヒストグラムを取る時のbinsの値
        :param parameter: calc_sigma(data, bin)の結果
        :param histogram_save_path: 生成したヒストグラムを保存するパス
        :param i: 何番目のROIデータかを示すindex
        :return: ヒストグラムが生成・保存される
        """
        ydata, xdata = np.histogram(data, bins=bin)

        list_y = []
        x_linspace = np.linspace(min(xdata), max(xdata), len(xdata) * 10)
        for x in x_linspace:
            list_y.append(self.gaussian(x, parameter[0], parameter[1], parameter[2]))

        fsize = 30
        fig = plt.figure(figsize=(18, 9))
        ax0 = fig.add_subplot(111)
        ax0.scatter(xdata[1:], ydata, label='mod_mean', color='darkcyan')
        ax0.plot(x_linspace, np.array(list_y), label='fitting', color='purple')
        ax0.set_xlabel('gray value', fontsize=fsize)
        ax0.set_ylabel('num of events', fontsize=fsize)
        ax0.legend(fontsize=fsize)
        ax0.tick_params(labelsize=fsize)
        ax0.set_title('(a, sigma, mean) = ({0}, {1}, {2})'.format(round(parameter[0], 4),
                                                                  round(parameter[1], 4),
                                                                  round(parameter[2], 4)),
                      fontsize=fsize)
        plt.tight_layout()
        plt.savefig(histogram_save_path + '/{0}_N{1}_hist.jpg'.format(os.path.basename(roi_csv_file)
                                                                      .replace('_roi_data.csv', ''), i + 1),
                    dpi=350)
        plt.close()
        print(histogram_save_path + '/{0}_N{1}_hist.png'.format(os.path.basename(roi_csv_file)
                                                                .replace('_roi_data.csv', ''), i + 1))
    def write_hist_sum_gausian(self, roi_csv_file, data, bin, parameter, histogram_save_path, i):
        """
        calc_sigma(data)で計算したパラメータを使用して、dataのヒストグラムとフィッティング曲線をプロットした
        グラフを生成し保存する関数
        :param roi_csv_file : dataが入っているroiの輝度データが入ったcsvファイル
        :param data: calc_sigma()で与えたフィッティングしたいデータの配列
        :param bin: ヒストグラムを取る時のbinsの値
        :param parameter: calc_sigma(data, bin)の結果
        :param histogram_save_path: 生成したヒストグラムを保存するパス
        :param i: 何番目のROIデータかを示すindex
        :return: ヒストグラムが生成・保存される
        """
        ydata, xdata = np.histogram(data, bins=bin)

        list_y = []
        x_linspace = np.linspace(min(xdata), max(xdata), len(xdata) * 10)
        for x in x_linspace:
            list_y.append(self.sum_gaussian(x, parameter[0], parameter[1], parameter[2],
                                            parameter[3], parameter[4], parameter[5]))

        fsize = 30
        fig = plt.figure(figsize=(18, 9))
        ax0 = fig.add_subplot(111)
        ax0.scatter(xdata[1:], ydata, label='mod_mean', color='darkcyan')
        ax0.plot(x_linspace, np.array(list_y), label='fitting', color='purple')
        ax0.set_xlabel('gray value', fontsize=fsize)
        ax0.set_ylabel('num of events', fontsize=fsize)
        ax0.legend(fontsize=fsize)
        ax0.tick_params(labelsize=fsize)
        ax0.set_title('(a, sigma, mean) = ({0}, {1}, {2}) + ({3}, {4}, {5})'.format(round(parameter[0], 4),
                                                                                    round(parameter[1], 4),
                                                                                    round(parameter[2], 4),
                                                                                    round(parameter[3], 4),
                                                                                    round(parameter[4], 4),
                                                                                    round(parameter[5], 4),
                                                                                    fontsize=fsize))
        plt.tight_layout()
        plt.savefig(histogram_save_path + '/{0}_N{1}_hist.jpg'.format(os.path.basename(roi_csv_file)
                                                                      .replace('_roi_data.csv', ''), i + 1),
                    dpi=350)
        plt.close()
        print(histogram_save_path + '/{0}_N{1}_hist.png'.format(os.path.basename(roi_csv_file)
                                                                .replace('_roi_data.csv', ''), i + 1))

    def write_hist2(self, dt, roi_csv_file, bin, histogram_save_path, baseline_save_path):
        """
        write_hist()を用いてroi_csv_fileの中にあるすべてのROIのベースライン補正後のデータのヒストグラムおよび
        フィッティング曲線をプロットしたグラフを生成する
        :param dt: イメージングの際のフレームレートの逆数
        :param roi_csv_file: make_roi_csv()で作成したcsvファイル
        :param bin: ヒストグラムを取る時のbinsの値
        :param histogram_save_path: 生成したヒストグラムを保存するパス
        :return: すべてのROIのヒストグラムが生成・保存される
        """
        print('\n########### write histograms ###########\n')
        # mod_mean = self.modify_baseline(roi_csv_file, dt)
        mod_mean = self.modify_baseline_window(roi_csv_file, baseline_save_path, dt, 25)

        for i in range(0, len(mod_mean)):
            self.write_hist(roi_csv_file, mod_mean[i], bin, self.calc_sigma(mod_mean[i], bin),
                            histogram_save_path, i)

    # ガウシアンフィッティング用
    def gaussian(self, x, a, sigma, mean):
        return a / np.sqrt(2.0*np.pi) / sigma * np.exp(-((x-mean)/sigma)**2/2)

    # 2つのガウシアンの足し合わせ
    def sum_gaussian(self, x, a1, sigma1, mean1, a2, sigma2, mean2):
        return self.gaussian(x, a1, sigma1, mean1) + self.gaussian(x, a2, sigma2, mean2)

    def raster_plot(self, result_csv_file, plot_save_path):
        """
        ラスタープロットを生成する関数
        :param result_csv_file: detect_firing()で作成した発火判定まで実施したcsvファイル
        :param plot_save_path: 作成したラスタープロットを保存するパス
        :return: ラスタープロットが生成・保存される
        """
        print('\n########### write raster plot ###########\n')
        df = pd.read_csv(result_csv_file)
        fsize = 30
        fig_raster = plt.figure(figsize=(18, 9))
        ax_raster = fig_raster.add_subplot(111)
        for j in range(1, int((len(df.columns) - 1) / 3) + 1):
            ax_raster.scatter(df['#Time (sec)'], df['firing_{}'.format(j)], marker='|', s=500, color='red')
        ax_raster.set_yticks(np.arange(1, int((len(df.columns) - 1) / 3) + 1))
        ax_raster.set_ylim(0.1, int((len(df.columns) - 1) / 3) + 0.9)
        ax_raster.set_ylabel('Cell No.', fontsize=fsize)
        ax_raster.set_xlabel('#Time (sec)', fontsize=fsize)
        plt.tick_params(labelsize=fsize)
        plt.savefig(fname=plot_save_path + '/' + os.path.basename(result_csv_file).replace('_firing_detect.csv', '')
                          + '_raster_plot.png',
                    dpi=350)
        plt.show()
        print(plot_save_path + '/' + os.path.basename(result_csv_file).replace('_firing_detect.csv', '')
              + '_raster_plot.png')

    def mod_mean_and_threshold_plot(self, result_csv_file, plot_save_path):
        """
        ベースラインを補正した平均輝度データであるmod_meanと閾値であるthresholdのグラフを生成する関数
        :param result_csv_file: detect_firing()で作成した発火判定まで実施したcsvファイル
        :param plot_save_path: 作成したグラフを保存するパス
        :return: グラフが生成・保存される
        """
        print('\n########### write mod_mean plot ###########\n')
        df = pd.read_csv(result_csv_file)
        fsize = 30
        for i in range(0, int((len(df.columns) - 1) / 3)):
            fig = plt.figure(figsize=(18, 9))
            ax0 = fig.add_subplot(111)
            ax0.plot(df['#Time (sec)'], df['mod_mean_{}'.format(i + 1)], label='mod_mean_{}'.format(i + 1))
            ax0.plot(df['#Time (sec)'], df['threshold_{}'.format(i + 1)], label='threshold_{}'.format(i + 1), lw=3)
            ax0.set_xlabel('#Time (sec)', fontsize=fsize)
            ax0.set_ylabel('gray value', fontsize=fsize)
            # グラフのレンジを指定
            ax0.set_ylim(-500, 2500)
            plt.tick_params(labelsize=fsize)
            plt.legend(fontsize=fsize)
            plt.tight_layout()
            plt.savefig(fname=plot_save_path + '/' + os.path.basename(result_csv_file).replace('_firing_detect.csv', '')
                              + '_N{}_mod_mean_plot.png'.format(i + 1),
                        dpi=350)
            plt.close()
            # plt.show()
            print(plot_save_path + '/' + os.path.basename(result_csv_file).replace('_firing_detect.csv', '')
                  + '_N{}_mod_mean_plot.png'.format(i + 1))
        print()

    def detect_firing(self, roi_csv_file, dt, result_csv_save_path, window, bin, baseline_save_path):
        """
        make_roi_csv()で作成したROIのデータが入ったcsvファイルを用い、輝度の平均データから発火を判定する
        modify_baseline()を用いて平均輝度データのベースラインを補正し、
        calc_sigma()を用いて発火していないときの輝度データの標準偏差を求める
        求めた標準偏差の3倍を閾値とし、閾値を越え、かつ、輝度データが上昇を続けるとき(立ち上がりの部分)発火と判定する
        発火していない→0
        発火している→Cell No.
        :param roi_csv_file: make_roi_csv()で作成したROIのデータのcsvファイル
        :param dt: 時間列を作るために与える引数。イメージングの際のフレームレートの逆数
        :param result_csv_save_path: ベースラインを補正した輝度データ、閾値、発火判定のデータを
                書き込んだcsvファイルを保存するためのパス
        :param window: modify_baseline_window()で使用する最小値を取るための時間窓 [sec]
        :param bin: ヒストグラムを取る時のbinsの値
        :return: 発火判定まで実施したcsvファイルが出力される
        """
        print('\n########### make firing detect csv file ###########\n')
        mod_mean = self.modify_baseline_window(roi_csv_file, baseline_save_path, dt, window)
        time = np.arange(0, len(mod_mean[0]) * dt, dt)

        df_result = pd.DataFrame({'#Time (sec)': time})

        cell_num = len(mod_mean)
        firing = np.zeros((cell_num, len(mod_mean[0])))

        for i in range(0, cell_num):
            print(self.calc_sigma(mod_mean[i], bin))
            threshold = 3 * self.calc_sigma(mod_mean[i], bin)[1] + self.calc_sigma(mod_mean[i], bin)[2]
            if max(mod_mean[i]) > 1000:
                for step in range(0, len(mod_mean[i]) - 1):
                    if threshold < mod_mean[i][step] < mod_mean[i][step + 1]:
                        firing[i][step] = i + 1
            else:
                print('Cell{} is not firing.'.format(i))

            df_result['mod_mean_{}'.format(i + 1)] = mod_mean[i]
            df_result['threshold_{}'.format(i + 1)] = threshold * np.ones(len(mod_mean[i]))
            df_result['firing_{}'.format(i + 1)] = firing[i]

        df_result.to_csv(result_csv_save_path + '/' + os.path.basename(roi_csv_file).replace('_roi_data.csv', '')
                         + '_firing_detect.csv')
        print('firing detect data is saved as :')
        print(result_csv_save_path + '/' + os.path.basename(roi_csv_file).replace('_roi_data.csv', '')
              + '_firing_detect.csv\n')

    def network_analysis(self, result_csv_file, CellPercent, dt, thres_burst_time):
        """
        ネットワークバーストの頻度・周波数・IBIを計算する関数
        :param result_csv_file: ベースラインを補正した輝度データ、閾値、発火判定のデータを書き込んだcsvファイル
        :param CellPercent: 存在する細胞の内、cell_percent以上の割合の細胞が同時に発火したときに
                              バースト発火の条件を満たしうる
        :param dt: イメージングの際のフレームレートの逆数
        :param thres_burst_time: 存在する細胞の内、cell_percent以上の割合の細胞が同時に発火し、かつ、
                                　thres_burst_time [sec]以上発火するとバースト発火としてみなす
        :return: 計算したネットワークバーストの頻度・周波数・IBIを表示する
        """
        print('\n########### network burst analysis ###########\n')
        df = pd.read_csv(result_csv_file)
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
            tmp = 0
            for i in range(0, cell_num):
                if firing[i, step] != 0:
                    tmp += 1
            cnt[step * dt] = tmp  # 同じ時刻に発火している細胞数

        print('time [sec]\tsum(firing cell)')
        for t in sorted(cnt.keys()):
            if cnt[t] >= cell_num * CellPercent:
                if t - prev_time > dt:
                    # 連続しているバーストは1つとして数える
                    print('{:.3f}\t\t{}'.format(t, cnt[t]))
                    burst_event += 1
                    if prev_time != 0:
                        IBI.append(t - prev_time)
                prev_time = t

        for i in range(0, cell_num):
            burst_flag = 0
            for step in range(0, len(df)):
                if firing[i, step] != 0:  # バーストの判定方法が山本先生のプログラムとは異なる
                    burst_flag += 1
                elif burst_flag != 0:  # 発火が止まったとき
                    if burst_flag * dt >= thres_burst_time:
                        burst_duration.append(burst_flag * dt)
                    burst_flag = 0

        print()
        print('ネットワークバーストの頻度 : {:.3f}秒間に{}回\n'.format(dt * (len(df) - 1), burst_event))
        print('ネットワークバーストの周波数 : {:.2f}×10^-3 Hz\n'.format(burst_event / dt / (len(df) - 1) * 1000))

        if len(burst_duration) >= 1:
            print('個別のバースト発火の平均持続時間(±標準偏差) : {:.3f}秒(±{:.3f}秒) ;'
                  ' n_burst = {}\n'.format(np.mean(burst_duration), statistics.pstdev(burst_duration),
                                           len(burst_duration)))

        if len(IBI) >= 1:
            print('ネットワークバーストの平均間隔(IBI)(±標準偏差) : {:.3f}秒(±{:.3f}秒) ;'
                  ' n_IBI = {}\n'.format(np.mean(IBI), statistics.pstdev(IBI), len(IBI)))
            print('\n')

    def all_analysis(self, dt, x, y, roi_num, fig_save_path, csv_save_path, window, bin, CellPercent, thres_burst_time):
        """
        解凍したcxdファイルを用意して、
        イメージング時のフレームレートの逆数、ROIのcenterとなる座標、ROIの形、
        生成したグラフを保存するパス、生成したcsvファイルを保存するパスを引数に渡すことでラスタープロット生成および
        ネットワークバーストの解析まで行う
        :param dt: イメージングの際のフレームレートの逆数
        :param x: 指定したROIのcenterのx座標のみが入った配列 (input_roi_centers())
        :param y: 指定したROIのcenterのy座標のみが入った配列 (input_roi_centers())
        :param roi_num: どのようなROIを取るかを指定する番号 (input_roi_centers())
        :param fig_save_path: 生成した各種グラフを保存するパス
        :param csv_save_path: 生成した各種csvファイルを保存するパス
        :param window: modify_baseline_window()で使用する最小値を取るための時間窓 [sec]
        :param bin: ヒストグラムを取る時のbinsの値
        :param CellPercent: 存在する細胞の内、cell_percent以上の割合の細胞が同時に発火したときに
                              バースト発火の条件を満たしうる (network_analysis())
        :param thres_burst_time: 存在する細胞の内、cell_percent以上の割合の細胞が同時に発火し、かつ、
                                　thres_burst_time [sec]以上発火するとバースト発火としてみなす (network_analysis())
        :return:
        """
        self.input_roi_centers(x, y, roi_num, fig_save_path)
        self.make_roi_csv(csv_save_path)
        self.detect_firing(roi_csv_file=csv_save_path + '/' + os.path.basename(self.path) + '_roi_data.csv',
                           dt=dt, result_csv_save_path=csv_save_path, window=window, bin=bin,
                           baseline_save_path=fig_save_path)
        self.write_hist2(dt=dt, roi_csv_file=csv_save_path + '/' + os.path.basename(self.path) + '_roi_data.csv',
                         bin=bin, histogram_save_path=fig_save_path, baseline_save_path=fig_save_path)
        self.mod_mean_and_threshold_plot(result_csv_file=csv_save_path + '/'
                                                         + os.path.basename(self.path).replace('_roi_data.csv', '')
                                                         + '_firing_detect.csv', plot_save_path=fig_save_path)
        self.network_analysis(result_csv_file=csv_save_path + '/' + os.path.basename(self.path)
                              .replace('_roi_data.csv', '') + '_firing_detect.csv', CellPercent=CellPercent, dt=dt,
                              thres_burst_time=thres_burst_time)
        self.raster_plot(result_csv_file=csv_save_path + '/' + os.path.basename(self.path).replace('_roi_data.csv', '')
                         + '_firing_detect.csv', plot_save_path=fig_save_path)

    def analysis_after_output_roi_csv(self, dt, roi_csv_file, csv_save_path, window, bin, fig_save_path,
                                      CellPercent, thres_burst_time):
        self.detect_firing(roi_csv_file=roi_csv_file,
                           dt=dt, result_csv_save_path=csv_save_path, window=window, bin=bin,
                           baseline_save_path=fig_save_path)
        self.write_hist2(dt=dt, roi_csv_file=roi_csv_file, bin=bin, histogram_save_path=fig_save_path,
                         baseline_save_path=fig_save_path)
        self.mod_mean_and_threshold_plot(result_csv_file=csv_save_path + '/'
                                                         + os.path.basename(roi_csv_file).replace('_roi_data.csv', '')
                                                         + '_firing_detect.csv', plot_save_path=fig_save_path)
        self.network_analysis(result_csv_file=csv_save_path + '/' + os.path.basename(roi_csv_file)
                              .replace('_roi_data.csv', '') + '_firing_detect.csv', CellPercent=CellPercent, dt=dt,
                              thres_burst_time=thres_burst_time)
        self.raster_plot(result_csv_file=csv_save_path + '/' +os.path.basename(roi_csv_file).replace('_roi_data.csv', '')
                                         + '_firing_detect.csv', plot_save_path=fig_save_path)

def main():
    start_time = time.time()
    cxd_data_path = 'C:/Users/6969p/Downloads/experimental_data/20190610/Data282.cxd'
    csv_save_path = 'C:/Users/6969p/Downloads/experimental_data/20190610'
    fig_save_path = 'C:/Users/6969p/Downloads/experimental_data/20190610'

    dt = 0.025  # [sec]

    x = np.array([114])
    y = np.array([102])

    if not os.path.isdir(csv_save_path):
        os.mkdir(csv_save_path)
    if not os.path.isdir(fig_save_path):
        os.mkdir(fig_save_path)

    c = Ca_analyze(path=cxd_data_path, width=168, height=128)

    """
    c.all_analysis(dt=dt, x=x, y=y, roi_num=1, fig_save_path=fig_save_path, csv_save_path=csv_save_path, window=25,
                   bin=50, CellPercent=0.75, thres_burst_time=6)
    """
    c.analysis_after_output_roi_csv(dt=dt, roi_csv_file='C:/Users/admin/Desktop/2019.7.2_100_analyze/Data397_roi_data.csv',
                                    csv_save_path=csv_save_path, fig_save_path=fig_save_path, window=25, bin=50,
                                    CellPercent=0.75, thres_burst_time=6)

    # c. plot_heatmap(fig_save_path)
    # c.modify_baseline_window('C:/Users/admin/Desktop/190618/190626/Data347_roi_data.csv', dt, 25)
    elapsed_time = time.time() - start_time
    print('elapsed time : {} sec'.format(round(elapsed_time, 1)))


if __name__ == '__main__':
    main()

