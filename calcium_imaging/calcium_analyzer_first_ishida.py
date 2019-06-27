"""
author : ishida
date : 20190127
Original program is written by Mr. Yamamoto
"""

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import statistics

class Calcium_analyze():
    def __init__(self, path, file_noise, file_main,
                 time_step=0.2, alpha=0.5, tau1=0.8, tau2=120, thres_noise=0.175,
                 cell_percent=0.25, thres_burst_time=15):
        self.path = path
        self.file_noise = file_noise
        self.file_main = file_main
        self.time_step = time_step  # sec
        self.alpha = alpha
        self.tau1 = tau1  # sec
        self.tau2 = tau2  # sec
        self.thres_noise = thres_noise  # sec^-1

        """
        Network burst definition 
            25%以上の細胞が3秒以上同時に発火
        """
        self.cell_percent = cell_percent
        self.thres_burst_time = thres_burst_time  # n * time_step [sec]

        self.file_name_noise = os.path.basename(self.file_noise).replace('.csv', '')
        self.df_noise = pd.read_csv(self.file_noise, header=0)
        self.num_cells_noise = int((len(self.df_noise.columns) - 1) / 4)  # number of cells (1細胞4データ+行数を表す列)
        self.num_rows_noise = len(self.df_noise)

        self.list_noise = []
        self.time_noise = np.zeros(self.num_rows_noise)
        self.mean_data_noise = np.zeros((self.num_rows_noise, self.num_cells_noise))
        self.smoothed_data_noise = np.zeros((self.num_rows_noise, self.num_cells_noise))
        self.baseline_data_noise = np.zeros((self.num_rows_noise, self.num_cells_noise))
        self.relative_intensity_noise = np.zeros((self.num_rows_noise, self.num_cells_noise))
        self.moving_average_noise = np.zeros((self.num_rows_noise, self.num_cells_noise))
        self.diff_data_noise = np.zeros((self.num_rows_noise, self.num_cells_noise))

        self.file_name_main = os.path.basename(self.file_main).replace('.csv', '')
        self.df_main = pd.read_csv(self.file_main, header=0)
        self.num_cells_main = int((len(self.df_main.columns) - 1) / 4)
        self.num_rows_main = len(self.df_main)

        self.list_main = []
        self.time_main = np.zeros(self.num_rows_main)
        self.mean_data_main = np.zeros((self.num_rows_main, self.num_cells_main))
        self.smoothed_data_main = np.zeros((self.num_rows_main, self.num_cells_main))
        self.baseline_data_main = np.zeros((self.num_rows_main, self.num_cells_main))
        self.relative_intensity_main = np.zeros((self.num_rows_main, self.num_cells_main))
        self.moving_average_main = np.zeros((self.num_rows_main, self.num_cells_main))
        self.diff_data_main = np.zeros((self.num_rows_main, self.num_cells_main))

    def run_noise(self):
        print('##########  NOISE  ##########')
        print()
        print('設定されたパラメータ : ')
        print('     撮影間隔 = ' + str(self.time_step) + ' [sec]')
        print('     平滑化処理の時定数 tau1 = ' + str(self.tau1) + ' [sec]')
        print('     ベースラインの更新に使う時定数 tau2 = ' + str(self.tau2) + ' [sec]')
        print('     スパイク列のラスタープロットを作成する際の閾値 = ' + str(self.thres_noise) + ' [sec^-1]')
        print()

        """
        入力ファイルからMeanのデータだけを抽出する
        """

        print('ステップ1: 平均輝度データの抽出中')
        print()

        for i in range(0, self.num_rows_noise):
            self.time_noise[i] = i * self.time_step

        for i in range(0, self.num_cells_noise):
            tmp1 = self.df_noise['Mean' + str(i + 1)].values
            for j in range(0, len(self.df_noise)):
                self.mean_data_noise[j][i] = tmp1[j]

        """
        平滑化したF(x)を計算
        """
        print('ステップ2: 輝度シグナルの平滑化処理中')
        print()

        tau1_steps = self.tau1 / self.time_step

        for i in range(0, self.num_cells_noise):
            for j in range(0, self.num_rows_noise):
                if tau1_steps / 2 < j < self.num_rows_noise - tau1_steps / 2:
                    k = - tau1_steps / 2

                    while k <= (tau1_steps / 2):
                        self.smoothed_data_noise[j][i] += self.mean_data_noise[j + int(k)][i] / (tau1_steps + 1)
                        k += 1

        """
        Baseline F0(t)の算出
        """
        print('ステップ3: ベースライン輝度の計算中 (少々お待ちください)')
        print()

        tau2_steps = self.tau2 / self.time_step

        for i in range(0, self.num_cells_noise):
            for j in range(1, self.num_rows_noise):
                if self.smoothed_data_noise[j][i] != 0:
                    k = j - tau2_steps / 2
                    while k <= j + tau2_steps / 2:
                        if 1 <= k < len(self.mean_data_noise):  # 時間 t > 0 について計算する !!!上から押さえないと動かない!!!どこか間違ってる?
                            tmp = self.mean_data_noise[int(k)][i]
                            if tmp != 0:
                                if k == j - tau2_steps / 2:
                                    self.baseline_data_noise[j][i] = tmp
                                elif tmp < self.baseline_data_noise[j][i]:
                                    self.baseline_data_noise[j][i] = tmp
                        k += 1

        """
        輝度の相対変化を計算
        """
        print('ステップ4: 相対輝度変化を計算中')
        print()

        for i in range(0, self.num_cells_noise):
            for j in range(0, self.num_rows_noise):
                if self.smoothed_data_noise[j][i] != 0:
                    F_t = self.mean_data_noise[j][i]
                    F0_t = self.baseline_data_noise[j][i]
                    if F0_t != 0:
                        self.relative_intensity_noise[j][i] = (F_t - F0_t) / F0_t

        # 平滑化: Exponentially-weighted moving average
        ema = pd.DataFrame(self.relative_intensity_noise).ewm(adjust=False, alpha=self.alpha).mean()

        df_mean = pd.DataFrame({'Time [sec]': self.time_noise,
                                'Mean1': self.mean_data_noise[:, 0],
                                'Smoothed1': self.smoothed_data_noise[:, 0],
                                'F_0 Cell1': self.baseline_data_noise[:, 0],
                                'RFU Cell1': self.relative_intensity_noise[:, 0],
                                'sm-RFU Cell1': np.ravel(np.array(ema))})
        # df_mean.to_csv(self.path + '/' + self.file_name_noise + '-mean.csv', mode='a')

        """
        一次微分の計算
        """
        print('ステップ5: 相対輝度変化の時間微分を計算中')
        print()

        for i in range(0, self.num_cells_noise):
            for j in range(0, self.num_rows_noise):
                if i == 0:
                    self.diff_data_noise[j][i] = self.mean_data_noise[j][i]
                else:
                    if (self.mean_data_noise[j][i] == 0) or (self.mean_data_noise[j-1][i] == 0) or (self.mean_data_noise[j+1][i] == 0):
                        self.diff_data_noise[j][i] = 0
                    else:
                        self.diff_data_noise[j][i] = (self.mean_data_noise[j+1][i] - self.mean_data_noise[j-1][i]) / (self.time_step * 2)
                        # 中心微分
        df_diff = pd.DataFrame({'d RFU Cell1 / dt': self.diff_data_noise[:, 0]})
        # df_diff.to_csv(self.path + '/' + self.file_name_noise + '-diff.csv', mode='a')

    def run(self):
        print()
        print()
        print('##########  MAIN  ##########')
        print()

        stdev = statistics.stdev(self.diff_data_noise[:, 0])

        self.thres_noise = stdev * 2.58

        print('設定されたパラメータ : ')
        print('     撮影間隔 = ' + str(self.time_step) + ' [sec]')
        print('     平滑化処理の時定数 tau1 = ' + str(self.tau1) + ' [sec]')
        print('     ベースラインの更新に使う時定数 tau2 = ' + str(self.tau2) + ' [sec]')
        print('     スパイク列のラスタープロットを作成する際の閾値 = ' + str(self.thres_noise) + ' [sec^-1]')
        print()

        """
        入力ファイルからMeanのデータだけを抽出する
        """

        print('ステップ1: 平均輝度データの抽出中')
        print()

        for i in range(0, self.num_rows_main):
            self.time_main[i] = i * self.time_step

        for i in range(0, self.num_cells_main):
            tmp1 = self.df_main['Mean' + str(i + 1)].values
            for j in range(0, len(self.df_main)):
                self.mean_data_main[j][i] = tmp1[j]

        """
        平滑化したF(x)を計算
        """
        print('ステップ2: 輝度シグナルの平滑化処理中')
        print()

        tau1_steps = self.tau1 / self.time_step

        for i in range(0, self.num_cells_main):
            for j in range(0, self.num_rows_main):
                if tau1_steps / 2 < j < self.num_rows_main - tau1_steps / 2:
                    k = - tau1_steps / 2

                    while k <= (tau1_steps / 2):
                        self.smoothed_data_main[j][i] += self.mean_data_main[j + int(k)][i] / (tau1_steps + 1)
                        k += 1

        """
        Baseline F0(t)の算出
        """
        print('ステップ3: ベースライン輝度の計算中 (少々お待ちください)')
        print()

        tau2_steps = self.tau2 / self.time_step

        for i in range(0, self.num_cells_main):
            for j in range(1, self.num_rows_main):
                if self.smoothed_data_main[j][i] != 0:
                    k = j - tau2_steps / 2
                    while k <= j + tau2_steps / 2:
                        if 1 <= k < len(self.mean_data_main):  # 時間 t > 0 について計算する !!!上から押さえないと動かない!!!どこか間違ってる?
                            tmp = self.mean_data_main[int(k)][i]
                            if tmp != 0:
                                if k == j - tau2_steps / 2:
                                    self.baseline_data_main[j][i] = tmp
                                elif tmp < self.baseline_data_main[j][i]:
                                    self.baseline_data_main[j][i] = tmp
                        k += 1

        """
        輝度の相対変化を計算
        """
        print('ステップ4: 相対輝度変化を計算中')
        print()

        for i in range(0, self.num_cells_main):
            for j in range(0, self.num_rows_main):
                if self.smoothed_data_main[j][i] != 0:
                    F_t = self.mean_data_main[j][i]
                    F0_t = self.baseline_data_main[j][i]
                    if F0_t != 0:
                        self.relative_intensity_main[j][i] = (F_t - F0_t) / F0_t

        # 平滑化: Exponentially-weighted moving average
        ema = pd.DataFrame(self.relative_intensity_main).ewm(adjust=False, alpha=self.alpha).mean()

        df_mean = pd.DataFrame({'Time [sec]': self.time_main,
                                'Mean1': self.mean_data_main[:, 0],
                                'Smoothed1': self.smoothed_data_main[:, 0],
                                'F_0 Cell1': self.baseline_data_main[:, 0],
                                'RFU Cell1': self.relative_intensity_main[:, 0],
                                'sm-RFU Cell1': np.ravel(np.array(ema))})
        # df_mean.to_csv(self.path + '/' + self.file_name_main + '-mean.csv', mode='a')

        """
        一次微分の計算
        """
        print('ステップ5: 相対輝度変化の時間微分を計算中')
        print()

        for i in range(0, self.num_cells_main):
            for j in range(0, self.num_rows_main):
                if i == 0:
                    self.diff_data_main[j][i] = self.mean_data_main[j][i]
                else:
                    if (self.mean_data_main[j][i] == 0) or (self.mean_data_main[j-1][i] == 0) or (self.mean_data_main[j+1][i] == 0):
                        self.diff_data_main[j][i] = 0
                    else:
                        self.diff_data_main[j][i] = (self.mean_data_main[j+1][i] - self.mean_data_main[j-1][i]) / (self.time_step * 2)
                        # 中心微分
        df_diff = pd.DataFrame({'d RFU Cell1 / dt': self.diff_data_main[:, 0]})
        # df_diff.to_csv(self.path + '/' + self.file_name_main + '-diff.csv', mode='a')

        """
        グラフの表示
        """
        fig1 = plt.figure(figsize=(18, 9))
        ax1 = fig1.add_subplot(111)
        ax1.plot(self.time_main, self.relative_intensity_main[:, 0])

        fig2 = plt.figure(figsize=(18, 9))
        ax2 = fig2.add_subplot(111)
        ax2.plot(self.time_main, self.diff_data_main[:, 0])
        plt.show()

        """
        ラスタープロット用のcsvデータの作製と書き出し
        """
        temp0 = []
        temp1 = []
        burst_duration = []
        # diff_gnu = -1 * np.ones(4)
        diff_gnu = []
        k = 0
        for i in range(0, self.num_cells_main):
            flag = 0
            for j in range(0, self.num_rows_main):
                if (flag == 0 and self.diff_data_main[j][i]>self.thres_noise) or (flag >= 0 and self.diff_data_main[j][i]>0):
                    flag += 1
                    temp0.append(self.time_main[j])
                    temp1.append(i)
                else:
                    if flag != 0:
                        if flag >= self.thres_burst_time:
                            duration = flag * self.time_step
                            burst_duration.append(duration)

                            for l in range(0, len(temp0)):
                                # diff_gnu = np.append(diff_gnu, np.array([temp0[l], temp1[l], self.time_step/2 - 0.05, 0.3]), axis=0)

                                diff_gnu[k][0] = temp0[l]
                                diff_gnu[k][1] = temp1[l]
                                diff_gnu[k][2] = self.time_step/2 - 0.05
                                diff_gnu[k][3] = 0.3

                                k += 1

                        flag = 0
                        temp0 = []
                        temp1 = []
        print(diff_gnu)

        df_diff_gnu = pd.DataFrame({'firing time [sec]': np.array(diff_gnu[:, 0]),
                                    'Cell No.': np.array(diff_gnu[:, 1]),
                                    'x-axis error range': np.array(diff_gnu[:, 2]),
                                    'y-axis error range': np.array(diff_gnu[:, 3])})
        # df_diff_gnu.to_csv(self.path + '/' + self.file_name_main + '-gnu.csv', mode='a')

        """
        Network burstの頻度・周波数・IBIを計算
        """
        burst_time = np.ravel(diff_gnu[:, 0])
        burst_event = 0
        prev_time = 0
        IBI = []
        cnt = {}  # keyを使って扱いたいため辞書を用意

        for firing_time in burst_time:
            cnt[firing_time] = cnt[firing_time] + 1  # 同じkeyがあれば、値が1ずつ増加していく
            # cnt[firing_time]: 要素の重複個数(同時に発火した細胞の個数)

        # カウントした回数(値)をkeyとともに出力
        for firing_time in sorted(cnt.items()):
            # firing_time: 細胞が発火した時間
            # cnt[firing_time]: 要素の重複個数 (同時に発火した細胞の個数)
            if cnt[firing_time] >= self.num_cells_main * self.cell_percent:
                tmp = round(firing_time - prev_time, 2)
                if tmp > self.time_step:
                    print(firing_time + '    ' + cnt[firing_time])  # 発火タイミングの個別出力
                    burst_event += 1
                    if prev_time != 0:
                        IBI.append(tmp)
                prev_time = firing_time

        print()
        print('Network burstの頻度: ' + str(round(burst_event, 0)) + ' 回 (計測時間: ' + str(self.time_step * (self.num_rows_main - 2)) + ')')
        print('Network burstの周波数: ' + str(round((burst_event / (self.time_step * (self.num_rows_main - 2))) * 1000, 1)) + ' ×10^-3 Hz')

        if len(burst_duration) >= 1:
            print('個別のburstの平均持続時間 (±標準偏差): ' + str(round(statistics.mean(burst_duration), 2))
                  + ' (± ' + str(round(statistics.stdev(burst_duration), 2)) + ' [sec];  n = ' + str(len(burst_duration)))

        if len(IBI) >= 1:
            print('Network burstの平均間隔(IBI) (±標準偏差): ' + str(round(statistics.mean(IBI), 2))
                  + ' (± ' + str(round(statistics.stdev(IBI), 2)) + ' [sec];  n = ' + str(len(IBI)))

        """
        csvファイルに書き込み
        """
        df_result = pd.DataFrame({'num of Network burst': burst_event,
                                  'measurement time [sec]': self.time_step * (self.num_rows_main - 2),
                                  'freq of Network burst [Hz]': burst_event / (self.time_step * (self.num_rows_main - 2)),
                                  'average duration time of burst [sec]': statistics.mean(burst_duration),
                                  'stdev duration time of burst [sec]': statistics.stdev(burst_duration),
                                  'num of duration time of burst': len(burst_duration),
                                  'average time interval of Network burst (IBI) [sec]': statistics.mean(IBI),
                                  'stdev of IBI [sec]': statistics.stdev(IBI),
                                  'num of IBI': len(IBI)})
        df_result.to_csv(self.path + '/' + self.file_name_main + '_result.csv', mode='a')

        df_IBIandDuration = pd.DataFrame({'time interval of Network burst(IBI) [sec]': IBI,
                                          'duration time of burst [sec]': burst_duration})
        df_IBIandDuration.to_csv(self.path + '/' + self.file_name_main + '_IBIandDuration.csv', mode='a')

        """
        ラスタープロット作成
        """
        fsize = 15
        fig3 = plt.figure(figsize=(18, 9))
        ax3 = fig3.add_subplot(111)
        ax3.scatter(df_diff_gnu['firing_time [sec]'], df_diff_gnu['Cell No.'], marker='|')
        ax3.set_xlim(0, self.time_step * self.num_rows_main - 2)
        ax3.set_ylim(0.1, self.num_cells_main + 0.9)
        ax3.set_xlabel('Time [sec]', fontsize=fsize)
        ax3.set_ylabel('Neuron #', fontsize=fsize)
        ax3.tick_params(labelsize=fsize)
        plt.tight_layout()
        # plt.show()
        plt.savefig(fname=self.path + self.file_name_main + '.jpg', dpi=350)
        plt.close()

        """
        注意書きの出力
        """
        print()
        print('============== 注 意 ===============')
        print('Burstの検出方法 : ΔF/Fのグラフの時間微分が閾値(THRES_NOISE)以上になったタイミングから、ゼロに戻るまで')
        print('従って解析結果やラスタープロットはTHRES_NOISEの値に極めて強く依存するので、その設定には注意を有する')
        print()
        print('Network burstの定義 : 解析対象とした細胞の半数以上がTIME_STEP( ' + self.time_step + ' sec)以上、同時に活動していること')
        print('===================================')


def main():
    path = 'C:/Users/ishida/Desktop'
    file_noise = 'C:/Users/ishida/Desktop/Results.csv'
    file_main = 'C:/Users/ishida/Desktop/main.csv'
    c = Calcium_analyze(path, file_noise=file_noise, file_main=file_main)
    c.run_noise()
    c.run()


if __name__ == '__main__':
    main()
