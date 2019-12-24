# coding=utf-8
"""
date:190915
created by takahashi & ishida

Extended by @Ittan_moment
update:20191021
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import os
import time
import compoundfiles as cf


class MakeRoiIntensityCSV:
    def __init__(self, cxd_path, data_name, x, y, data_pixel_width=336, data_pixel_height=256):
        self.cxd_path = cxd_path
        self.data_name = data_name
        self.num = 0
        self.data_pixel_height = data_pixel_height
        self.data_pixel_width = data_pixel_width
        self.roi = self.set_roi_area_pixels(x=x, y=y, roi_shape=1)
        self.name_of_roi_intensity_csv_file = ''

    def set_roi_area_pixels(self, x, y, roi_shape):
        centers = np.zeros(len(x))  # 指定したROIのcenterとなる(一次元の)番地を保存する配列
        for i in range(0, len(x)):
            print('center_{0}   x : {1},  y : {2}'.format(i, x[i], y[i]))  # ROIのx座標，y座標をprint
            centers[i] = y[i] * self.data_pixel_width + x[i]

        # roi_shape:
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
        roi = []
        if roi_shape == 0:
            roi = np.zeros((len(x), 13))
            for _center in range(0, len(x)):
                roi[_center] = np.array([centers[_center] - self.data_pixel_width * 2,

                                         centers[_center] - self.data_pixel_width - 1,
                                         centers[_center] - self.data_pixel_width,
                                         centers[_center] - self.data_pixel_width + 1,

                                         centers[_center] - 2, centers[_center] - 1, centers[_center],
                                         centers[_center] + 1, centers[_center] + 2,

                                         centers[_center] + self.data_pixel_width - 1,
                                         centers[_center] + self.data_pixel_width,
                                         centers[_center] + self.data_pixel_width + 1,

                                         centers[_center] + self.data_pixel_width * 2])

        """
                ROI image #1:
                ■□
                □□

                ■:center
        """
        if roi_shape == 1:
            roi = np.zeros((len(x), 4))
            for _center in range(0, len(x)):
                roi[_center] = np.array([centers[_center], centers[_center] + 1,

                                         centers[_center] + self.data_pixel_width,
                                         centers[_center] + self.data_pixel_width + 1])

        return roi

    def check_roi_areas_visually(self, heatmap_save_path):
        # ROIをヒートマップに表示して視覚的に確認する
        cxd_file = cf.CompoundFileReader(self.cxd_path)
        bitmap = cxd_file.open(cxd_file.root['Field Data']['Field 1']['i_Image1']['Bitmap 1'])
        data = bitmap.read()

        intensity = np.zeros(self.data_pixel_height*self.data_pixel_width*2)
        j = 0
        for i in range(0, int(len(data) / 2)):
            intensity[i] = (int.from_bytes([data[j], data[j + 1]], 'little'))
            j += 2

        tmp_intensity = (np.array(intensity) + 0) * 5
        for _center in range(0, len(self.roi)):
            for j in range(0, len(self.roi[0])):
                tmp_intensity[int(self.roi[_center, j])] = 4095

        cnt = 0
        map_roi = np.zeros((self.data_pixel_height, self.data_pixel_width))
        for i in range(0, self.data_pixel_height):
            for j in range(0, self.data_pixel_width):
                map_roi[i][j] = tmp_intensity[cnt]
                cnt += 1

        plt.figure(figsize=(12, 9))
        sns.heatmap(map_roi, square=True, vmax=4095, vmin=0)

        plt.savefig(heatmap_save_path + '/' + self.data_name + '_ROI.png', dpi=350)
        plt.close()
        # plt.show()
        print('ROI check heatmap is saved as :')
        print(heatmap_save_path + '/' + self.data_name + '_ROI.png')
        print()

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
        roi_len = len(self.roi)
        roi0_len = len(self.roi[0])
        roi_pixel_data = np.zeros((roi_len, len(self.roi[0]), 0))
        roi_mean_data = np.zeros((roi_len, 0))
        roi_min_data = np.zeros((roi_len, 0))
        roi_max_data = np.zeros((roi_len, 0))

        roi_pixel_data0 = np.zeros((roi_len, roi0_len, 1))
        roi_mean_data0 = np.zeros((roi_len, 1))
        roi_min_data0 = np.zeros((roi_len, 1))
        roi_max_data0 = np.zeros((roi_len, 1))

        with cf.CompoundFileReader(self.cxd_path) as cxd_file:
            f = 0
            st = time.time()
            while 1:

                try:
                    with cxd_file.open(cxd_file.root['Field Data']['Field {}'.format(f + 1)]['i_Image1']['Bitmap 1']) as bitmap:
                        data = bitmap.read()
                except:
                    break

                roi_pixel_data1 = roi_pixel_data0
                roi_mean_data1 = roi_mean_data0
                roi_min_data1 = roi_min_data0
                roi_max_data1 = roi_max_data0
                bitmap.close()
                if time.time() > st + 0.5:
                    print("\r{:>5} files processed".format(f + 1), end="")
                    st = time.time()
                for _center in range(0, roi_len):
                    for _roi_pixel in range(0, roi0_len):
                        # roi_pixel_data[_center, _roi_pixel, f] = list_1[int(self.roi[_center, _roi_pixel])]
                        j = int(self.roi[_center, _roi_pixel]) * 2
                        roi_pixel_data1[_center, _roi_pixel, 0] = int.from_bytes([data[j], data[j + 1]], 'little')

                    roi_mean_data1[_center, 0] = np.mean(roi_pixel_data1[_center, :, 0])
                    roi_min_data1[_center, 0] = min(roi_pixel_data1[_center, :, 0])
                    roi_max_data1[_center, 0] = max(roi_pixel_data1[_center, :, 0])

                roi_pixel_data = np.append(roi_pixel_data, roi_pixel_data1, axis=2)
                roi_mean_data = np.append(roi_mean_data, roi_mean_data1, axis=1)
                roi_min_data = np.append(roi_min_data, roi_min_data1, axis=1)
                roi_max_data = np.append(roi_max_data, roi_max_data1, axis=1)
                f += 1

            self.num = f
            print("\r{:>5} files processed".format(f))
            df_roi_data = pd.DataFrame({'Area1': len(self.roi[0]) * np.ones(len(roi_mean_data[0])),
                                        'Mean1': roi_mean_data[0],
                                        'Min1': roi_min_data[0],
                                        'Max1': roi_max_data[0]},
                                       columns=['Area1', 'Mean1', 'Min1', 'Max1'])

            for i in range(1, len(self.roi)):
                df_roi_data['Area{}'.format(i + 1)] = roi0_len * np.ones(len(roi_mean_data[0]))
                df_roi_data['Mean{}'.format(i + 1)] = roi_mean_data[i]
                df_roi_data['Min{}'.format(i + 1)] = roi_min_data[i]
                df_roi_data['Max{}'.format(i + 1)] = roi_max_data[i]


        self.name_of_roi_intensity_csv_file = csv_save_path + '/' \
                                              + self.data_name + '_roi_data.csv'
        df_roi_data.to_csv(self.name_of_roi_intensity_csv_file)
        print()
        print('ROI pixels data is saved as :')
        print(csv_save_path + '/' + self.data_name + '_roi_data.csv\n')


if __name__ == '__main__':
    c = MakeRoiIntensityCSV(cxd_path='E:/Temp/Data899.cxd',data_name='Data899',
                                 x=[161],
                                 y=[134],
                                 data_pixel_width=336,
                                 data_pixel_height=256)
    c.make_roi_csv(csv_save_path='E:/Temp')

