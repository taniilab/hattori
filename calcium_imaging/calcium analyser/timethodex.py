# coding=utf-8
"""
date:190915
created by takahashi & ishida

Extended by @Ittan_moment
update:20191021
"""


import numpy as np
import time
import os
import core as cr
import pandas as pd
import setroipointgui as mr
import analysisfiring as af
from tkinter import filedialog
import roitoimage as rti
import re
import compoundfiles as cf

def main():
    start_time = time.time()
    print("takahashi_ishida_method Extended V2.0.1+11.2")
    print("created by takahashi & ishida")
    print("extended by kurakake")
    print("\n###########cxdファイルの読み込み###########\n")

    fTyp = [('cxdファイル', '*.cxd')]
    file_path = filedialog.askopenfilename(filetypes=fTyp)
    if file_path == '':
        raise FileNotFoundError('cxdファイルが見つかりません')
    data_path = file_path[:-4]
    data_name = re.search(r'/Data\d+',data_path).group()[1:]
    print("OK！今日解析するのは "+data_path+"！")
    csv_save_path = data_path + '/csv'
    fig_save_path = data_path + '/fig'
    roi_save_path = csv_save_path + '/' + os.path.basename(
            data_path) +'_roi_pos_data.csv'
    stim_save_path = csv_save_path + '/stim_data.csv'
    rec_time = 0

    # cxdファイルから録画時間及びBinning情報を抽出
    with cf.CompoundFileReader(file_path) as cxd:
        '''
        First Field Data & Time と Last Field Data & Timeのファイル構造
        addr    0   1   2   3   4   5   6   7
                ----------~~~~~~~~~~??========
        - : ミリ秒
        ~ : 秒?
        = : 日付?
        '''
        time_start_bin = cxd.open(cxd.root['File Info']['First Field Date & Time']).read()
        time_start = int((time_start_bin[5]<<24) + (time_start_bin[4]<<16) + (time_start_bin[3]<<8) + time_start_bin[2])
        time_end_bin = cxd.open(cxd.root['File Info']['Last Field Date & Time']).read()
        time_end = int((time_end_bin[5]<<24) + (time_end_bin[4]<<16) + (time_end_bin[3]<<8) + time_end_bin[2])
        rec_time = (time_end - time_start) / 8
        print('録画時間は{:.1f} secね'.format(rec_time))
        binning_data = cxd.open(
            cxd.root['Field Data']['Field 1']['Details']['Binning'])
        data = binning_data.read()
        input_data = int(data[6]/4)
        print("Binningは{}かな".format(input_data))

    if input_data == 1:
        width = 1348
        height = 1024
    elif input_data == 2:
        width = 672
        height = 512
    elif input_data == 4:
        width = 336
        height = 256
    elif input_data == 8:
        width = 168
        height = 128
    else:
        raise ValueError("Binning 読み込みエラー! 開発者へ連絡してください!")

    mk = mr.SetRoiPointGui(roi_save_path=roi_save_path, stim_save_path=stim_save_path,
                           cxd_path=file_path, fig_save_path=fig_save_path,
                           width=width, height=height,data_name = data_name)

    # パスが存在しない場合新たに作成する
    if not os.path.isdir(data_path):
        os.mkdir(data_path)
    if not os.path.isdir(csv_save_path):
        os.mkdir(csv_save_path)
    if not os.path.isdir(fig_save_path):
        os.mkdir(fig_save_path)
    if not os.path.isfile(roi_save_path):
        mk.make_roi_file_dialog()
    if not os.path.isfile(stim_save_path):
        mk.make_stim_file_dialog()

    # 他ファイルの読み込み
    mk.read_roi_and_stim_file()
    c = cr.AnalysisRunClass(csv_save_path=csv_save_path, graph_save_path=fig_save_path, rec_time=rec_time,
                            data_name = data_name, histogram_bins=100)
    x = mk.x
    y = mk.y
    z = mk.z
    print("\n###########解析モード選択###########\n")
    print("1:roiの解析+プロット\n2:roi解析済みデータを再プロット\n3:座標データを画像に出力")

    a = input(">>")

    if a == '1':
        c.analysis_from_unzipped_cxd_data(cxd_path=file_path,
                                      x=x,
                                      y=y,
                                      z=z,
                                      data_pixel_width=width,
                                      data_pixel_height=height)
    elif a == '2':
        roi_intensity_csv_file = csv_save_path + '/' + os.path.basename(
            data_path) + '_roi_data.csv'
        c.analysis_from_roi_intensity_csv(roi_intensity_csv_file=roi_intensity_csv_file,z=z)

    elif a == '3':
        image_path=filedialog.askopenfilename()
        rti.RoiToImage.save_roi_to_image(data_name = data_name, input_img_path=image_path,fig_save_path=fig_save_path,x=x,y=y,width=width)
    
    elapsed_time = time.time() - start_time
    print('elapsed time : {} sec'.format(round(elapsed_time, 1)))


if __name__ == '__main__':
    main()
