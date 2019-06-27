"""
date: 20190611
created by: ishida
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import time
import glob
from PIL import Image
import matplotlib.animation as animation

class CXD_read:
    def __init__(self, path, width=336, height=256):
        self.path = path
        self.folders = path + '/Field Data/Field*'
        self.files = {}
        self.files = glob.glob(self.folders)
        self.height = height
        self.width = width

    def plot_heatmap(self, save_path):
        k = 1
        for f in range(0, len(self.files)):
            print(self.path + '/Field Data/Field {}/i_Image1/Bitmap 1'.format(k))
            bitmap = open(self.path + '/Field Data/Field {}/i_Image1/Bitmap 1'.format(k), 'rb')
            data = bitmap.read()

            list_1 = []
            j = 0
            for i in range(0, int(len(data)/2)):
                list_1.append(int.from_bytes([data[j], data[j + 1]], 'little'))
                j += 2

            cnt = 0
            map = np.zeros((self.height, self.width))
            for i in range(0, self.height):
                for j in range(0, self.width):
                    map[i][j] = int(list_1[cnt])
                    cnt += 1

            if k == 1:
                max_index_list = np.argmax(np.array(list_1))
                self.max_index_y = int(max_index_list / self.width)
                self.max_index_x = int(max_index_list - self.max_index_y * self.width)
                print(self.max_index_x, self.max_index_y)
            plt.figure(figsize=(12, 9))
            sns.heatmap(map, square=True, vmax=4025, vmin=100)
            plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
            plt.savefig(save_path + '/Field_{}.png'.format(k), dpi=350)
            plt.close()
            # plt.show()
            k += 1

    def select_roi(self):
        print(self.path + '/Field Data/Field 1/i_Image1/Bitmap 1')
        bitmap = open(self.path + '/Field Data/Field 1/i_Image1/Bitmap 1', 'rb')
        data = bitmap.read()

        list_1 = []
        j = 0
        for i in range(0, int(len(data)/2)):
            list_1.append(int.from_bytes([data[j], data[j + 1]], 'little'))
            j += 2


        max_index_list = np.argmax(np.array(list_1))
        self.max_index_y = int(max_index_list / self.width)
        self.max_index_x = int(max_index_list - self.max_index_y * self.width)
        print('max data index (x, y) = ({0}, {1})'.format(self.max_index_x, self.max_index_y))

        """
        ROI image:
        　　□
        　□□□
        □□□□□
        　□□□
        　　□
        """

        self.roi = np.array([max_index_list - self.width * 2,
                        max_index_list - self.width - 1, max_index_list - self.width, max_index_list - self.width + 1,
                        max_index_list - 2, max_index_list - 1, max_index_list, max_index_list + 1, max_index_list + 2,
                        max_index_list + self.width - 1, max_index_list + self.width, max_index_list + self.width + 1,
                        max_index_list + self.width * 2])
        tmp = np.array(list_1)
        for i in range(0, len(self.roi)):
            tmp[self.roi[i]] = 0

        cnt = 0
        map_roi = np.zeros((self.height, self.width))
        for i in range(0, self.height):
            for j in range(0, self.width):
                map_roi[i][j] = tmp[cnt]
                cnt += 1


        plt.figure(figsize=(12, 9))
        sns.heatmap(map_roi, square=True, vmax=4025, vmin=0)
        plt.show()

    def make_gif(self, png_path, png_save_path, gif_name):
        pngs = png_path + '/*.png'
        _pngs = {}
        _pngs = glob.glob(pngs)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.spines['right'].set_color('None')
        ax.spines['left'].set_color('None')
        ax.spines['top'].set_color('None')
        ax.spines['bottom'].set_color('None')
        ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
        ax.tick_params(bottom=False, left=False, right=False, top=False)
        ims = []

        k = 1
        for f in range(0, int(len(_pngs)/100)):
            print(png_path + '/Field_{}.png'.format(k))
            tmp = Image.open(png_path + '/Field_{}.png'.format(k))
            ims.append([plt.imshow(tmp, interpolation='spline36')])
            k += 100

        ani = animation.ArtistAnimation(fig, ims, interval=20)
        ani.save(png_save_path + '/' + gif_name, writer='imagemagick')



def main():
    start_time = time.time()
    data_path = '//192.168.13.10/Public/hattori/Data265'
    save_path = '//192.168.13.10/Public/hattori/Data265png'
    png_save_path = '//192.168.13.10/Public/hattori/Data265png/gif'
    c = CXD_read(data_path)
    # c.plot_heatmap(save_path)
    # c.make_gif(png_path=save_path, png_save_path=png_save_path, gif_name='Data183.gif')
    c.select_roi()
    elapsed_time = time.time() - start_time
    print('elapsed time : {} sec'.format(round(elapsed_time, 1)))


if __name__ == '__main__':
    main()
