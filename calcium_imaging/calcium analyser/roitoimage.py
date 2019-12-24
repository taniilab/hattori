# coding=utf-8
"""
created @Ittan_moment
update:20191113
roiを画像へアウトプット
"""

import numpy as np
import cv2
import os


class RoiToImage:
    def save_roi_to_image(data_name, input_img_path, fig_save_path, x, y,
                          width):

        read = cv2.imread(input_img_path)

        imwidth = read.shape[1]

        rate = float(imwidth / width)

        for i in range(len(x)):
            cv2.putText(read, str(i+1), (int(x[i] * rate + 10), int(y[i] * rate - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                        (0, 0, 40), thickness=3)
            cv2.putText(read, str(i+1), (int(x[i] * rate + 10), int(y[i] * rate - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                        (0, 255, 0), thickness=2)

            cv2.ellipse(read, (int(x[i] * rate), int(y[i] * rate)), (15, 15), 0, 0, 360, (0, 0, 40), thickness=3)
            cv2.ellipse(read, (int(x[i] * rate), int(y[i] * rate)), (15, 15), 0, 0, 360, (0, 255, 0), thickness=2)
        cv2.imshow('output', read)
        cv2.imwrite(fig_save_path + '/' +data_name+'_RoiMappingFromImage_' +
                    os.path.splitext(os.path.basename(input_img_path))[0]+'.png', read)

        while 1:
            cv2.waitKey(20)
            if cv2.getWindowProperty('output', 0) < 0:
                break




