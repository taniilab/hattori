# coding=utf-8
"""
created @Ittan_moment
update:20191113
roiをguiで指定します
"""

import numpy as np
import cv2
import csv
from tkinter import filedialog
import compoundfiles as cf


class MouseParam:
    # マウスの動作を返すやつ
    def __init__(self,input_img_name):
        self.mouseEvent = {"x":None, "y":None, "event":None, "flags":None}
        cv2.setMouseCallback(input_img_name, self.__CallBackFunc,None)

    def __CallBackFunc(self, eventType, x, y, flags, userdata):
        self.mouseEvent["x"] = x
        self.mouseEvent["y"] = y
        self.mouseEvent["event"] = eventType
        self.mouseEvent["flags"] = flags

    def getData(self):
        return self.mouseEvent

    def getEvent(self):
        return self.mouseEvent["event"]

    def getX(self):
        return self.mouseEvent["x"]

    def getY(self):
        return self.mouseEvent["y"]

    def getPos(self):
        return (self.mouseEvent["x"], self.mouseEvent["y"])


class SetRoiPointGui:
    def __init__(self, roi_save_path, stim_save_path, cxd_path, fig_save_path, width, height, data_name):
        self.roi_save_path = roi_save_path
        self.stim_save_path = stim_save_path
        self.fig_save_path = fig_save_path
        self.cxd_path = cxd_path
        self.width = width
        self.height = height
        self.data_name = data_name
        self.x = np.array([0])
        self.y = np.array([0])
        self.z = np.array([0])

    def read_roi_and_stim_file(self):
        print("座標指定ファイル読み込み:")

        with open(self.roi_save_path,'r') as f:
            reader = csv.reader(f)
            l = [row for row in reader]
        self.x = np.zeros(len(l)-1)
        self.y = np.zeros(len(l)-1)

        for i in range(len(l)-1):
            self.x[i] = int(float(l[i][0]))
            self.y[i] = int(float(l[i][1]))
        print(np.append(self.x.reshape(len(self.x),1),self.y.reshape(len(self.y),1),axis=1))

        print("刺激時間指定ファイル読み込み:")
        with open(self.stim_save_path,'r') as f:
            reader = csv.reader(f)
            for row in reader:
                l = row
        self.z = np.zeros(len(l))
        for i in range(len(l)):
            self.z[i] = int(l[i])
        print(self.z)
        return

    def make_roi_file_dialog(self):
        print("roi設定ファイルがないみたい！")
        print("roiを指定してね")
        print("まずcxdのbitmapをpngに変換するね")
        cxd_file = cf.CompoundFileReader(self.cxd_path)
        bitmap = cxd_file.open(cxd_file.root['Field Data']['Field 1']['i_Image1']['Bitmap 1'])
        data = bitmap.read()

        output = np.zeros((self.height, self.width, 3))
        j = 0

        for i in range(0, int(len(data) / 2)):
            intensity = int.from_bytes([data[j], data[j + 1]], 'little')
            output[int(i / self.width)][i % self.width][0] = min(intensity/2, 255)
            output[int(i / self.width)][i % self.width][1] = max(min((intensity-512)/2, 255),0)
            output[int(i / self.width)][i % self.width][2] = max(min((intensity-256)/2, 255),0)
            j += 2

        cv2.imwrite(self.fig_save_path+'/'+self.data_name+'RoiMapping.png', output)
        print("左クリックで座標選択、右クリックで終了、ホイールクリックで最後の座標を削除")
        read = cv2.imread(self.fig_save_path+'/'+self.data_name+'RoiMapping.png')

        window_name = "input window"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, read)
        mouseData = MouseParam(window_name)

        lastpoint = (0,0)
        num = 0
        outputxy = [[]]
        while 1:
            cv2.waitKey(20)
            event = mouseData.getEvent()
            if cv2.getWindowProperty(window_name, 0) < 0:
                break
            if event == cv2.EVENT_LBUTTONDOWN:
                point = mouseData.getPos()
                x = mouseData.getX()
                y = mouseData.getY()
                if lastpoint != point:
                    num+=1
                    lastpoint = point
                    xy = np.zeros((1,2))
                    xy[0][0]=x
                    xy[0][1]=y
                    cv2.putText(read, str(num), point, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), thickness=1)
                    cv2.ellipse(read, point, (5, 5), 0, 0, 360, (0, 255, 0))
                    cv2.imshow(window_name, read)
                    print(mouseData.getPos())
                    if(num!=1):
                        outputxy = np.concatenate([outputxy,xy],axis=0)
                    else:
                        outputxy = xy

            elif event == cv2.EVENT_MBUTTONDOWN:
                point = mouseData.getPos()
                x = mouseData.getX()
                y = mouseData.getY()
                if lastpoint != point:
                    lastpoint = point
                    if num > 0:
                        num-=1
                        outputxy = np.delete(outputxy,num,0)
                        read = cv2.imread(self.fig_save_path+'/'+self.data_name+'RoiMapping.png')
                        for i in range(num):
                            cv2.putText(read, str(i+1), (int(outputxy[i][0]), int(outputxy[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), thickness=1)
                            cv2.ellipse(read, (int(outputxy[i][0]), int(outputxy[i][1])), (5, 5), 0, 0, 360, (0, 255, 0))
                    else:
                        print("消すものがないよ？")
                    cv2.imshow(window_name,read)

            elif event == cv2.EVENT_RBUTTONDOWN:
                break

        cv2.destroyAllWindows()
        xy = np.zeros((1, 2))
        xy[0][0]=-1
        xy[0][1]=-1
        outputxy = np.concatenate([outputxy,xy],axis=0)
        cv2.imwrite(self.fig_save_path+'/'+self.data_name+'RoiMapping.png', read)
        with open(self.roi_save_path,'w', newline="") as f:
            writer = csv.writer(f)
            writer.writerows(outputxy)



    def make_stim_file_dialog(self):
        print("stim設定ファイルがないみたい")
        print("刺激導入時刻を指定してね")
        print("60,65,70,...のようにカンマ区切りで入力して。え？刺激を導入しない？じゃあそのままEnter")
        input_data = input(">>")
        if(input_data!=""):
            stim = np.array(input_data.split(","), dtype=np.int)
        else:
            stim = np.array([-1])
        with open(self.stim_save_path,'w', newline="") as f:
            writer = csv.writer(f)
            writer.writerow(stim)
        return
