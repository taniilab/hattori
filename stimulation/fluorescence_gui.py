# -*- coding: utf-8 -*-
# Fluorescence Plotter for HCImage Live
# programmed by Kouhei Hattori of Waseda University

"""
After running the software, please enter "shift + F". Converts pixel values
on mouse cursor coordinates at the time of input to grayscale and plots them.
Currently, it is programmed for single cell and is not compatible with
multicellular plotting.
"""

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import numpy as np
import serial
from time import sleep
import pandas as pd
import datetime
import pyautogui as pag
import keyboard as kb

class Ui_MainWindow(object):
    timer: QTimer

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1800, 900)
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralWidget.sizePolicy().hasHeightForWidth())
        self.centralWidget.setSizePolicy(sizePolicy)
        self.centralWidget.setObjectName("centralWidget")

        self.splitter = QtWidgets.QSplitter(self.centralWidget)
        self.splitter.setGeometry(QtCore.QRect(0, 0, 1481, 941))
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName("splitter")

        # values and button
        self.splitter_left = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.treeWidget = QtWidgets.QTreeWidget(self.splitter_left)
        self.treeWidget.setAutoScrollMargin(22)
        self.treeWidget.setObjectName("treeWidget")
        self.item_0 = QtWidgets.QTreeWidgetItem(self.treeWidget)

        self.button = QtWidgets.QPushButton('Stimulation start')
        self.bfont = self.button.font()
        self.bfont.setPointSizeF(20)
        self.button.setFont(self.bfont)
        self.click_flg = False
        self.stim_flg = False
        self.button.setStyleSheet("background-color: rgb(230,230,230)")
        self.button.clicked.connect(self.on_click)

        self.splitter_left.addWidget(self.button)
        self.splitter.addWidget(self.splitter_left)

        # plotter
        self.tabWidget = QtWidgets.QTabWidget(self.splitter)
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.tabWidget.addTab(self.tab, "")

        # othors
        MainWindow.setCentralWidget(self.centralWidget)
        self.mainToolBar = QtWidgets.QToolBar(MainWindow)
        self.mainToolBar.setObjectName("mainToolBar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.mainToolBar)
        self.menuBar = QtWidgets.QMenuBar(MainWindow)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 1493, 26))
        self.menuBar.setObjectName("menuBar")
        self.menu = QtWidgets.QMenu(self.menuBar)
        self.menu.setObjectName("menu")
        MainWindow.setMenuBar(self.menuBar)
        self.menuBar.addAction(self.menu.menuAction())
        layout_main = QtWidgets.QHBoxLayout()
        layout_main.addWidget(self.splitter)
        self.centralWidget.setLayout(layout_main)

        # glaph setting
        self.pixel_pitch = 10
        self.x, self.y = 0, 0  # plot position
        self.index = np.arange(0, 1000)
        self.stim_data = np.zeros(len(self.index))
        self.flu_data = np.zeros(len(self.index))
        self.fig = Figure()
        self.fc = FigureCanvas(self.fig)
        self.ax1 = self.fig.add_subplot(1, 1, 1)
        self.ax2 = self.ax1.twinx()
        self.fc.draw()
        self.bg = self.fig.canvas.copy_from_bbox(self.ax1.bbox)
        self.fig.tight_layout()

        layout_glaph_tab = QtWidgets.QVBoxLayout()
        layout_glaph_tab.addWidget(self.fc)
        self.tab.setLayout(layout_glaph_tab)

        # value display initialization
        self.view_data_len = 30
        for i in range(0, self.view_data_len-1):
            self.item_0 = QtWidgets.QTreeWidgetItem(self.treeWidget)

        # plot interval setting
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.fluorescence_measurment)
        self.timer.start(33)

        # stimulation interval setting
        self.timer_stim = QtCore.QTimer()
        self.timer_stim.timeout.connect(self.stimulate)
        self.amplitude = 0
        self.stim_time = [0]
        self.counter = 0

        # keyboard input setting
        # When the shortcut key set here is input, the plot is made with pixels on cursor coordinates when input.
        kb.add_hotkey('shift+F', lambda: self.plot_position())

        # serial communication setting
        # 11520 kbps
        self.port_number = "COM11"
        self.ser = serial.Serial(self.port_number, 115200, timeout=1)
        print(str(self.port_number) + " Opened!!")
        self.tmp_counter = 0
        # FG initialization
        self.send_command("WMW34" + "\n")
        self.send_command("WMA0" + "\n")
        self.send_command("WMF200000" + "\n")#200mhz
        self.send_command("WMN1" + "\n")

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


    def on_click(self):
        if self.click_flg == False:
            self.click_flg = True
            self.stim_flg = True
            self.timer_stim.start(5000)  # 5s
            self.button.setStyleSheet("background-color: rgb(100,230,180)")
            self.button.setText("Stimulating ...")
        else:
            self.reset_stim_setting()


    def send_command(self, command):
        self.ser.write(command.encode())
        print(command)


    def stimulate(self):
        if self.amplitude == 10:
            self.reset_stim_setting()
        else:
            self.amplitude+=1
            self.send_command("WMA" + str(self.amplitude) + "\n")
            self.stim_time = np.append(self.stim_time, self.index[-1])


    def reset_stim_setting(self):
        self.amplitude = 0
        self.click_flg = False
        self.stim_flg = False
        self.timer_stim.stop()
        self.send_command("WMA0" + "\n")
        self.button.setStyleSheet("background-color: rgb(230, 230, 230)")
        self.button.setText("Stimulation start")


    def fluorescence_measurment(self):
        """
        # 2x2 pixels
        # use red value as grayscale value
        self.rgb1 = pag.pixel(self.x, self.y)
        self.rgb2 = pag.pixel(self.x+self.pixel_pitch, self.y)
        self.rgb3 = pag.pixel(self.x, self.y+self.pixel_pitch)
        self.rgb4 = pag.pixel(self.x+self.pixel_pitch, self.y+self.pixel_pitch)
        self.gray1 = self.rgb1[0]
        self.gray2 = self.rgb2[0]
        self.gray3 = self.rgb3[0]
        self.gray4 = self.rgb4[0]
        self.gray = (self.gray1+self.gray2+self.gray3+self.gray4)/4
        """
        """
        # accurately
        self.gray1 = (77*self.rgb1[0]+150*self.rgb1[1]+29*self.rgb1[2])/256
        self.gray2 = (77*self.rgb2[0]+150*self.rgb2[1]+29*self.rgb2[2])/256
        self.gray3 = (77*self.rgb3[0]+150*self.rgb3[1]+29*self.rgb3[2])/256
        self.gray4 = (77*self.rgb4[0]+150*self.rgb4[1]+29*self.rgb4[2])/256
        self.gray = (self.gray1+self.gray2+self.gray3+self.gray4)/4
        """
        # simple
        self.rgb = pag.pixel(self.x, self.y)
        self.gray = self.rgb[0]
        self.index = np.delete(self.index, 0)
        self.index = np.append(self.index, self.index[-1]+1)

        self.stim_data = np.delete(self.stim_data, 0)
        if self.stim_flg == True:
            self.stim_data = np.append(self.stim_data, 5)
            self.stim_flg = False
        else:
            self.stim_data = np.append(self.stim_data, 0)

        self.flu_data = np.delete(self.flu_data, 0)
        self.flu_data = np.append(self.flu_data, [self.gray])
        self.ax1.clear()
        self.ax2.clear()
        self.ax1.plot(self.index, self.flu_data, color="seagreen")
        self.ax2.plot([self.stim_time[-1], self.stim_time[-1]], [0, 5], "red", linestyle='dashed')
        self.ax1.set_ylim([0, 255])
        self.fc.draw()

        self.treeWidget.takeTopLevelItem(0)
        self.item_0 = QtWidgets.QTreeWidgetItem(self.treeWidget)
        self.treeWidget.topLevelItem(self.view_data_len-1).setText(0, str(self.rgb[0]))
        self.treeWidget.topLevelItem(self.view_data_len-1).setText(1, str(self.rgb[1]))
        self.treeWidget.topLevelItem(self.view_data_len-1).setText(2, str(self.rgb[2]))

        """
        df = pd.DataFrame(columns =
                          [self.vx1[self.counter],
                           self.vx2[self.counter],
                           self.vy1[self.counter],
                           self.vy2[self.counter],
                           self.x,
                           self.y])
        df.to_csv(save_path + self.filename, mode="a")
        """

    def plot_position(self):
        self.x, self.y = pag.position()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Fluorescence Plotter"))
        self.treeWidget.headerItem().setText(0, _translate("MainWindow", "R"))
        self.treeWidget.headerItem().setText(1, _translate("MainWindow", "G"))
        self.treeWidget.headerItem().setText(2, _translate("MainWindow", "B"))
        __sortingEnabled = self.treeWidget.isSortingEnabled()
        self.treeWidget.setSortingEnabled(False)
        self.treeWidget.topLevelItem(0).setText(0, _translate("MainWindow", "0"))
        self.treeWidget.topLevelItem(0).setText(1, _translate("MainWindow", "0"))
        self.treeWidget.topLevelItem(0).setText(2, _translate("MainWindow", "0"))
        self.treeWidget.setSortingEnabled(__sortingEnabled)
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "glaph"))
        self.menu.setTitle(_translate("MainWindow", "tanii lab"))
