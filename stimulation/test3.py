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
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
import os



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

        # value, path_box, and button

        self.splitter_left = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.treeWidget = QtWidgets.QTreeWidget(self.splitter_left)
        self.treeWidget.setAutoScrollMargin(22)
        self.treeWidget.setObjectName("treeWidget")
        self.item_0 = QtWidgets.QTreeWidgetItem(self.treeWidget)

        self.stim_amp_label = QtWidgets.QLabel("Standard amplitude (peak to peak) [V]")
        self.def_stim_amp = "1"
        self.stim_amp_line = QtWidgets.QLineEdit(self.def_stim_amp)
        self.layout_stim_amp = QtWidgets.QHBoxLayout()
        self.layout_stim_amp.addWidget(self.stim_amp_label)
        self.layout_stim_amp.addWidget(self.stim_amp_line)
        self.stim_amp_w = QtWidgets.QWidget()
        self.stim_amp_w.setLayout(self.layout_stim_amp)

        self.stim_count_label = QtWidgets.QLabel("Number of stimuli (仮)")
        self.def_stim_count = "5"
        self.stim_count_line = QtWidgets.QLineEdit(self.def_stim_count)
        self.layout_stim_count = QtWidgets.QHBoxLayout()
        self.layout_stim_count.addWidget(self.stim_count_label)
        self.layout_stim_count.addWidget(self.stim_count_line)
        self.stim_count_w = QtWidgets.QWidget()
        self.stim_count_w.setLayout(self.layout_stim_count)

        self.stim_deltaV_label = QtWidgets.QLabel("Amplification rate (仮 V)")
        self.def_stim_deltaV = "1"
        self.stim_deltaV_line = QtWidgets.QLineEdit(self.def_stim_deltaV)
        self.layout_stim_deltaV = QtWidgets.QHBoxLayout()
        self.layout_stim_deltaV.addWidget(self.stim_deltaV_label)
        self.layout_stim_deltaV.addWidget(self.stim_deltaV_line)
        self.stim_deltaV_w = QtWidgets.QWidget()
        self.stim_deltaV_w.setLayout(self.layout_stim_deltaV)

        self.stim_interval_label = QtWidgets.QLabel("Interval of stimuli (仮 ms)")
        self.def_stim_interval = "5000"
        self.stim_interval_line = QtWidgets.QLineEdit(self.def_stim_interval)
        self.layout_stim_interval = QtWidgets.QHBoxLayout()
        self.layout_stim_interval.addWidget(self.stim_interval_label)
        self.layout_stim_interval.addWidget(self.stim_interval_line)
        self.stim_interval_w = QtWidgets.QWidget()
        self.stim_interval_w.setLayout(self.layout_stim_interval)

        self.save_path_label = QtWidgets.QLabel("Save path")
        self.def_path = "G:/Stim_G/csvdata/"
        self.save_path_line = QtWidgets.QLineEdit(self.def_path)
        self.layout_save_path = QtWidgets.QHBoxLayout()
        self.layout_save_path.addWidget(self.save_path_label)
        self.layout_save_path.addWidget(self.save_path_line)
        self.save_path_w = QtWidgets.QWidget()
        self.save_path_w.setLayout(self.layout_save_path)

        self.stim_button = QtWidgets.QPushButton('Stimulate')
        self.bfont = self.stim_button.font()
        self.bfont.setPointSizeF(20)
        self.stim_button.setFont(self.bfont)
        self.click_flg = False
        self.stim_flg = False
        self.stim_button.setStyleSheet("background-color: rgb(230,230,230)")
        self.stim_button.clicked.connect(self.on_click_stimlate)

        self.com_button = QtWidgets.QPushButton('Connect')
        self.bfont = self.com_button.font()
        self.bfont.setPointSizeF(20)
        self.com_button.setFont(self.bfont)
        self.FG_connect_flg = False
        self.com_button.setStyleSheet("background-color: rgb(230,230,230)")
        self.com_button.clicked.connect(self.on_click_com)

        self.start_button = QtWidgets.QPushButton('Measure')
        self.bfont = self.stim_button.font()
        self.bfont.setPointSizeF(20)
        self.start_button.setFont(self.bfont)
        self.start_button.setStyleSheet("background-color: rgb(230,230,230)")
        self.start_button.clicked.connect(self.on_click_start)

        self.splitter_left.addWidget(self.stim_amp_w)
        self.splitter_left.addWidget(self.stim_count_w)
        self.splitter_left.addWidget(self.stim_deltaV_w)
        self.splitter_left.addWidget(self.stim_interval_w)
        self.splitter_left.addWidget(self.save_path_w)
        self.splitter_left.addWidget(self.stim_button)
        self.splitter_left.addWidget(self.com_button)
        self.splitter_left.addWidget(self.start_button)
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
        self.x, self.y = 1000, 1000  # plot position
        self.index = np.arange(0, 1000)
        self.stim_data = np.zeros(len(self.index))
        self.flu_data = np.zeros(len(self.index))

        # pyqtgraph
        self.glaph_tab = pg.GraphicsWindow(title="fluorescence")
        self.p1 = self.glaph_tab.addPlot()
        #self.p1.setXRange(0,5)
        self.curve1 = self.p1.plot(self.index, self.flu_data)
        self.stim_list = []
        layout_glaph_tab = QtWidgets.QVBoxLayout()
        layout_glaph_tab.addWidget(self.glaph_tab)
        self.tab.setLayout(layout_glaph_tab)

        # value display initialization
        self.view_data_len = 30
        for i in range(0, self.view_data_len-1):
            self.item_0 = QtWidgets.QTreeWidgetItem(self.treeWidget)

        # plot interval setting
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.fluorescence_measurment)
        #self.timer.start()

        # stimulation interval setting
        self.timer_stim = QtCore.QTimer()
        self.timer_stim.timeout.connect(self.stimulate)
        self.amplitude = 0
        self.stim_for_csv = 0
        self.counter = 0
        self.tmr_stim_counter = 0

        # keyboard input setting
        # When the shortcut key set here is input, the plot is made with pixels on cursor coordinates when input.
        kb.add_hotkey('shift+F', lambda: self.plot_position())

        # FG initialization
        self.FG_init_state = 0
        self.timer_FG_init = QtCore.QTimer()
        self.timer_FG_init.timeout.connect(self.FG_initialization)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


    def FG_initialization(self):
        if self.FG_init_state == 0:
            self.send_command("WMA0" + "\n")# 0 V
            self.FG_init_state += 1
        elif self.FG_init_state == 1:
            self.send_command("WMF200000" + "\n")# 200mhz
            self.FG_init_state += 1
        elif self.FG_init_state == 2:
            self.send_command("WMW34" + "\n")# arbitary wave
            self.FG_init_state += 1
        elif self.FG_init_state == 3:
            self.send_command("WMO0" + "\n")# offset 0 V
            self.FG_init_state += 1
        elif self.FG_init_state == 4:
            self.send_command("WMN1" + "\n")# output on
            self.FG_init_state += 1
        elif self.FG_init_state == 5:
            self.timer_FG_init.stop()


    def on_click_stimlate(self):
        if self.click_flg == False:
            self.click_flg = True
            self.stim_flg = True
            self.stim_amp = int(self.stim_amp_line.text())
            self.stim_deltaV = int(self.stim_deltaV_line.text())
            self.stim_interval = int(self.stim_interval_line.text())
            self.stim_count = int(self.stim_count_line.text())

            print(self.stim_amp)
            self.timer_stim.start(5000)  # 5s
            self.stim_button.setStyleSheet("background-color: rgb(100,230,180)")
            self.stim_button.setText("Stimulating ...")
        else:
            self.reset_stim_setting()


    def on_click_com(self):
        if self.FG_connect_flg == False:
            self.FG_connect_flg = True
            # serial communication setting
            # 11520 kbps
            self.port_number = "COM9"
            self.ser = serial.Serial(self.port_number, 115200, timeout=1)
            print(str(self.port_number) + " Opened!!")
            self.tmp_counter = 0

            self.timer_FG_init.start(500)  # Need a delay for each command
            self.com_button.setStyleSheet("background-color: rgb(100,230,180)")
            self.com_button.setText("Connected")
        else:
            pass


    def on_click_start(self):
        # save
        self.date = datetime.datetime.today()
        self.save_path = self.save_path_line.text()
        if os.path.exists(self.save_path) != True:
            os.makedirs(self.save_path)
        self.start_button.setStyleSheet("background-color: rgb(100,230,180)")
        self.start_button.setText("Now measuring...")
        self.timer.start()


    def send_command(self, command):
        self.ser.write(command.encode())
        print(command)


    # multiple
    def stimulate(self):
        if self.tmr_stim_counter >= self.stim_count:
            self.reset_stim_setting()
        if self.amplitude == 0:
            self.amplitude = self.stim_amp
        else:
            self.amplitude+=self.stim_deltaV

        self.stim_for_csv = 255
        self.tmr_stim_counter += 1
        self.send_command("WMA" + str(self.amplitude) + "\n")
        # visualize
        self.vline = pg.InfiniteLine(angle=90, movable=False)
        self.p1.addItem(self.vline, ignoreBounds=True)
        self.vline.setPos(self.index[-1])
    """
    # single
    def stimulate(self):
        self.amplitude = 4
        self.stim_for_csv = 255
        self.send_command("WMA" + str(self.amplitude) + "\n")
        # visualize
        self.vline = pg.InfiniteLine(angle=90, movable=False)
        self.p1.addItem(self.vline, ignoreBounds=True)
        self.vline.setPos(self.index[-1])
        self.reset_stim_setting()
    """

    def reset_stim_setting(self):
        self.amplitude = 0
        self.tmr_stim_counter = 0
        self.click_flg = False
        self.stim_flg = False
        self.timer_stim.stop()
        self.send_command("WMA0" + "\n")
        self.stim_button.setStyleSheet("background-color: rgb(230, 230, 230)")
        self.stim_button.setText("Stimulate")


    def fluorescence_measurment(self):
        # 2x2 pixels, accurate
        """
        self.rgb1 = pag.pixel(self.x, self.y)
        self.rgb2 = pag.pixel(self.x+self.pixel_pitch, self.y)
        self.rgb3 = pag.pixel(self.x, self.y+self.pixel_pitch)
        self.rgb4 = pag.pixel(self.x+self.pixel_pitch, self.y+self.pixel_pitch)
        self.gray1 = (77*self.rgb1[0]+150*self.rgb1[1]+29*self.rgb1[2])/256
        self.gray2 = (77*self.rgb2[0]+150*self.rgb2[1]+29*self.rgb2[2])/256
        self.gray3 = (77*self.rgb3[0]+150*self.rgb3[1]+29*self.rgb3[2])/256
        self.gray4 = (77*self.rgb4[0]+150*self.rgb4[1]+29*self.rgb4[2])/256
        self.gray = (self.gray1+self.gray2+self.gray3+self.gray4)/4
        """
        # simple
        self.rgb1 = pag.pixel(self.x, self.y)
        self.gray = self.rgb1[0]
        self.index = np.delete(self.index, 0)
        self.index = np.append(self.index, self.index[-1]+1)
        self.flu_data = np.delete(self.flu_data, 0)
        self.flu_data = np.append(self.flu_data, [self.gray])

        # plot
        # pyqtgraph
        self.curve1.setData(self.index, self.flu_data)

        self.treeWidget.takeTopLevelItem(0)
        self.item_0 = QtWidgets.QTreeWidgetItem(self.treeWidget)
        self.treeWidget.topLevelItem(self.view_data_len-1).setText(0, str(self.rgb1[0]))
        self.treeWidget.topLevelItem(self.view_data_len-1).setText(1, str(self.rgb1[1]))
        self.treeWidget.topLevelItem(self.view_data_len-1).setText(2, str(self.rgb1[2]))

        # save
        self.date_tmp = datetime.datetime.today()
        self.tm = str(self.date_tmp.hour) + ', ' + str(self.date_tmp.minute) + ', ' + str(self.date_tmp.second) + ', ' + str(self.date_tmp.microsecond)
        self.filename = ("fluorescence" + str(self.date.year) + '_' + str(self.date.month) + '_' +
                    str(self.date.day)  + '_' + str(self.date.hour) + '_' + str(self.date.minute) + '_' + str(self.date.second) + '_.csv')
        #df = pd.DataFrame({"time": self.tm, "Intensity": self.gray})
        df = pd.DataFrame(columns=[self.tm, self.stim_for_csv, self.gray])
        #print(self.save_path + self.filename)

        df.to_csv(self.save_path + self.filename, mode="a")
        if self.stim_for_csv == 255:
            self.stim_for_csv = 0


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