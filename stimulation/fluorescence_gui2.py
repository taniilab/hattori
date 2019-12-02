# -*- coding: utf-8 -*-
# Fluorescence Plotter for HCImage Live
# programmed by Kouhei Hattori, Hekiru Kurakake of Waseda University

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
import time
import csv
import shutil

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

        self.stim_com_label = QtWidgets.QLabel("COM")
        self.def_stim_com = "9"
        self.stim_com_line = QtWidgets.QLineEdit(self.def_stim_com)
        self.layout_stim_com = QtWidgets.QHBoxLayout()
        self.layout_stim_com.addWidget(self.stim_com_label)
        self.layout_stim_com.addWidget(self.stim_com_line)
        self.stim_com_w = QtWidgets.QWidget()
        self.stim_com_w.setLayout(self.layout_stim_com)

        self.stim_waveform_label = QtWidgets.QLabel("Waveform number(34~: arbitary)")
        self.def_stim_waveform = "34"
        self.stim_waveform_line = QtWidgets.QLineEdit(self.def_stim_waveform)
        self.layout_stim_waveform = QtWidgets.QHBoxLayout()
        self.layout_stim_waveform.addWidget(self.stim_waveform_label)
        self.layout_stim_waveform.addWidget(self.stim_waveform_line)
        self.stim_waveform_w = QtWidgets.QWidget()
        self.stim_waveform_w.setLayout(self.layout_stim_waveform)

        self.stim_amp_label = QtWidgets.QLabel("Initial amplitude (peak to peak) [V]")
        self.def_stim_amp = "1"
        self.stim_amp_line = QtWidgets.QLineEdit(self.def_stim_amp)
        self.layout_stim_amp = QtWidgets.QHBoxLayout()
        self.layout_stim_amp.addWidget(self.stim_amp_label)
        self.layout_stim_amp.addWidget(self.stim_amp_line)
        self.stim_amp_w = QtWidgets.QWidget()
        self.stim_amp_w.setLayout(self.layout_stim_amp)

        self.stim_count_label = QtWidgets.QLabel("Number of stimuli")
        self.def_stim_count = "5"
        self.stim_count_line = QtWidgets.QLineEdit(self.def_stim_count)
        self.layout_stim_count = QtWidgets.QHBoxLayout()
        self.layout_stim_count.addWidget(self.stim_count_label)
        self.layout_stim_count.addWidget(self.stim_count_line)
        self.stim_count_w = QtWidgets.QWidget()
        self.stim_count_w.setLayout(self.layout_stim_count)

        self.stim_deltaV_label = QtWidgets.QLabel("Delta amplitude [V]")
        self.def_stim_deltaV = "1"
        self.stim_deltaV_line = QtWidgets.QLineEdit(self.def_stim_deltaV)
        self.layout_stim_deltaV = QtWidgets.QHBoxLayout()
        self.layout_stim_deltaV.addWidget(self.stim_deltaV_label)
        self.layout_stim_deltaV.addWidget(self.stim_deltaV_line)
        self.stim_deltaV_w = QtWidgets.QWidget()
        self.stim_deltaV_w.setLayout(self.layout_stim_deltaV)

        self.stim_interval_label = QtWidgets.QLabel("Interval of stimuli")
        self.def_stim_interval = "1"
        self.stim_interval_line = QtWidgets.QLineEdit(self.def_stim_interval)
        self.stim_interval_line.setValidator(QtGui.QIntValidator())
        self.layout_stim_interval = QtWidgets.QHBoxLayout()
        self.layout_stim_interval.addWidget(self.stim_interval_label)
        self.layout_stim_interval.addWidget(self.stim_interval_line)
        self.stim_interval_w = QtWidgets.QWidget()
        self.stim_interval_w.setLayout(self.layout_stim_interval)

        self.stim_firststimulation_label = QtWidgets.QLabel("First Stimulation(-1=disabled)")
        self.def_stim_firststimulation = "60"
        self.stim_firststimulation_line = QtWidgets.QLineEdit(self.def_stim_firststimulation)
        self.layout_stim_firststimulation = QtWidgets.QHBoxLayout()
        self.layout_stim_firststimulation.addWidget(self.stim_firststimulation_label)
        self.layout_stim_firststimulation.addWidget(self.stim_firststimulation_line)
        self.stim_firststimulation_w = QtWidgets.QWidget()
        self.stim_firststimulation_w.setLayout(self.layout_stim_firststimulation)
        self.first_stim_flag = False

        self.stim_secondstimulation_label = QtWidgets.QLabel("Second Stimulation(-1=disabled)")
        self.def_stim_secondstimulation = "180"
        self.stim_secondstimulation_line = QtWidgets.QLineEdit(self.def_stim_secondstimulation)
        self.layout_stim_secondstimulation = QtWidgets.QHBoxLayout()
        self.layout_stim_secondstimulation.addWidget(self.stim_secondstimulation_label)
        self.layout_stim_secondstimulation.addWidget(self.stim_secondstimulation_line)
        self.stim_secondstimulation_w = QtWidgets.QWidget()
        self.stim_secondstimulation_w.setLayout(self.layout_stim_secondstimulation)
        self.second_stim_flag = False

        self.save_path_label = QtWidgets.QLabel("Save path")
        self.def_path = "G:/Stim_G/csvdata/"
        self.save_path_line = QtWidgets.QLineEdit(self.def_path)
        self.layout_save_path = QtWidgets.QHBoxLayout()
        self.layout_save_path.addWidget(self.save_path_label)
        self.layout_save_path.addWidget(self.save_path_line)
        self.save_path_w = QtWidgets.QWidget()
        self.save_path_w.setLayout(self.layout_save_path)

        self.save1_button = QtWidgets.QPushButton('Save1')
        self.bfont = self.save1_button.font()
        self.bfont.setPointSizeF(20)
        self.save1_button.setFont(self.bfont)
        self.save1_button.setStyleSheet("background-color: rgb(230,230,230)")
        self.save1_button.clicked.connect(self.save1)
        self.save2_button = QtWidgets.QPushButton('Save2')
        self.bfont = self.save2_button.font()
        self.bfont.setPointSizeF(20)
        self.save2_button.setFont(self.bfont)
        self.save2_button.setStyleSheet("background-color: rgb(230,230,230)")
        #self.save2_button.clicked.connect(self.save2)
        self.save3_button = QtWidgets.QPushButton('Save3')
        self.bfont = self.save3_button.font()
        self.bfont.setPointSizeF(20)
        self.save3_button.setFont(self.bfont)
        self.save3_button.setStyleSheet("background-color: rgb(230,230,230)")
        #self.save3_button.clicked.connect(self.save3)
        self.layout_save = QtWidgets.QHBoxLayout()
        self.layout_save.addWidget(self.save1_button)
        self.layout_save.addWidget(self.save2_button)
        self.layout_save.addWidget(self.save3_button)
        self.save_button = QtWidgets.QWidget()
        self.save_button.setLayout(self.layout_save)

        self.load1_button = QtWidgets.QPushButton('Load1')
        self.bfont = self.load1_button.font()
        self.bfont.setPointSizeF(20)
        self.load1_button.setFont(self.bfont)
        self.load1_button.setStyleSheet("background-color: rgb(230,230,230)")
        self.load1_button.clicked.connect(self.load1)
        self.load2_button = QtWidgets.QPushButton('Load2')
        self.bfont = self.load2_button.font()
        self.bfont.setPointSizeF(20)
        self.load2_button.setFont(self.bfont)
        self.load2_button.setStyleSheet("background-color: rgb(230,230,230)")
        #self.load2_button.clicked.connect(self.load2)
        self.load3_button = QtWidgets.QPushButton('Load3')
        self.bfont = self.load3_button.font()
        self.bfont.setPointSizeF(20)
        self.load3_button.setFont(self.bfont)
        self.load3_button.setStyleSheet("background-color: rgb(230,230,230)")
        #self.load3_button.clicked.connect(self.load3)
        self.layout_load = QtWidgets.QHBoxLayout()
        self.layout_load.addWidget(self.load1_button)
        self.layout_load.addWidget(self.load2_button)
        self.layout_load.addWidget(self.load3_button)
        self.load_button = QtWidgets.QWidget()
        self.load_button.setLayout(self.layout_load)

        self.stim_button = QtWidgets.QPushButton('Manual Stimulate(機能停止中)')
        self.bfont = self.stim_button.font()
        self.bfont.setPointSizeF(20)
        self.stim_button.setFont(self.bfont)
        self.stim_button.setStyleSheet("background-color: rgb(230,230,230)")
        self.stim_button.clicked.connect(self.on_click_stimlate)

        self.com_button = QtWidgets.QPushButton('Disconnected')
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

        self.splitter_left.addWidget(self.stim_waveform_w)
        self.splitter_left.addWidget(self.stim_com_w)
        self.splitter_left.addWidget(self.stim_amp_w)
        self.splitter_left.addWidget(self.stim_count_w)
        self.splitter_left.addWidget(self.stim_deltaV_w)
        self.splitter_left.addWidget(self.stim_interval_w)
        self.splitter_left.addWidget(self.stim_firststimulation_w)
        self.splitter_left.addWidget(self.stim_secondstimulation_w)
        self.splitter_left.addWidget(self.save_path_w)
        self.splitter_left.addWidget(self.save_button)
        self.splitter_left.addWidget(self.load_button)
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
        self.index = np.arange(0, 3000)
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
        self.ms_start_flag = False
        self.rec_time = 0
        self.start_time = time.time()
        self.dummy_for_start_flag = np.ones(100)
        #self.timer.start()

        # stimulation interval setting
        self.timer_stim = QtCore.QTimer()
        self.timer_stim.timeout.connect(self.stimulate)
        self.amplitude = 0
        self.stim_for_csv = 0
        self.counter = 0

        # stimulation interval fix
        # 5秒以上の間隔を開けての刺激に対応する。
        self.timer_stim_reset = QtCore.QTimer()
        self.timer_stim_reset.timeout.connect(self.stimulate_interval_fix)


        # keyboard input setting
        # When the shortcut key set here is input, the plot is made with pixels on cursor coordinates when input.
        kb.add_hotkey('shift+F', lambda: self.plot_position())

        # Auto Stimulation System
        self.stim_interval = 5
        self.stim_firststimulation = 60
        self.stim_secondstimulation = -1

        # FG initialization
        self.FG_init_state = 0
        self.timer_FG_init = QtCore.QTimer()
        self.timer_FG_init.timeout.connect(self.FG_initialization)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(1)
        self.load_previous()
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


    def FG_initialization(self):
        if self.FG_init_state == 0:
            self.send_command("WMA0" + "\n")# 0 V
            self.FG_init_state += 1
        elif self.FG_init_state == 1:
            self.send_command("WMF200000" + "\n")# 200mhz
            self.FG_init_state += 1
        elif self.FG_init_state == 2:
            self.send_command("WMW" + self.stim_waveform_line.text() + "\n")# WMW34 -> arbitary wave 1
            self.FG_init_state += 1
        elif self.FG_init_state == 3:
            self.send_command("WMO0" + "\n")# offset 0 V
            self.FG_init_state += 1
        elif self.FG_init_state == 4:
            self.send_command("WMN1" + "\n")# output on
            self.FG_init_state += 1
        elif self.FG_init_state == 5:
            self.timer_FG_init.stop()
            self.FG_init_state = 0

    def on_click_stimlate(self):
        if self.click_flg == False:
            self.click_flg = True
            self.stim_flg = True
            self.stim_amp = self.stim_amp_line.text()
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
            self.port_number = "COM" + self.stim_com_line.text()
            self.ser = serial.Serial(self.port_number, 115200, timeout=1)
            print(str(self.port_number) + " Opened!!")

            self.timer_FG_init.start(500)  # Need a delay for each command
            self.com_button.setStyleSheet("background-color: rgb(100,230,180)")
            self.com_button.setText("Connected")
            self.stim_counter = 0
        else:
            self.FG_connect_flg = False
            self.ser.close()
            print(str(self.port_number) + " Closed!!")
            self.com_button.setStyleSheet("background-color: rgb(230,230,230)")
            self.com_button.setText("Disconnected")
            self.stim_counter = 0

    def on_click_start(self):
        if self.ms_start_flag == False:
            self.save_previous()
            print("Starting...")
            self.date = datetime.datetime.today()
            self.save_path = self.save_path_line.text()
            if os.path.exists(self.save_path) != True:
                os.makedirs(self.save_path)
            self.start_time = time.time()
            self.amplitude = float(self.stim_amp_line.text())

            self.timer.start()

            try:
                self.stim_interval =int(self.stim_interval_line.text())
                self.stim_firststimulation = int(self.stim_firststimulation_line.text())
                self.stim_secondstimulation = int(self.stim_secondstimulation_line.text())
            except:
                print("数値への変換に失敗??")
                self.stim_interval = 1
                self.stim_firststimulation = 60
                self.stim_secondstimulation = -1

            if self.stim_firststimulation == -1:
                self.first_stim_flag = True
                print("first stimulation disabled")
            if self.stim_secondstimulation == -1:
                self.second_stim_flag = True
                print("second stimulation disabled")

            self.stim_firststimulation = self.stim_firststimulation - self.stim_interval
            self.stim_secondstimulation = self.stim_secondstimulation - self.stim_interval

        elif self.ms_start_flag == True:
            self.timer.stop()
            self.ms_start_flag = False
            self.first_stim_flag = False
            self.second_stim_flag = False
            self.start_button.setStyleSheet("background-color: rgb(230,230,230)")
            self.start_button.setText("Measure")
            print("End")

    def save_previous(self):
        cfg_path = os.getcwd()+'/previous.cfg'
        data = np.zeros(8,dtype=int)
        data[0] = self.stim_waveform_line.text()
        data[1] = self.stim_com_line.text()
        data[2] = self.stim_amp_line.text()
        data[3] = self.stim_count_line.text()
        data[4] = self.stim_deltaV_line.text()
        data[5] = self.stim_interval_line.text()
        data[6] = self.stim_firststimulation_line.text()
        data[7] = self.stim_secondstimulation_line.text()
        data2 = []
        data2.append(self.save_path_line.text())
        with open(cfg_path, 'w' , newline = "") as f:
            writer = csv.writer(f)
            writer.writerow(data)
            writer.writerow(data2)
        return

    def load_previous(self):
        cfg_path = os.getcwd()+'/previous.cfg'
        if not os.path.isfile(cfg_path):
            return
        data = np.zeros(8,dtype=int)
        data[0] = self.stim_waveform_line.text()
        data[1] = self.stim_com_line.text()
        data[2] = self.stim_amp_line.text()
        data[3] = self.stim_count_line.text()
        data[4] = self.stim_deltaV_line.text()
        data[5] = self.stim_interval_line.text()
        data[6] = self.stim_firststimulation_line.text()
        data[7] = self.stim_secondstimulation_line.text()
        data2 = []
        data2.append(self.save_path_line.text())
        with open(cfg_path, 'w' , newline = "") as f:
            writer = csv.writer(f)
            writer.writerow(data)
            writer.writerow(data2)
        return

    def save1(self):
        cfg_path = os.getcwd()+'/save1.cfg'
        data = np.zeros(8,dtype=int)
        data[0] = self.stim_waveform_line.text()
        data[1] = self.stim_com_line.text()
        data[2] = self.stim_amp_line.text()
        data[3] = self.stim_count_line.text()
        data[4] = self.stim_deltaV_line.text()
        data[5] = self.stim_interval_line.text()
        data[6] = self.stim_firststimulation_line.text()
        data[7] = self.stim_secondstimulation_line.text()
        data2 = []
        data2.append(self.save_path_line.text())
        with open(cfg_path, 'w' , newline = "") as f:
            writer = csv.writer(f)
            writer.writerow(data)
            writer.writerow(data2)
        return


    def load1(self):
        cfg_path = os.getcwd()+'/save1.cfg'
        with open(cfg_path,'r') as f:
            reader = csv.reader(f)
            l = [row for row in reader]
            self.stim_waveform_line.setText(l[0][0])
            self.stim_com_line.setText(l[0][1])
            self.stim_amp_line.setText(l[0][2])
            self.stim_count_line.setText(l[0][3])
            self.stim_deltaV_line.setText(l[0][4])
            self.stim_interval_line.setText(l[0][5])
            self.stim_firststimulation_line.setText(l[0][6])
            self.stim_secondstimulation_line.setText(l[0][7])
            self.save_path_line.setText(l[1][0])

        return

    def save2(self):
        cfg_path = os.getcwd()+'/save2.cfg'
        data = np.zeros(8,dtype=int)
        data[0] = self.stim_waveform_line.text()
        data[1] = self.stim_com_line.text()
        data[2] = self.stim_amp_line.text()
        data[3] = self.stim_count_line.text()
        data[4] = self.stim_deltaV_line.text()
        data[5] = self.stim_interval_line.text()
        data[6] = self.stim_firststimulation_line.text()
        data[7] = self.stim_secondstimulation_line.text()
        data2 = []
        data2.append(self.save_path_line.text())
        with open(cfg_path, 'w' , newline = "") as f:
            writer = csv.writer(f)
            writer.writerow(data)
            writer.writerow(data2)
        return

    def load2(self):
        cfg_path = os.getcwd()+'/save2.cfg'
        with open(cfg_path,'r') as f:
            reader = csv.reader(f)
            l = [row for row in reader]
            self.stim_waveform_line.setText(l[0][0])
            self.stim_com_line.setText(l[0][1])
            self.stim_amp_line.setText(l[0][2])
            self.stim_count_line.setText(l[0][3])
            self.stim_deltaV_line.setText(l[0][4])
            self.stim_interval_line.setText(l[0][5])
            self.stim_firststimulation_line.setText(l[0][6])
            self.stim_secondstimulation_line.setText(l[0][7])
            self.save_path_line.setText(l[1][0])

        return

    def save3(self):
        cfg_path = os.getcwd()+'/save3.cfg'
        data = np.zeros(8,dtype=int)
        data[0] = self.stim_waveform_line.text()
        data[1] = self.stim_com_line.text()
        data[2] = self.stim_amp_line.text()
        data[3] = self.stim_count_line.text()
        data[4] = self.stim_deltaV_line.text()
        data[5] = self.stim_interval_line.text()
        data[6] = self.stim_firststimulation_line.text()
        data[7] = self.stim_secondstimulation_line.text()
        data2 = []
        data2.append(self.save_path_line.text())
        with open(cfg_path, 'w' , newline = "") as f:
            writer = csv.writer(f)
            writer.writerow(data)
            writer.writerow(data2)
        return

    def load3(self):
        cfg_path = os.getcwd()+'/save3.cfg'
        with open(cfg_path,'r') as f:
            reader = csv.reader(f)
            l = [row for row in reader]
            self.stim_waveform_line.setText(l[0][0])
            self.stim_com_line.setText(l[0][1])
            self.stim_amp_line.setText(l[0][2])
            self.stim_count_line.setText(l[0][3])
            self.stim_deltaV_line.setText(l[0][4])
            self.stim_interval_line.setText(l[0][5])
            self.stim_firststimulation_line.setText(l[0][6])
            self.stim_secondstimulation_line.setText(l[0][7])
            self.save_path_line.setText(l[1][0])

        return

    def send_command(self, command):
        self.ser.write(command.encode())
        print(command)

    # multiple
    def stimulate(self):
        if self.stim_counter >= int(self.stim_count_line.text()):
            self.reset_stim_setting()
        else:
            if self.stim_counter != 0:
                self.amplitude += round(float(self.stim_deltaV_line.text()), 1)
            self.stim_counter += 1
            self.stim_for_csv = 255
            self.send_command("WMA" + str(self.amplitude) + "\n")
            # visualize
            self.vline = pg.InfiniteLine(angle=90, movable=False)
            self.p1.addItem(self.vline, ignoreBounds=True)
            self.vline.setPos(self.index[-1])
            self.timer_stim_reset.start(200)

    def stimulate_interval_fix(self):
        # 5秒以上の刺激に対応する.
        # 刺激導入後200秒後に呼び出され、amplitudeをリセットする
        self.send_command("WMA0\n")
        self.timer_stim_reset.stop()

    def reset_stim_setting(self):
        self.amplitude = 0
        self.stim_counter = 0
        self.click_flg = False
        self.stim_flg = False
        self.timer_stim.stop()
        self.send_command("WMA0" + "\n")
        self.stim_button.setStyleSheet("background-color: rgb(230, 230, 230)")
        self.stim_button.setText("Manual Stimulate")

    def fluorescence_measurment(self):
        if self.ms_start_flag == False:
            self.start_button.setStyleSheet("background-color: rgb(255,234,13)")
            self.start_button.setText("Start measuring on HCimage!!!")
        elif self.ms_start_flag == True:
            self.start_button.setStyleSheet("background-color: rgb(100,230,180)")
            self.start_button.setText("Now measuring...")

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
        self.curve1.setData(self.index, self.flu_data)

        # value
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
        df = pd.DataFrame(columns=[self.tm, self.stim_for_csv, self.gray])

        # Auto Stimulation System
        if (self.first_stim_flag == False and self.ms_start_flag == True and time.time() - self.start_time > self.stim_firststimulation):
            print("Auto Stimulation System starts...[FIRST STIMULATION]")
            print("Interval of Stimulation:" + str(self.stim_interval) + "seconds")
            print("Start time of Stimulation:" + str(self.stim_firststimulation) + "seconds")
            self.first_stim_flag = True
            self.stim_amp = self.stim_amp_line.text()
            self.timer_stim.start(self.stim_interval*1000)  # 5s
            self.stim_button.setStyleSheet("background-color: rgb(100,230,180)")
            self.stim_button.setText("Stimulating ...")

        if (self.second_stim_flag == False and self.ms_start_flag == True and time.time() - self.start_time > self.stim_secondstimulation):
            print("Auto Stimulation System starts...[SECOND STIMULATION]")
            print("Interval of Stimulation:" + str(self.stim_interval) + "seconds")
            print("Start time of Stimulation:" + str(self.stim_secondstimulation) + "seconds")
            self.second_stim_flag = True
            self.stim_amp = self.stim_amp_line.text()
            self.timer_stim.start(self.stim_interval*1000)  # 5s
            self.stim_button.setStyleSheet("background-color: rgb(100,230,180)")
            self.stim_button.setText("Stimulating ...")

        # FIFO
        self.dummy_for_start_flag = np.roll(self.dummy_for_start_flag, -1)
        self.dummy_for_start_flag[-1] = self.gray

        # メモ　データ冒頭部の邪魔な部分を無視するために追記
        # 過去１００プロット分の蛍光データを保存しておくリストを用意
        # リストの末項以外の値が全て同じ値の時に計測開始フラグ（self.ms_start_flag）が立つようにしてある
        # self.dummy_for_start_flag[self.dummy_for_start_flag != self.dummy_for_start_flag[0]]　→　リスト先頭の値と異なる要素を全て抽出
        # 誤検出回避のため、self.dummy_for_start_flagの値は計測場所の輝度値で全て初期化してある（def plot_position(self)）
        if (self.ms_start_flag == False):
            self.dummy2 = self.dummy_for_start_flag[self.dummy_for_start_flag != self.dummy_for_start_flag[0]]
            if self.dummy2 == self.dummy_for_start_flag[-1]:
                #print(self.dummy_for_start_flag)
                self.ms_start_flag = True
                self.start_time = time.time()
        else:
            df.to_csv(self.save_path + self.filename, mode="a")

        if self.stim_for_csv == 255:
            self.stim_for_csv = 0


    def plot_position(self):
        self.x, self.y = pag.position()
        self.kanon = pag.pixel(self.x, self.y)
        self.dummy_for_start_flag = self.kanon[0] * np.ones(len(self.dummy_for_start_flag))
        print(self.dummy_for_start_flag[0])


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
