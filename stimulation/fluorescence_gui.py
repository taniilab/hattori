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


class Ui_MainWindow(object):
    timer: QTimer

    def setupUi(self, MainWindow):
        # number of plot
        self.nump = 5

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

        self.button = QtWidgets.QPushButton('Stimulate')
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
        self.x_init = 1000
        self.y_init = 1000
        self.index = np.arange(0, 1000)
        self.stim_data = np.zeros(len(self.index))
        self.flu_data = np.zeros(len(self.index))
        self.rgb_init = []
        self.gray_init = 0
        self.x = [self.x_init]
        self.y = [self.y_init]
        self.rgb = [self.rgb_init]
        self.gray = [self.gray_init]
        self.glaph_tab = pg.GraphicsWindow(title="fluorescence")
        self.p = []
        self.curve = []
        for i in range(self.nump):
            self.glaph_tab.nextRow()
            self.p.append(self.glaph_tab.addPlot())
            self.curve.append(self.p[i].plot(self.index, self.flu_data))
            self.x.append(self.x_init)
            self.y.append(self.y_init)
            self.rgb.append(self.rgb_init)
            self.gray.append(self.gray_init)
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
        self.timer.start()

        # stimulation interval setting
        self.timer_stim = QtCore.QTimer()
        self.timer_stim.timeout.connect(self.stimulate)
        self.amplitude = 0
        self.stim_for_csv = 0
        self.counter = 0

        # keyboard input setting
        # When the shortcut key set here is input, the plot is made with pixels on cursor coordinates when input.
        kb.add_hotkey("shift+0", lambda: self.plot_position0())# No.0 neuron
        kb.add_hotkey("shift+1", lambda: self.plot_position1())  # No.0 neuron
        kb.add_hotkey("shift+2", lambda: self.plot_position2())  # No.0 neuron
        kb.add_hotkey("shift+3", lambda: self.plot_position3())  # No.0 neuron
        kb.add_hotkey("shift+4", lambda: self.plot_position4())  # No.0 neuron


        # serial communication setting
        # 11520 kbps
        self.port_number = "COM11"
        self.ser = serial.Serial(self.port_number, 115200, timeout=1)
        print(str(self.port_number) + " Opened!!")
        self.tmp_counter = 0

        # FG initialization
        self.FG_init_state = 0
        self.timer_FG_init = QtCore.QTimer()
        self.timer_FG_init.timeout.connect(self.FG_initialization)
        self.timer_FG_init.start(500)# Need a delay for each command

        # save
        self.date = datetime.datetime.today()
        self.save_path = "C:/Users/Tanii_Lab/Desktop/test/"

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


    def on_click(self):
        if self.click_flg == False:
            self.click_flg = True
            self.stim_flg = True
            self.timer_stim.start(5000)  # 5s
            self.button.setStyleSheet("background-color: rgb(100,230,180)")
            self.button.setText("Stimulating ...")

        else:
            self.reset_stim_setting()

    def plot_position0(self):
        self.x[0], self.y[0] = pag.position()
        print("0")
    def plot_position1(self):
        self.x[1], self.y[1] = pag.position()
        print("1")
    def plot_position2(self):
        self.x[2], self.y[2] = pag.position()
    def plot_position3(self):
        self.x[3], self.y[3] = pag.position()
    def plot_position4(self):
        self.x[4], self.y[4] = pag.position()



    def send_command(self, command):
        self.ser.write(command.encode())
        print(command)


    def stimulate(self):
        if self.amplitude == 10:
            self.reset_stim_setting()
        else:
            self.amplitude+=1
            self.stim_for_csv = 255
            self.send_command("WMA" + str(self.amplitude) + "\n")
            # visualize
            self.vline = pg.InfiniteLine(angle=90, movable=False)
            self.p[0].addItem(self.vline, ignoreBounds=True)
            self.vline.setPos(self.index[-1])


    def reset_stim_setting(self):
        self.amplitude = 0
        self.click_flg = False
        self.stim_flg = False
        self.timer_stim.stop()
        self.send_command("WMA0" + "\n")
        self.button.setStyleSheet("background-color: rgb(230, 230, 230)")
        self.button.setText("Stimulate")


    def fluorescence_measurment(self):
        self.index = np.delete(self.index, 0)
        self.index = np.append(self.index, self.index[-1] + 1)
        self.flu_data = np.delete(self.flu_data, 0)
        self.flu_data = np.append(self.flu_data, [self.gray[0]])

        for i in range(len(self.p)):
            self.rgb[i] = pag.pixel(self.x[i], self.y[i])
            self.gray[i] = self.rgb[i][0]
            self.curve[i].setData(self.index, self.flu_data)

        self.treeWidget.takeTopLevelItem(0)
        self.item_0 = QtWidgets.QTreeWidgetItem(self.treeWidget)
        self.treeWidget.topLevelItem(self.view_data_len-1).setText(0, str(self.rgb[0][0]))
        self.treeWidget.topLevelItem(self.view_data_len-1).setText(1, str(self.rgb[0][1]))
        self.treeWidget.topLevelItem(self.view_data_len-1).setText(2, str(self.rgb[0][2]))

        # save
        df = pd.DataFrame(columns=[self.stim_for_csv, self.gray])
        self.filename = (str(self.date.year) + '_' + str(self.date.month) + '_' +
                    str(self.date.day) + '_' + str(self.date.hour) + '_' +
                    str(self.date.minute) + '_' + str(self.date.second) + '_.csv')
        df.to_csv(self.save_path + self.filename, mode="a")
        if self.stim_for_csv == 255:
            self.stim_for_csv = 0



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
