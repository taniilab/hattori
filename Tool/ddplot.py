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
from PyQt5.QtGui import QIcon
import numpy as np
import os
from PyQt5.QtWidgets import *
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

class Ui_MainWindow(object):
    timer: QTimer
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(500, 500)
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralWidget.sizePolicy().hasHeightForWidth())
        self.centralWidget.setSizePolicy(sizePolicy)
        self.centralWidget.setObjectName("centralWidget")
        self.centralWidget.setStyleSheet("QLabel {font: 13pt Arial}" "QLineEdit {font: 13pt Arial}")

        self.layout = QtWidgets.QVBoxLayout()
        self.db = DropButton("CSVファイルをここにドラッグするぴっぴ")
        self.db.setStyleSheet("QPushButton{font-size: 30px;"
                              "font-family: MS Sans Serif;"
                              "color: rgb(255, 255, 255);"
                              "background-color: rgb(204, 102, 102);}")

        self.skiprows_label = QtWidgets.QLabel("Skip rows")
        self.def_skiprows = "0"
        self.skiprows_line = QtWidgets.QLineEdit(self.def_skiprows)
        self.layout_skiprows = QtWidgets.QHBoxLayout()
        self.layout_skiprows.addWidget(self.skiprows_label)
        self.layout_skiprows.addWidget(self.skiprows_line)
        self.skiprows_w = QtWidgets.QWidget()
        self.skiprows_w.setLayout(self.layout_skiprows)

        self.dpi_label = QtWidgets.QLabel("DPI")
        self.def_dpi = "600"
        self.dpi_line = QtWidgets.QLineEdit(self.def_dpi)
        self.layout_dpi = QtWidgets.QHBoxLayout()
        self.layout_dpi.addWidget(self.dpi_label)
        self.layout_dpi.addWidget(self.dpi_line)
        self.dpi_w = QtWidgets.QWidget()
        self.dpi_w.setLayout(self.layout_dpi)

        self.width_label = QtWidgets.QLabel("Width [inch]")
        self.def_width = "3"
        self.width_line = QtWidgets.QLineEdit(self.def_width)
        self.layout_width = QtWidgets.QHBoxLayout()
        self.layout_width.addWidget(self.width_label)
        self.layout_width.addWidget(self.width_line)
        self.width_w = QtWidgets.QWidget()
        self.width_w.setLayout(self.layout_width)

        self.height_label = QtWidgets.QLabel("Height [inch]")
        self.def_height = "2"
        self.height_line = QtWidgets.QLineEdit(self.def_height)
        self.layout_height = QtWidgets.QHBoxLayout()
        self.layout_height.addWidget(self.height_label)
        self.layout_height.addWidget(self.height_line)
        self.height_w = QtWidgets.QWidget()
        self.height_w.setLayout(self.layout_height)



        self.layout.addWidget(self.db)
        self.layout.addWidget(self.skiprows_w)
        self.layout.addWidget(self.dpi_w)
        self.layout.addWidget(self.width_w)
        self.layout.addWidget(self.height_w)
        self.centralWidget.setLayout(self.layout)
        MainWindow.setCentralWidget(self.centralWidget)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.readcsv)
        self.timer.start(100)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "CAWAII Plotter"))
        MainWindow.setWindowIcon(QIcon('maria.png'))

    def readcsv(self):
        if self.db.drop_flg == True:
            self.db.drop_flg = False
            print(self.db.mineData)

            df = pd.read_csv(self.db.mineData, skiprows=int(self.skiprows_line.text()))
            #print(df)

            fig = plt.figure(figsize=(int(self.width_line.text()),
                                      int(self.height_line.text())),
                             dpi=int(self.dpi_line.text()))
            self.ax = fig.add_subplot(1, 1, 1)
            print(df)
            print(df.columns[0])
            self.ax.plot(df[str(df.columns[0])], df[str(df.columns[3])])
            plt.show()


class DropButton(QPushButton):

    def __init__(self, title, parent=None):
        super().__init__(title, parent)
        self.mineData = "Null"
        self.drop_flg = False
        # ボタンに対してドロップ操作を可能にする
        self.setAcceptDrops(True)


    def dragEnterEvent(self, e):
        # ドラッグ可能なデータ形式を設定
        self.mineData = e.mimeData().text()
        e.acceptProposedAction()

    def dropEvent(self, e):
        self.drop_flg = True
        # ドロップしたときにボタンラベルを入れ替える
        #self.setText(e.mimeData().text())
