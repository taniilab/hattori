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
        self.db = DropButton("CSVファイルをここにかわいくドラッグして下さいね")
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

        self.plot_line_w_label = QtWidgets.QLabel("Plot linewidth")
        self.def_plot_line_w = "1"
        self.plot_line_w_line = QtWidgets.QLineEdit(self.def_plot_line_w)
        self.layout_plot_line_w = QtWidgets.QHBoxLayout()
        self.layout_plot_line_w.addWidget(self.plot_line_w_label)
        self.layout_plot_line_w.addWidget(self.plot_line_w_line)
        self.plot_line_w_w = QtWidgets.QWidget()
        self.plot_line_w_w.setLayout(self.layout_plot_line_w)

        self.ax_line_w_label = QtWidgets.QLabel("Axis linewidth")
        self.def_ax_line_w = "2"
        self.ax_line_w_line = QtWidgets.QLineEdit(self.def_ax_line_w)
        self.layout_ax_line_w = QtWidgets.QHBoxLayout()
        self.layout_ax_line_w.addWidget(self.ax_line_w_label)
        self.layout_ax_line_w.addWidget(self.ax_line_w_line)
        self.ax_line_w_w = QtWidgets.QWidget()
        self.ax_line_w_w.setLayout(self.layout_ax_line_w)

        self.x_label_label = QtWidgets.QLabel("X label")
        self.def_x_label = "Time [s]"
        self.x_label_line = QtWidgets.QLineEdit(self.def_x_label)
        self.layout_x_label = QtWidgets.QHBoxLayout()
        self.layout_x_label.addWidget(self.x_label_label)
        self.layout_x_label.addWidget(self.x_label_line)
        self.x_label_w = QtWidgets.QWidget()
        self.x_label_w.setLayout(self.layout_x_label)

        self.y_label_label = QtWidgets.QLabel("Y label")
        self.def_y_label = "Fruorescence intensity [a.u.]"
        self.y_label_line = QtWidgets.QLineEdit(self.def_y_label)
        self.layout_y_label = QtWidgets.QHBoxLayout()
        self.layout_y_label.addWidget(self.y_label_label)
        self.layout_y_label.addWidget(self.y_label_line)
        self.y_label_w = QtWidgets.QWidget()
        self.y_label_w.setLayout(self.layout_y_label)

        self.xy_label_fsize_label = QtWidgets.QLabel("XY label font-size")
        self.def_xy_label_fsize = "5"
        self.xy_label_fsize_line = QtWidgets.QLineEdit(self.def_xy_label_fsize)
        self.layout_xy_label_fsize = QtWidgets.QHBoxLayout()
        self.layout_xy_label_fsize.addWidget(self.xy_label_fsize_label)
        self.layout_xy_label_fsize.addWidget(self.xy_label_fsize_line)
        self.xy_label_fsize_w = QtWidgets.QWidget()
        self.xy_label_fsize_w.setLayout(self.layout_xy_label_fsize)

        self.ax_tick_fsize_label = QtWidgets.QLabel("Axis tick font-size")
        self.def_ax_tick_fsize = "5"
        self.ax_tick_fsize_line = QtWidgets.QLineEdit(self.def_ax_tick_fsize)
        self.layout_ax_tick_fsize = QtWidgets.QHBoxLayout()
        self.layout_ax_tick_fsize.addWidget(self.ax_tick_fsize_label)
        self.layout_ax_tick_fsize.addWidget(self.ax_tick_fsize_line)
        self.ax_tick_fsize_w = QtWidgets.QWidget()
        self.ax_tick_fsize_w.setLayout(self.layout_ax_tick_fsize)


        self.save_path_label = QtWidgets.QLabel("Save path")
        self.def_save_path = "C:/"
        self.save_path_line = QtWidgets.QLineEdit(self.def_save_path)
        self.layout_save_path = QtWidgets.QHBoxLayout()
        self.layout_save_path.addWidget(self.save_path_label)
        self.layout_save_path.addWidget(self.save_path_line)
        self.save_path_w = QtWidgets.QWidget()
        self.save_path_w.setLayout(self.layout_save_path)

        self.save_button = QtWidgets.QPushButton('Save figure')
        self.save_button.setStyleSheet("QPushButton{font-size: 30px;"
                                       "font-family: MS Sans Serif;"
                                       "color: rgb(255, 255, 255);"
                                       "background-color: rgb(65, 65, 65);}")
        self.save_button.clicked.connect(self.on_click_savefig)

        self.layout.addWidget(self.db)
        self.layout.addWidget(self.skiprows_w)
        self.layout.addWidget(self.dpi_w)
        self.layout.addWidget(self.width_w)
        self.layout.addWidget(self.height_w)
        self.layout.addWidget(self.plot_line_w_w)
        self.layout.addWidget(self.ax_line_w_w)
        self.layout.addWidget(self.x_label_w)
        self.layout.addWidget(self.y_label_w)
        self.layout.addWidget(self.xy_label_fsize_w)
        self.layout.addWidget(self.ax_tick_fsize_w)
        self.layout.addWidget(self.save_path_w)
        self.layout.addWidget(self.save_button)
        """
        self.layout.addWidget()
        self.layout.addWidget()
        self.layout.addWidget()
        """
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
            fig = plt.figure(figsize=(float(self.width_line.text()),
                                      float(self.height_line.text())),
                             dpi=float(self.dpi_line.text()))
            self.ax = fig.add_subplot(1, 1, 1)
            print(df)
            print(df.columns[0])
            self.ax.plot(df[str(df.columns[0])], df[str(df.columns[3])],
                         color="black",
                         linewidth=float(self.plot_line_w_line.text()))
            self.ax.spines["right"].set_linewidth(0)
            self.ax.spines["top"].set_linewidth(0)
            self.ax.spines["left"].set_linewidth(self.ax_line_w_line.text())
            self.ax.spines["bottom"].set_linewidth(self.ax_line_w_line.text())
            self.ax.set_xlabel(str(self.x_label_line.text()),
                               fontsize=float(self.xy_label_fsize_line.text()),
                               color="black")
            self.ax.set_ylabel(str(self.y_label_line.text()),
                               fontsize=float(self.xy_label_fsize_line.text()),
                               color="black")
            self.ax.tick_params(labelsize=str(self.ax_tick_fsize_line.text()), colors="black")
            fig.tight_layout()
            plt.show()

    def on_click_savefig(self):
        print("Saved!!!")

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
