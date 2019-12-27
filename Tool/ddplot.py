# -*- coding: utf-8 -*-
# CAWAII Plotter
# programmed by Kouhei Hattoriof Waseda University

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QIcon
import numpy as np
import os
from PyQt5.QtWidgets import *
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime

class Ui_MainWindow(object):
    timer: QTimer
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(480, 720)
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralWidget.sizePolicy().hasHeightForWidth())
        self.centralWidget.setSizePolicy(sizePolicy)
        self.centralWidget.setObjectName("centralWidget")
        self.centralWidget.setStyleSheet("QLabel {font: 12pt Arial}"
                                         "QComboBox {font: 12pt Arial}"
                                         "QCheckBox {font: 12pt Arial}"
                                         "QLineEdit {font: 12pt Arial}")

        self.layout = QtWidgets.QVBoxLayout()
        self.db = DropButton("ここにCSVファイルをドラッグして下さい")
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

        self.plot_line_w_label = QtWidgets.QLabel("Plot line-width")
        self.def_plot_line_w = "1"
        self.plot_line_w_line = QtWidgets.QLineEdit(self.def_plot_line_w)
        self.layout_plot_line_w = QtWidgets.QHBoxLayout()
        self.layout_plot_line_w.addWidget(self.plot_line_w_label)
        self.layout_plot_line_w.addWidget(self.plot_line_w_line)
        self.plot_line_w_w = QtWidgets.QWidget()
        self.plot_line_w_w.setLayout(self.layout_plot_line_w)

        self.plot_color_label = QtWidgets.QLabel("Plot line-color")
        self.def_plot_color = "red"
        self.plot_color_cmbox = QtWidgets.QComboBox()
        self.items = {"black", "blue", "red", "green", "purple", "orange"}
        self.plot_color_cmbox.addItems(self.items)
        self.plot_color_cmbox.setCurrentText("black")
        self.layout_plot_color = QtWidgets.QHBoxLayout()
        self.layout_plot_color.addWidget(self.plot_color_label)
        self.layout_plot_color.addWidget(self.plot_color_cmbox)
        self.plot_color_w = QtWidgets.QWidget()
        self.plot_color_w.setLayout(self.layout_plot_color)

        self.ax_line_w_label = QtWidgets.QLabel("Axis line-width")
        self.def_ax_line_w = "2"
        self.ax_line_w_line = QtWidgets.QLineEdit(self.def_ax_line_w)
        self.layout_ax_line_w = QtWidgets.QHBoxLayout()
        self.layout_ax_line_w.addWidget(self.ax_line_w_label)
        self.layout_ax_line_w.addWidget(self.ax_line_w_line)
        self.ax_line_w_w = QtWidgets.QWidget()
        self.ax_line_w_w.setLayout(self.layout_ax_line_w)

        self.ax_spines_top_chbox = QtWidgets.QCheckBox("Top spine")
        self.ax_spines_bottom_chbox = QtWidgets.QCheckBox("Bottom spine")
        self.ax_spines_left_chbox = QtWidgets.QCheckBox("Left spine")
        self.ax_spines_right_chbox = QtWidgets.QCheckBox("Right spine")
        self.ax_spines_bottom_chbox.toggle()
        self.ax_spines_left_chbox.toggle()
        self.layout_ax_spines = QtWidgets.QHBoxLayout()
        self.layout_ax_spines.addWidget(self.ax_spines_top_chbox)
        self.layout_ax_spines.addWidget(self.ax_spines_bottom_chbox)
        self.layout_ax_spines.addWidget(self.ax_spines_left_chbox)
        self.layout_ax_spines.addWidget(self.ax_spines_right_chbox)
        self.ax_spines_w = QtWidgets.QWidget()
        self.ax_spines_w.setLayout(self.layout_ax_spines)

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

        self.fig_font_label = QtWidgets.QLabel("Figure font")
        self.def_fig_font = "Arial"
        self.fig_font_line = QtWidgets.QLineEdit(self.def_fig_font)
        self.layout_fig_font = QtWidgets.QHBoxLayout()
        self.layout_fig_font.addWidget(self.fig_font_label)
        self.layout_fig_font.addWidget(self.fig_font_line)
        self.fig_font_w = QtWidgets.QWidget()
        self.fig_font_w.setLayout(self.layout_fig_font)

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
        self.def_save_path = "Z:/test/"
        self.save_path_line = QtWidgets.QLineEdit(self.def_save_path)
        self.layout_save_path = QtWidgets.QHBoxLayout()
        self.layout_save_path.addWidget(self.save_path_label)
        self.layout_save_path.addWidget(self.save_path_line)
        self.save_path_w = QtWidgets.QWidget()
        self.save_path_w.setLayout(self.layout_save_path)

        self.replot_button = QtWidgets.QPushButton('Replot')
        self.replot_button.setStyleSheet("QPushButton{font-size: 30px;"
                                         "font-family: MS Sans Serif;"
                                         "color: rgb(255, 255, 255);"
                                         "background-color: rgb(204, 102, 102);}")
        self.replot_button.clicked.connect(self.on_click_replot_figure)

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
        self.layout.addWidget(self.plot_color_w)
        self.layout.addWidget(self.ax_line_w_w)
        self.layout.addWidget(self.ax_spines_w)
        self.layout.addWidget(self.x_label_w)
        self.layout.addWidget(self.y_label_w)
        self.layout.addWidget(self.fig_font_w)
        self.layout.addWidget(self.xy_label_fsize_w)
        self.layout.addWidget(self.ax_tick_fsize_w)
        self.layout.addWidget(self.save_path_w)
        self.layout.addWidget(self.replot_button)
        self.layout.addWidget(self.save_button)
        """
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
            plt.rcParams["font.family"] = str(self.fig_font_line.text())
            self.df = pd.read_csv(self.db.mineData, skiprows=int(self.skiprows_line.text()))
            self.plot()

    def plot(self):
        fig = plt.figure(figsize=(float(self.width_line.text()),
                                       float(self.height_line.text())),
                              dpi=float(self.dpi_line.text()))
        self.ax = fig.add_subplot(1, 1, 1)
        self.ax.plot(self.df[str(self.df.columns[0])], self.df[str(self.df.columns[3])],
                     color=str(self.plot_color_cmbox.currentText()),
                     linewidth=float(self.plot_line_w_line.text()))

        if self.ax_spines_top_chbox.isChecked():
            self.ax.spines["top"].set_linewidth(self.ax_line_w_line.text())
        else:
            self.ax.spines["top"].set_linewidth(0)
        if self.ax_spines_bottom_chbox.isChecked():
            self.ax.spines["bottom"].set_linewidth(self.ax_line_w_line.text())
        else:
            self.ax.spines["bottom"].set_linewidth(0)
        if self.ax_spines_left_chbox.isChecked():
            self.ax.spines["left"].set_linewidth(self.ax_line_w_line.text())
        else:
            self.ax.spines["left"].set_linewidth(0)
        if self.ax_spines_right_chbox.isChecked():
            self.ax.spines["right"].set_linewidth(self.ax_line_w_line.text())
        else:
            self.ax.spines["right"].set_linewidth(0)

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
        self.date = datetime.datetime.today()
        self.filename = ("fig" + str(self.date.year) + '_' + str(self.date.month) + '_' +
                         str(self.date.day) + '_' + str(self.date.hour) + '_' + str(self.date.minute) + '_' + str(
                         self.date.second)+ ".png")
        print(self.filename)
        plt.savefig(str(self.save_path_line.text())+str(self.filename))
        print("Saved!!!\n")

    def on_click_replot_figure(self):
        self.plot()

class DropButton(QPushButton):

    def __init__(self, title, parent=None):
        super().__init__(title, parent)
        self.mineData = "Null"
        self.drop_flg = False
        self.setAcceptDrops(True)


    def dragEnterEvent(self, e):
        self.mineData = e.mimeData().text()
        e.acceptProposedAction()

    def dropEvent(self, e):
        self.drop_flg = True
