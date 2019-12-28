# -*- coding: utf-8 -*-
# CAWAII Plotter
# programmed by Kouhei Hattori of Waseda University

"""
今後実装予定(時期未定)の機能
・二軸プロット
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
import datetime
import csv

class Ui_MainWindow(object):
    timer: QTimer
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(640, 960)
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralWidget.sizePolicy().hasHeightForWidth())
        self.centralWidget.setSizePolicy(sizePolicy)
        self.centralWidget.setObjectName("centralWidget")
        self.centralWidget.setStyleSheet("QLabel {font: 12pt Arial}"
                                         "QComboBox {font: 12pt Arial; font-weight: bold}"
                                         "QCheckBox {font: 12pt Arial; font-weight: bold}"
                                         "QLineEdit {font: 12pt Arial; font-weight: bold}")

        self.layout = QtWidgets.QVBoxLayout()
        self.dd_button = DropButton("ここにCSVファイルをかわいくドラッグして下さいね")
        self.dd_button.setStyleSheet("QPushButton{font-size: 30px;"
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

        self.xrow_label = QtWidgets.QLabel("X row (default → 0th)")
        self.def_xrow = "0"
        self.xrow_line = QtWidgets.QLineEdit(self.def_xrow)
        self.layout_xrow = QtWidgets.QHBoxLayout()
        self.layout_xrow.addWidget(self.xrow_label)
        self.layout_xrow.addWidget(self.xrow_line)
        self.xrow_w = QtWidgets.QWidget()
        self.xrow_w.setLayout(self.layout_xrow)

        self.yrow_label = QtWidgets.QLabel("Y row (up to 6 rows)")
        self.def_yrow = "1, 2"
        self.yrow_line = QtWidgets.QLineEdit(self.def_yrow)
        self.layout_yrow = QtWidgets.QHBoxLayout()
        self.layout_yrow.addWidget(self.yrow_label)
        self.layout_yrow.addWidget(self.yrow_line)
        self.yrow_w = QtWidgets.QWidget()
        self.yrow_w.setLayout(self.layout_yrow)

        self.xcoefficient_label = QtWidgets.QLabel("Coefficient (X)")
        self.def_xcoefficient = "1"
        self.ycoefficient_label = QtWidgets.QLabel("Coefficient (Y)")
        self.def_ycoefficient = "1"
        self.ycoefficient_line = QtWidgets.QLineEdit(self.def_ycoefficient)
        self.xcoefficient_line = QtWidgets.QLineEdit(self.def_xcoefficient)
        self.layout_coefficient = QtWidgets.QHBoxLayout()
        self.layout_coefficient.addWidget(self.xcoefficient_label)
        self.layout_coefficient.addWidget(self.xcoefficient_line)
        self.layout_coefficient.addWidget(self.ycoefficient_label)
        self.layout_coefficient.addWidget(self.ycoefficient_line)
        self.coefficient_w = QtWidgets.QWidget()
        self.coefficient_w.setLayout(self.layout_coefficient)

        self.stimtiming_label = QtWidgets.QLabel("Stimulation timing [s]")
        self.def_stimtiming = ""
        self.stimtiming_line = QtWidgets.QLineEdit(self.def_stimtiming)
        self.dd_stimtiming_button = DropButton("CSVファイル(刺激タイミング)")
        self.dd_stimtiming_button.setStyleSheet("QPushButton{font-size: 15px;"
                                                "font-weight: bold;"
                                                "font-family: MS Sans Serif;"
                                                "color: rgb(255, 255, 255);"
                                                "background-color: rgb(204, 102, 102);}")
        self.layout_stimtiming = QtWidgets.QHBoxLayout()
        self.layout_stimtiming.addWidget(self.stimtiming_label)
        self.layout_stimtiming.addWidget(self.stimtiming_line)
        self.layout_stimtiming.addWidget(self.dd_stimtiming_button)
        self.stimtiming_w = QtWidgets.QWidget()
        self.stimtiming_w.setLayout(self.layout_stimtiming)

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
        self.height_label = QtWidgets.QLabel("Height [inch]")
        self.def_height = "2"
        self.width_line = QtWidgets.QLineEdit(self.def_width)
        self.height_line = QtWidgets.QLineEdit(self.def_height)
        self.layout_width_height = QtWidgets.QHBoxLayout()
        self.layout_width_height.addWidget(self.width_label)
        self.layout_width_height.addWidget(self.width_line)
        self.layout_width_height.addWidget(self.height_label)
        self.layout_width_height.addWidget(self.height_line)
        self.width_height_w = QtWidgets.QWidget()
        self.width_height_w.setLayout(self.layout_width_height)

        self.plot_line_w_label = QtWidgets.QLabel("Plot line-width")
        self.def_plot_line_w = "1"
        self.plot_line_w_line = QtWidgets.QLineEdit(self.def_plot_line_w)
        self.layout_plot_line_w = QtWidgets.QHBoxLayout()
        self.layout_plot_line_w.addWidget(self.plot_line_w_label)
        self.layout_plot_line_w.addWidget(self.plot_line_w_line)
        self.plot_line_w_w = QtWidgets.QWidget()
        self.plot_line_w_w.setLayout(self.layout_plot_line_w)

        self.stim_line_w_label = QtWidgets.QLabel("Stim. line-width")
        self.def_stim_line_w = "0.3"
        self.stim_line_w_line = QtWidgets.QLineEdit(self.def_stim_line_w)
        self.layout_stim_line_w = QtWidgets.QHBoxLayout()
        self.layout_stim_line_w.addWidget(self.stim_line_w_label)
        self.layout_stim_line_w.addWidget(self.stim_line_w_line)
        self.stim_line_w_w = QtWidgets.QWidget()
        self.stim_line_w_w.setLayout(self.layout_stim_line_w)

        ### up to 7 plots ###
        self.plot_color_label = []
        self.plot_color_label.append(QtWidgets.QLabel("Plot line-color1"))
        self.plot_color_label.append(QtWidgets.QLabel("Plot line-color2"))
        self.plot_color_label.append(QtWidgets.QLabel("Plot line-color3"))
        self.plot_color_label.append(QtWidgets.QLabel("Plot line-color4"))
        self.plot_color_label.append(QtWidgets.QLabel("Plot line-color5"))
        self.plot_color_label.append(QtWidgets.QLabel("Plot line-color6"))
        self.plot_color_label.append(QtWidgets.QLabel("Plot line-color7(Stim. Tim.)"))
        self.plot_color_cmbox = []
        self.plot_color_cmbox.append(QtWidgets.QComboBox())
        self.plot_color_cmbox.append(QtWidgets.QComboBox())
        self.plot_color_cmbox.append(QtWidgets.QComboBox())
        self.plot_color_cmbox.append(QtWidgets.QComboBox())
        self.plot_color_cmbox.append(QtWidgets.QComboBox())
        self.plot_color_cmbox.append(QtWidgets.QComboBox())
        self.plot_color_cmbox.append(QtWidgets.QComboBox())
        self.def_plot_color = "black"
        self.cl = Color_list()
        self.color_items = self.cl.list()
        self.color_items = sorted(self.color_items)
        self.plot_color_cmbox[0].addItems(self.color_items)
        self.plot_color_cmbox[1].addItems(self.color_items)
        self.plot_color_cmbox[2].addItems(self.color_items)
        self.plot_color_cmbox[3].addItems(self.color_items)
        self.plot_color_cmbox[4].addItems(self.color_items)
        self.plot_color_cmbox[5].addItems(self.color_items)
        self.plot_color_cmbox[6].addItems(self.color_items)
        self.plot_color_cmbox[0].setCurrentText(self.def_plot_color)
        self.plot_color_cmbox[1].setCurrentText(self.def_plot_color)
        self.plot_color_cmbox[2].setCurrentText(self.def_plot_color)
        self.plot_color_cmbox[3].setCurrentText(self.def_plot_color)
        self.plot_color_cmbox[4].setCurrentText(self.def_plot_color)
        self.plot_color_cmbox[5].setCurrentText(self.def_plot_color)
        self.plot_color_cmbox[6].setCurrentText(self.def_plot_color)
        self.plot_color_style_label = []
        self.plot_color_style_label.append(QtWidgets.QLabel("Linestyle1"))
        self.plot_color_style_label.append(QtWidgets.QLabel("Linestyle2"))
        self.plot_color_style_label.append(QtWidgets.QLabel("Linestyle3"))
        self.plot_color_style_label.append(QtWidgets.QLabel("Linestyle4"))
        self.plot_color_style_label.append(QtWidgets.QLabel("Linestyle5"))
        self.plot_color_style_label.append(QtWidgets.QLabel("Linestyle6"))
        self.plot_color_style_label.append(QtWidgets.QLabel("Linestyle7"))
        self.plot_color_style_cmbox = []
        self.plot_color_style_cmbox.append(QtWidgets.QComboBox())
        self.plot_color_style_cmbox.append(QtWidgets.QComboBox())
        self.plot_color_style_cmbox.append(QtWidgets.QComboBox())
        self.plot_color_style_cmbox.append(QtWidgets.QComboBox())
        self.plot_color_style_cmbox.append(QtWidgets.QComboBox())
        self.plot_color_style_cmbox.append(QtWidgets.QComboBox())
        self.plot_color_style_cmbox.append(QtWidgets.QComboBox())
        self.def_plot_color_style = "solid"
        self.color_style_items = {"solid", "dashed", "dashdot", "dotted"}
        self.color_style_items = sorted(self.color_style_items)
        self.plot_color_style_cmbox[0].addItems(self.color_style_items)
        self.plot_color_style_cmbox[1].addItems(self.color_style_items)
        self.plot_color_style_cmbox[2].addItems(self.color_style_items)
        self.plot_color_style_cmbox[3].addItems(self.color_style_items)
        self.plot_color_style_cmbox[4].addItems(self.color_style_items)
        self.plot_color_style_cmbox[5].addItems(self.color_style_items)
        self.plot_color_style_cmbox[6].addItems(self.color_style_items)
        self.plot_color_style_cmbox[0].setCurrentText(self.def_plot_color_style)
        self.plot_color_style_cmbox[1].setCurrentText(self.def_plot_color_style)
        self.plot_color_style_cmbox[2].setCurrentText(self.def_plot_color_style)
        self.plot_color_style_cmbox[3].setCurrentText(self.def_plot_color_style)
        self.plot_color_style_cmbox[4].setCurrentText(self.def_plot_color_style)
        self.plot_color_style_cmbox[5].setCurrentText(self.def_plot_color_style)
        self.plot_color_style_cmbox[6].setCurrentText(self.def_plot_color_style)

        self.layout_plot_color = []
        self.layout_plot_color.append(QtWidgets.QHBoxLayout())
        self.layout_plot_color.append(QtWidgets.QHBoxLayout())
        self.layout_plot_color.append(QtWidgets.QHBoxLayout())
        self.layout_plot_color.append(QtWidgets.QHBoxLayout())
        self.layout_plot_color.append(QtWidgets.QHBoxLayout())
        self.layout_plot_color.append(QtWidgets.QHBoxLayout())
        self.layout_plot_color.append(QtWidgets.QHBoxLayout())
        self.layout_plot_color[0].addWidget(self.plot_color_label[0])
        self.layout_plot_color[0].addWidget(self.plot_color_cmbox[0])
        self.layout_plot_color[0].addWidget(self.plot_color_style_label[0])
        self.layout_plot_color[0].addWidget(self.plot_color_style_cmbox[0])
        self.layout_plot_color[1].addWidget(self.plot_color_label[1])
        self.layout_plot_color[1].addWidget(self.plot_color_cmbox[1])
        self.layout_plot_color[1].addWidget(self.plot_color_style_label[1])
        self.layout_plot_color[1].addWidget(self.plot_color_style_cmbox[1])
        self.layout_plot_color[2].addWidget(self.plot_color_label[2])
        self.layout_plot_color[2].addWidget(self.plot_color_cmbox[2])
        self.layout_plot_color[2].addWidget(self.plot_color_style_label[2])
        self.layout_plot_color[2].addWidget(self.plot_color_style_cmbox[2])
        self.layout_plot_color[3].addWidget(self.plot_color_label[3])
        self.layout_plot_color[3].addWidget(self.plot_color_cmbox[3])
        self.layout_plot_color[3].addWidget(self.plot_color_style_label[3])
        self.layout_plot_color[3].addWidget(self.plot_color_style_cmbox[3])
        self.layout_plot_color[4].addWidget(self.plot_color_label[4])
        self.layout_plot_color[4].addWidget(self.plot_color_cmbox[4])
        self.layout_plot_color[4].addWidget(self.plot_color_style_label[4])
        self.layout_plot_color[4].addWidget(self.plot_color_style_cmbox[4])
        self.layout_plot_color[5].addWidget(self.plot_color_label[5])
        self.layout_plot_color[5].addWidget(self.plot_color_cmbox[5])
        self.layout_plot_color[5].addWidget(self.plot_color_style_label[5])
        self.layout_plot_color[5].addWidget(self.plot_color_style_cmbox[5])
        self.layout_plot_color[6].addWidget(self.plot_color_label[6])
        self.layout_plot_color[6].addWidget(self.plot_color_cmbox[6])
        self.layout_plot_color[6].addWidget(self.plot_color_style_label[6])
        self.layout_plot_color[6].addWidget(self.plot_color_style_cmbox[6])
        self.plot_color_w = []
        self.plot_color_w.append(QtWidgets.QWidget())
        self.plot_color_w.append(QtWidgets.QWidget())
        self.plot_color_w.append(QtWidgets.QWidget())
        self.plot_color_w.append(QtWidgets.QWidget())
        self.plot_color_w.append(QtWidgets.QWidget())
        self.plot_color_w.append(QtWidgets.QWidget())
        self.plot_color_w.append(QtWidgets.QWidget())
        self.plot_color_w[0].setLayout(self.layout_plot_color[0])
        self.plot_color_w[1].setLayout(self.layout_plot_color[1])
        self.plot_color_w[2].setLayout(self.layout_plot_color[2])
        self.plot_color_w[3].setLayout(self.layout_plot_color[3])
        self.plot_color_w[4].setLayout(self.layout_plot_color[4])
        self.plot_color_w[5].setLayout(self.layout_plot_color[5])
        self.plot_color_w[6].setLayout(self.layout_plot_color[6])
        #####################

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
                                       "background-color: rgb(204, 102, 102);}")
        self.save_button.clicked.connect(self.on_click_savefig)

        self.save_setting_label = QtWidgets.QLabel("Settings")
        self.save_setting_label.setStyleSheet("QLabel{font-size: 30px;"
                                         "font-family: MS Sans Serif;"
                                         "color: rgb(0, 0, 0);"
                                         "font-weight: bold;}")

        self.save_setting1_button = QtWidgets.QPushButton('Save1')
        self.save_setting1_button.setStyleSheet("QPushButton{font-size: 30px;"
                                         "font-family: MS Sans Serif;"
                                         "color: rgb(255, 255, 255);"
                                         "background-color: rgb(65, 65, 65);}")
        self.save_setting1_button.clicked.connect(self.save_setting1)
        self.save_setting2_button = QtWidgets.QPushButton('Save2')
        self.save_setting2_button.setStyleSheet("QPushButton{font-size: 30px;"
                                         "font-family: MS Sans Serif;"
                                         "color: rgb(255, 255, 255);"
                                         "background-color: rgb(65, 65, 65);}")
        self.save_setting2_button.clicked.connect(self.save_setting2)
        self.save_setting3_button = QtWidgets.QPushButton('Save3')
        self.save_setting3_button.setStyleSheet("QPushButton{font-size: 30px;"
                                         "font-family: MS Sans Serif;"
                                         "color: rgb(255, 255, 255);"
                                         "background-color: rgb(65, 65, 65);}")
        self.save_setting3_button.clicked.connect(self.save_setting3)
        self.layout_save_setting = QtWidgets.QHBoxLayout()
        self.layout_save_setting.addWidget(self.save_setting1_button)
        self.layout_save_setting.addWidget(self.save_setting2_button)
        self.layout_save_setting.addWidget(self.save_setting3_button)
        self.save_setting_button_w = QtWidgets.QWidget()
        self.save_setting_button_w.setLayout(self.layout_save_setting)

        self.load_setting1_button = QtWidgets.QPushButton('Load1')
        self.load_setting1_button.setStyleSheet("QPushButton{font-size: 30px;"
                                         "font-family: MS Sans Serif;"
                                         "color: rgb(255, 255, 255);"
                                         "background-color: rgb(65, 65, 65);}")
        self.load_setting1_button.clicked.connect(self.load_setting1)
        self.load_setting2_button = QtWidgets.QPushButton('Load2')
        self.load_setting2_button.setStyleSheet("QPushButton{font-size: 30px;"
                                         "font-family: MS Sans Serif;"
                                         "color: rgb(255, 255, 255);"
                                         "background-color: rgb(65, 65, 65);}")
        self.load_setting2_button.clicked.connect(self.load_setting2)
        self.load_setting3_button = QtWidgets.QPushButton('Load3')
        self.load_setting3_button.setStyleSheet("QPushButton{font-size: 30px;"
                                         "font-family: MS Sans Serif;"
                                         "color: rgb(255, 255, 255);"
                                         "background-color: rgb(65, 65, 65);}")
        self.load_setting3_button.clicked.connect(self.load_setting3)
        self.layout_load_setting = QtWidgets.QHBoxLayout()
        self.layout_load_setting.addWidget(self.load_setting1_button)
        self.layout_load_setting.addWidget(self.load_setting2_button)
        self.layout_load_setting.addWidget(self.load_setting3_button)
        self.load_setting_button_w = QtWidgets.QWidget()
        self.load_setting_button_w.setLayout(self.layout_load_setting)

        self.layout.addWidget(self.dd_button)
        self.parameter_setting_area = QtWidgets.QScrollArea()
        self.scroll_layout = QtWidgets.QVBoxLayout()
        self.inner = QtWidgets.QWidget()
        self.parameter_setting_area.setWidgetResizable(True)
        self.parameter_setting_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.parameter_setting_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.inner.setLayout(self.scroll_layout)
        self.parameter_setting_area.setWidget(self.inner)
        self.layout.addWidget(self.parameter_setting_area)
        self.scroll_layout.addWidget(self.skiprows_w)
        self.scroll_layout.addWidget(self.xrow_w)
        self.scroll_layout.addWidget(self.yrow_w)
        self.scroll_layout.addWidget(self.coefficient_w)
        self.scroll_layout.addWidget(self.stimtiming_w)
        self.scroll_layout.addWidget(self.dpi_w)
        self.scroll_layout.addWidget(self.width_height_w)
        self.scroll_layout.addWidget(self.plot_line_w_w)
        self.scroll_layout.addWidget(self.stim_line_w_w)
        self.scroll_layout.addWidget(self.plot_color_w[0])
        self.scroll_layout.addWidget(self.plot_color_w[1])
        self.scroll_layout.addWidget(self.plot_color_w[2])
        self.scroll_layout.addWidget(self.plot_color_w[3])
        self.scroll_layout.addWidget(self.plot_color_w[4])
        self.scroll_layout.addWidget(self.plot_color_w[5])
        self.scroll_layout.addWidget(self.plot_color_w[6])
        self.scroll_layout.addWidget(self.ax_line_w_w)
        self.scroll_layout.addWidget(self.ax_spines_w)
        self.scroll_layout.addWidget(self.x_label_w)
        self.scroll_layout.addWidget(self.y_label_w)
        self.scroll_layout.addWidget(self.fig_font_w)
        self.scroll_layout.addWidget(self.xy_label_fsize_w)
        self.scroll_layout.addWidget(self.ax_tick_fsize_w)
        self.scroll_layout.addWidget(self.save_path_w)
        self.layout.addWidget(self.replot_button)
        self.layout.addWidget(self.save_button)
        self.layout.addWidget(self.save_setting_label)
        self.layout.addWidget(self.save_setting_button_w)
        self.layout.addWidget(self.load_setting_button_w)

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
        if self.dd_button.drop_flg == True:
            self.dd_button.drop_flg = False
            print(self.dd_button.mineData)
            plt.rcParams["font.family"] = str(self.fig_font_line.text())
            self.df = pd.read_csv(self.dd_button.mineData, skiprows=int(self.skiprows_line.text()))
            self.plot()

        if self.dd_stimtiming_button.drop_flg == True:
            self.dd_stimtiming_button.drop_flg = False
            print(self.dd_stimtiming_button.mineData)
            print("")
            self.df_stimtiming = pd.read_csv(self.dd_stimtiming_button.mineData,
                                             header=None,
                                             delimiter="x")# other than comma
            self.stimtiming_line.setText(str(self.df_stimtiming[0][0]))

    def plot(self):
        fig = plt.figure(figsize=(float(self.width_line.text()),
                                       float(self.height_line.text())),
                              dpi=float(self.dpi_line.text()))
        self.ax = fig.add_subplot(1, 1, 1)
        print(self.xrow_line.text())
        print(type(self.xrow_line.text()))
        self.selected_xrow = self.xrow_line.text()
        self.selected_yrow = [x.strip() for x in self.yrow_line.text().split(',')]

        if self.stimtiming_line.text() != "":
            self.selected_stimtiming = [x.strip() for x in self.stimtiming_line.text().split(',')]
            self.selected_stimtiming = [float(n) for n in self.selected_stimtiming]  # str to float
            print(self.selected_stimtiming)
            for i in range(len(self.selected_stimtiming)):
                self.ax.axvline(self.selected_stimtiming[i],
                                color=str(self.plot_color_cmbox[6].currentText()),
                                linewidth=float(self.stim_line_w_line.text()),
                                linestyle=str(self.plot_color_style_cmbox[6].currentText()))

        for i in range(len(self.selected_yrow)):
            self.ax.plot(self.df[str(self.df.columns[int(self.selected_xrow)])]*float(self.xcoefficient_line.text()),
                         self.df[str(self.df.columns[int(self.selected_yrow[i])])]*float(self.ycoefficient_line.text()),
                         color=str(self.plot_color_cmbox[i].currentText()),
                         linewidth=float(self.plot_line_w_line.text()),
                         linestyle=str(self.plot_color_style_cmbox[i].currentText()))


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

    def save_profile(self, cfg_path):
        data = []
        data.append(int(self.skiprows_line.text()))
        data.append(int(self.dpi_line.text()))
        data.append(float(self.width_line.text()))
        data.append(float(self.height_line.text()))
        data.append(float(self.plot_line_w_line.text()))
        data.append(str(self.plot_color_cmbox[0].currentText()))
        data.append(str(self.plot_color_cmbox[1].currentText()))
        data.append(str(self.plot_color_cmbox[2].currentText()))
        data.append(str(self.plot_color_cmbox[3].currentText()))
        data.append(str(self.plot_color_cmbox[4].currentText()))
        data.append(str(self.plot_color_cmbox[5].currentText()))
        data.append(str(self.plot_color_cmbox[6].currentText()))
        data.append(float(self.ax_line_w_line.text()))
        data.append(int(self.ax_spines_top_chbox.isChecked()))
        data.append(int(self.ax_spines_bottom_chbox.isChecked()))
        data.append(int(self.ax_spines_left_chbox.isChecked()))
        data.append(int(self.ax_spines_right_chbox.isChecked()))
        data.append(str(self.x_label_line.text()))
        data.append(str(self.y_label_line.text()))
        data.append(str(self.fig_font_line.text()))
        data.append(int(self.xy_label_fsize_line.text()))
        data.append(int(self.ax_tick_fsize_line.text()))
        data.append(str(self.save_path_line.text()))
        with open(cfg_path, 'w', newline="") as f:
            writer = csv.writer(f)
            writer.writerow(data)
            print(data)
            print("Setting saved\n")
        return

    def load_profile(self,cfg_path):
        if not os.path.isfile(cfg_path):
            print("not exist")
            return
        with open(cfg_path,'r') as f:
            reader = csv.reader(f)
            l = [row for row in reader]
            self.skiprows_line.setText(str(l[0][0]))
            self.dpi_line.setText(str(l[0][1]))
            self.width_line.setText(str(l[0][2]))
            self.height_line.setText(str(l[0][3]))
            self.plot_line_w_line.setText(str(l[0][4]))
            self.plot_color_cmbox[0].setCurrentText(str(l[0][5]))
            self.plot_color_cmbox[1].setCurrentText(str(l[0][6]))
            self.plot_color_cmbox[2].setCurrentText(str(l[0][7]))
            self.plot_color_cmbox[3].setCurrentText(str(l[0][8]))
            self.plot_color_cmbox[4].setCurrentText(str(l[0][9]))
            self.plot_color_cmbox[5].setCurrentText(str(l[0][10]))
            self.plot_color_cmbox[6].setCurrentText(str(l[0][11]))
            self.ax_line_w_line.setText(str(l[0][12]))
            self.ax_spines_top_chbox.setChecked(int(l[0][13]))
            self.ax_spines_bottom_chbox.setChecked(int(l[0][14]))
            self.ax_spines_left_chbox.setChecked(int(l[0][15]))
            self.ax_spines_right_chbox.setChecked(int(l[0][16]))
            self.x_label_line.setText(str(l[0][17]))
            self.y_label_line.setText(str(l[0][18]))
            self.fig_font_line.setText(str(l[0][19]))
            self.xy_label_fsize_line.setText(str(l[0][20]))
            self.ax_tick_fsize_line.setText(str(l[0][21]))
            self.save_path_line.setText(str(l[0][22]))
            print(l)
            print("Setting loaded\n")
        return

    def save_previous_setting(self):
        self.save_profile(os.getcwd()+'/previous.cfg')

    def load_previous_setting(self):
        self.load_profile(os.getcwd()+'/previous.cfg')

    def save_setting1(self):
        self.save_profile(os.getcwd()+'/save1.cfg')

    def load_setting1(self):
        self.load_profile(os.getcwd() + '/save1.cfg')

    def save_setting2(self):
        self.save_profile(os.getcwd() + '/save2.cfg')

    def load_setting2(self):
        self.load_profile(os.getcwd() + '/save2.cfg')

    def save_setting3(self):
        self.save_profile(os.getcwd() + '/save3.cfg')

    def load_setting3(self):
        self.load_profile(os.getcwd() + '/save3.cfg')

    def on_click_savefig(self):
        self.date = datetime.datetime.today()
        self.filename = ("fig" + str(self.date.year) + '_' + str(self.date.month) + '_' +
                         str(self.date.day) + '_' + str(self.date.hour) + '_' + str(self.date.minute) + '_' + str(
                         self.date.second)+ ".png")
        if os.path.isdir(str(self.save_path_line.text())):
            self.save_path_tmp = str(self.save_path_line.text())+str(self.filename)
        else:
            self.save_path_tmp = os.path.dirname(self.dd_button.mineData.strip("file:///"))+"/"+str(self.filename)
            print("Target directory does not exist !")
        plt.savefig(self.save_path_tmp)
        print(self.save_path_tmp)
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

class Color_list():

    def __init__(self):
        pass

    def list(self):
        return {"black", "dimgrey", "dimgray", "gray", "grey", "darkgrey", "darkgray", "silver",
                "lightgray", "lightgrey", "gainsboro", "whitesmoke", "white", "snow", "rosybrown",
                "lightcoral", "indianred", "brown", "firebrick", "maroon", "darkred", "red", "mistyrose",
                "salmon", "tomato", "darksalmon", "coral", "orangered", "lightsalmon", "sienna",
                "seashell", "chocolate", "saddlebrown", "sandybrown", "peachpuff", "peru", "linen", "bisque",
                "darkorange", "burlywood", "antiquewhite", "tan", "navajowhite", "wheat", "oldlace",
                "floralwhite", "darkgoldenrod", "goldenrod", "cornsilk", "gold", "lemonchiffon", "khaki",
                "palegoldenrod", "darkkhaki", "ivory", "beige", "lightyellow", "lightgoldenrodyellow", "olive",
                "yellow", "olivedrab", "lawngreen", "darkolivegreen", "greenyellow", "chartreuse",
                "lawngreen", "sage", "lightsage", "darksage", "honeydew", "darkseagreen", "palegreen",
                "lightgreen", "forestgreen", "limegreen", "darkgreen", "green", "lime", "seagreen",
                "mediumseagreen", "springgreen", "mintcream", "mediumspringgreen", "mediumaquamarine",
                "aquamarine", "turquoise", "lightseagreen", "mediumturquoise", "azure", "lightcyan",
                "paleturquoise", "darkslategray", "darkslategrey", "teal", "darkcyan", "aqua", "cyan",
                "darkturquoise", "cadetblue", "powderblue", "lightblue", "deepskyblue", "skyblue",
                "lightskyblue", "steelblue", "aliceblue", "dodgerblue", "lightslategrey", "lightslategray",
                "slategray", "slategrey", "lightsteelblue", "cornflowerblue", "royalblue", "ghostwhite",
                "lavender", "midnightblue", "navy", "darkblue", "mediumblue", "blue", "slateblue",
                "darkslateblue", "mediumslateblue", "mediumpurple", "blueviolet", "indigo", "darkorchid",
                "darkviolet", "mediumorchid", "thistle", "plum", "violet", "purple", "darkmagenta", "fuchsia",
                "magenta", "orchid", "mediumvioletred", "deeppink", "hotpink", "lavenderblush", "palevioletred",
                "crimson", "pink", "lightpink"}