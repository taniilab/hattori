import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QMenu, QSizePolicy, QMessageBox, QWidget, QPushButton, QLabel, QHBoxLayout, QVBoxLayout, QGroupBox, QSlider, QLineEdit
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from neuron import Neuron_HR as Neuron


class CentralWidget(QWidget):
    def __init__(self, parent=None):
        super(CentralWidget, self).__init__(parent)
        self.initUI()

    def initUI(self):
        self.glaph = PlotCanvas(None, width=5, height=4)
        navi = NavigationToolbar(self.glaph, self)

        # glaph
        layout_glaph = QVBoxLayout()
        layout_glaph.addWidget(self.glaph)
        layout_glaph.addWidget(navi)

        # HR model palm range
        self.a_max = 3
        self.a_min = -1
        self.b_max = 5
        self.b_min = 1
        self.c_max = 3
        self.c_min = 1
        self.d_max = 10
        self.d_min = 1
        self.r_max = 0.1
        self.r_min = 0.001
        self.s_max = 5
        self.s_min = 1
        self.xr_max = 3
        self.xr_min = -3
        self.i_max = 10
        self.i_min = -10
        self.gcmp_max = 20
        self.gcmp_min = 0
        self.delay_max = 150
        self.delay_min = 0

        self.slider_min = 1
        self.slider_max = 100

        # HR model palameter
        layout1_2 = QHBoxLayout()
        layout1 = QVBoxLayout()
        self.slider1 = QSlider(Qt.Horizontal)
        self.slider1.setRange(self.slider_min, self.slider_max)
        self.slider1.setValue(int(self.glaph.a - self.a_min) /
                              (self.a_max - self.a_min)*self.slider_max)
        self.label1 = QLabel('a :')
        layout1_2.addWidget(self.label1)
        layout1_2.addWidget(self.slider1)
        self.textbox1 = QLineEdit()
        self.textbox1.setText(str(self.glaph.a))
        layout1.addWidget(self.textbox1)
        layout1.addLayout(layout1_2)

        layout2_2 = QHBoxLayout()
        layout2 = QVBoxLayout()
        self.slider2 = QSlider(Qt.Horizontal)
        self.slider2.setRange(self.slider_min, self.slider_max)
        self.slider2.setValue(int(self.glaph.b - self.b_min) /
                              (self.b_max - self.b_min)*self.slider_max)
        self.label2 = QLabel('b :')
        layout2_2.addWidget(self.label2)
        layout2_2.addWidget(self.slider2)
        self.textbox2 = QLineEdit()
        self.textbox2.setText(str(self.glaph.b))
        layout2.addWidget(self.textbox2)
        layout2.addLayout(layout2_2)

        layout3_2 = QHBoxLayout()
        layout3 = QVBoxLayout()
        self.slider3 = QSlider(Qt.Horizontal)
        self.slider3.setRange(self.slider_min, self.slider_max)
        self.slider3.setValue(int(self.glaph.c - self.c_min) /
                              (self.c_max - self.c_min)*self.slider_max)
        self.label3 = QLabel('c :')
        layout3_2.addWidget(self.label3)
        layout3_2.addWidget(self.slider3)
        self.textbox3 = QLineEdit()
        self.textbox3.setText(str(self.glaph.c))
        layout3.addWidget(self.textbox3)
        layout3.addLayout(layout3_2)

        layout4_2 = QHBoxLayout()
        layout4 = QVBoxLayout()
        self.slider4 = QSlider(Qt.Horizontal)
        self.slider4.setRange(self.slider_min, self.slider_max)
        self.slider4.setValue(int(self.glaph.d - self.d_min) /
                              (self.d_max - self.d_min)*self.slider_max)
        self.label4 = QLabel('d :')
        layout4_2.addWidget(self.label4)
        layout4_2.addWidget(self.slider4)
        self.textbox4 = QLineEdit()
        self.textbox4.setText(str(self.glaph.d))
        layout4.addWidget(self.textbox4)
        layout4.addLayout(layout4_2)

        layout5_2 = QHBoxLayout()
        layout5 = QVBoxLayout()
        self.slider5 = QSlider(Qt.Horizontal)
        self.slider5.setRange(self.slider_min, self.slider_max)
        self.slider5.setValue(int(self.glaph.r - self.r_min) /
                              (self.r_max - self.r_min)*self.slider_max)
        self.label5 = QLabel('r :')
        layout5_2.addWidget(self.label5)
        layout5_2.addWidget(self.slider5)
        self.textbox5 = QLineEdit()
        self.textbox5.setText(str(self.glaph.r))
        layout5.addWidget(self.textbox5)
        layout5.addLayout(layout5_2)

        layout6_2 = QHBoxLayout()
        layout6 = QVBoxLayout()
        self.slider6 = QSlider(Qt.Horizontal)
        self.slider6.setRange(self.slider_min, self.slider_max)
        self.slider6.setValue(int(self.glaph.s - self.s_min) /
                              (self.s_max - self.s_min)*self.slider_max)
        self.label6 = QLabel('s :')
        layout6_2.addWidget(self.label6)
        layout6_2.addWidget(self.slider6)
        self.textbox6 = QLineEdit()
        self.textbox6.setText(str(self.glaph.s))
        layout6.addWidget(self.textbox6)
        layout6.addLayout(layout6_2)

        layout7_2 = QHBoxLayout()
        layout7 = QVBoxLayout()
        self.slider7 = QSlider(Qt.Horizontal)
        self.label7 = QLabel('xr :')
        self.slider7.setRange(self.slider_min, self.slider_max)
        self.slider7.setValue(int(self.glaph.xr - self.xr_min) /
                              (self.xr_max - self.xr_min)*self.slider_max)
        layout7_2.addWidget(self.label7)
        layout7_2.addWidget(self.slider7)
        self.textbox7 = QLineEdit()
        self.textbox7.setText(str(self.glaph.xr))
        layout7.addWidget(self.textbox7)
        layout7.addLayout(layout7_2)

        layout8_2 = QHBoxLayout()
        layout8 = QVBoxLayout()
        self.slider8 = QSlider(Qt.Horizontal)
        self.slider8.setRange(self.slider_min, self.slider_max)
        self.slider8.setValue(int(self.glaph.i - self.i_min) /
                              (self.i_max - self.i_min)*self.slider_max)
        self.label8 = QLabel('I :')
        layout8_2.addWidget(self.label8)
        layout8_2.addWidget(self.slider8)
        self.textbox8 = QLineEdit()
        self.textbox8.setText(str(self.glaph.i))
        layout8.addWidget(self.textbox8)
        layout8.addLayout(layout8_2)

        layout9_2 = QHBoxLayout()
        layout9 = QVBoxLayout()
        self.slider9 = QSlider(Qt.Horizontal)
        self.slider9.setRange(self.slider_min, self.slider_max)
        self.slider9.setValue(int(self.glaph.gcmp - self.gcmp_min) /
                              (self.gcmp_max - self.gcmp_min)*self.slider_max)
        self.label9 = QLabel('gcmp :')
        layout9_2.addWidget(self.label9)
        layout9_2.addWidget(self.slider9)
        self.textbox9 = QLineEdit()
        self.textbox9.setText(str(self.glaph.gcmp))
        layout9.addWidget(self.textbox9)
        layout9.addLayout(layout9_2)

        layout10_2 = QHBoxLayout()
        layout10 = QVBoxLayout()
        self.slider10 = QSlider(Qt.Horizontal)
        self.slider10.setRange(self.slider_min, self.slider_max)
        self.slider10.setValue(int(self.glaph.delay - self.delay_min) /
                              (self.delay_max - self.delay_min)*self.slider_max)
        self.label10 = QLabel('delay :')
        layout10_2.addWidget(self.label10)
        layout10_2.addWidget(self.slider10)
        self.textbox10 = QLineEdit()
        self.textbox10.setText(str(self.glaph.delay))
        layout10.addWidget(self.textbox10)
        layout10.addLayout(layout10_2)

        layout_palm = QVBoxLayout()
        layout_palm.addLayout(layout1)
        layout_palm.addLayout(layout2)
        layout_palm.addLayout(layout3)
        layout_palm.addLayout(layout4)
        layout_palm.addLayout(layout5)
        layout_palm.addLayout(layout6)
        layout_palm.addLayout(layout7)
        layout_palm.addLayout(layout8)
        layout_palm.addLayout(layout9)
        layout_palm.addLayout(layout10)

        groupBox1 = QGroupBox("palm")
        groupBox2 = QGroupBox("glaph")
        sizePolicy1 = groupBox1.sizePolicy()
        sizePolicy2 = groupBox2.sizePolicy()
        sizePolicy1.setHorizontalStretch(2)
        sizePolicy2.setHorizontalStretch(7)
        groupBox1.setSizePolicy(sizePolicy1)
        groupBox2.setSizePolicy(sizePolicy2)
        groupBox1.setLayout(layout_palm)
        groupBox2.setLayout(layout_glaph)

        layout_main = QHBoxLayout()
        layout_main.addWidget(groupBox1)
        layout_main.addWidget(groupBox2)
        self.setLayout(layout_main)

        # signal
        self.textbox1.textChanged.connect(self.text1_changed)
        self.textbox2.textChanged.connect(self.text2_changed)
        self.textbox3.textChanged.connect(self.text3_changed)
        self.textbox4.textChanged.connect(self.text4_changed)
        self.textbox5.textChanged.connect(self.text5_changed)
        self.textbox6.textChanged.connect(self.text6_changed)
        self.textbox7.textChanged.connect(self.text7_changed)
        self.textbox8.textChanged.connect(self.text8_changed)
        self.textbox9.textChanged.connect(self.text9_changed)
        self.textbox10.textChanged.connect(self.text10_changed)
        self.slider1.valueChanged.connect(self.slider1_changed)
        self.slider2.valueChanged.connect(self.slider2_changed)
        self.slider3.valueChanged.connect(self.slider3_changed)
        self.slider4.valueChanged.connect(self.slider4_changed)
        self.slider5.valueChanged.connect(self.slider5_changed)
        self.slider6.valueChanged.connect(self.slider6_changed)
        self.slider7.valueChanged.connect(self.slider7_changed)
        self.slider8.valueChanged.connect(self.slider8_changed)
        self.slider9.valueChanged.connect(self.slider9_changed)
        self.slider10.valueChanged.connect(self.slider10_changed)


    # slot
    def text1_changed(self):
        if self.textbox1.text() is "":
            self.glaph.replot_a(float(self.glaph.tmp_a))
        else:
            self.glaph.replot_a(float(self.textbox1.text()))

    def text2_changed(self):
        if self.textbox2.text() is "":
            self.glaph.replot_b(float(self.glaph.tmp_b))
        else:
            self.glaph.replot_b(float(self.textbox2.text()))

    def text3_changed(self):
        if self.textbox3.text() is "":
            self.glaph.replot_c(float(self.glaph.tmp_c))
        else:
            self.glaph.replot_c(float(self.textbox3.text()))

    def text4_changed(self):
        if self.textbox4.text() is "":
            self.glaph.replot_d(float(self.glaph.tmp_d))
        else:
            self.glaph.replot_d(float(self.textbox4.text()))

    def text5_changed(self):
        if self.textbox5.text() is "":
            self.glaph.replot_r(float(self.glaph.tmp_r))
        else:
            self.glaph.replot_r(float(self.textbox5.text()))

    def text6_changed(self):
        if self.textbox6.text() is "":
            self.glaph.replot_s(float(self.glaph.tmp_s))
        else:
            self.glaph.replot_s(float(self.textbox6.text()))

    def text7_changed(self):
        if self.textbox7.text() is "":
            self.glaph.replot_xr(float(self.glaph.tmp_xr))
        else:
            self.glaph.replot_xr(float(self.textbox7.text()))

    def text8_changed(self):
        if self.textbox8.text() is "":
            self.glaph.replot_i(float(self.glaph.tmp_i))
        else:
            self.glaph.replot_i(float(self.textbox8.text()))

    def text9_changed(self):
        if self.textbox9.text() is "":
            self.glaph.replot_gcmp(float(self.glaph.tmp_gcmp))
        else:
            self.glaph.replot_gcmp(float(self.textbox9.text()))

    def text10_changed(self):
        if self.textbox10.text() is "":
            self.glaph.replot_delay(float(self.glaph.tmp_delay))
        else:
            self.glaph.replot_delay(float(self.textbox10.text()))

    def slider1_changed(self):
        self.s1palm = round((self.slider1.value()/self.slider_max) *
                             (self.a_max-self.a_min)+self.a_min, 3)
        self.textbox1.setText(str(self.s1palm))

    def slider2_changed(self):
        self.s2palm = str((self.slider2.value()/self.slider_max) *
                          (self.b_max-self.b_min)+self.b_min)
        self.textbox2.setText(self.s2palm)

    def slider3_changed(self):
        self.s3palm = str((self.slider3.value()/self.slider_max) *
                          (self.c_max-self.c_min)+self.c_min)
        self.textbox3.setText(self.s3palm)

    def slider4_changed(self):
        self.s4palm = str((self.slider4.value()/self.slider_max) *
                          (self.d_max-self.d_min)+self.d_min)
        self.textbox4.setText(self.s4palm)

    def slider5_changed(self):
        self.s5palm = str((self.slider5.value()/self.slider_max) *
                          (self.r_max-self.r_min)+self.r_min)
        self.textbox5.setText(self.s5palm)

    def slider6_changed(self):
        self.s6palm = str((self.slider6.value()/self.slider_max) *
                          (self.s_max-self.s_min)+self.s_min)
        self.textbox6.setText(self.s6palm)

    def slider7_changed(self):
        self.s7palm = str((self.slider7.value()/self.slider_max) *
                          (self.xr_max-self.xr_min)+self.xr_min)
        self.textbox7.setText(self.s7palm)

    def slider8_changed(self):
        self.s8palm = str((self.slider8.value()/self.slider_max) *
                          (self.i_max-self.i_min)+self.i_min)
        self.textbox8.setText(self.s8palm)

    def slider9_changed(self):
        self.s9palm = str((self.slider9.value()/self.slider_max) *
                          (self.gcmp_max-self.gcmp_min)+self.gcmp_min)
        self.textbox9.setText(self.s9palm)

    def slider10_changed(self):
        self.s10palm = str((self.slider10.value()/self.slider_max) *
                           (self.delay_max-self.delay_min)+self.delay_min)
        self.textbox10.setText(self.s10palm)


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.left = 100
        self.top = 100
        self.title = 'Hindmarsh-Rose model - simulater'
        self.width = 2048
        self.height = 1900
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.setCentralWidget(CentralWidget(self))


class PlotCanvas(FigureCanvas):

    def __init__(self, parent=None, width=10, height=8, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.plot_init()

    def plot_init(self):
        self.nr = Neuron()

        self.a = 1
        self.b = 3
        self.c = 1
        self.d = 5
        self.r = 0.003
        self.s = 4
        self.i = 0
        self.xr = -1.56
        self.gcmp = 0
        self.delay = 0

        self.ax0 = self.figure.add_subplot(221)
        self.ax1 = self.figure.add_subplot(222)
        self.ax2 = self.figure.add_subplot(223)
        self.ax3 = self.figure.add_subplot(224)
        self.ax0.set_title('N0')
        self.ax1.set_title('N1')
        self.ax2.set_title('N2')
        self.ax3.set_title('N3')
        plt.title('Hindmarsh-Rose model')

        self.tmp_a = self.a
        self.tmp_b = self.b
        self.tmp_c = self.c
        self.tmp_d = self.d
        self.tmp_r = self.r
        self.tmp_s = self.s
        self.tmp_xr = self.xr
        self.tmp_i = self.i
        self.tmp_gcmp = self.gcmp
        self.tmp_delay = self.delay

        self.plot(self.a, self.b, self.c, self.d, self.r, self.s, self.xr,
                  self.i, self.gcmp, self.delay)

    def replot_a(self, a):
        self.line0.remove()
        self.line1.remove()
        self.line2.remove()
        self.line3.remove()
        self.tmp_a = a
        self.plot(self.tmp_a, self.b, self.c, self.d, self.r, self.s, self.xr,
                  self.i, self.gcmp, self.delay)

    def replot_b(self, b):
        self.line0.remove()
        self.line1.remove()
        self.line2.remove()
        self.line3.remove()
        self.tmp_b = b
        self.plot(self.a, self.tmp_b, self.c, self.d, self.r, self.s, self.xr,
                  self.i, self.gcmp, self.delay)

    def replot_c(self, c):
        self.line0.remove()
        self.line1.remove()
        self.line2.remove()
        self.line3.remove()
        self.tmp_c = c
        self.plot(self.a, self.b, self.tmp_c, self.d, self.r, self.s, self.xr,
                  self.i, self.gcmp, self.delay)

    def replot_d(self, d):
        self.line0.remove()
        self.line1.remove()
        self.line2.remove()
        self.line3.remove()
        self.tmp_d = d
        self.plot(self.a, self.b, self.c, self.tmp_d, self.r, self.s, self.xr,
                  self.i, self.gcmp, self.delay)

    def replot_r(self, r):
        self.line0.remove()
        self.line1.remove()
        self.line2.remove()
        self.line3.remove()
        self.tmp_r = r
        self.plot(self.a, self.b, self.c, self.d, self.tmp_r, self.s, self.xr,
                  self.i, self.gcmp, self.delay)

    def replot_s(self, s):
        self.line0.remove()
        self.line1.remove()
        self.line2.remove()
        self.line3.remove()
        self.tmp_s = s
        self.plot(self.a, self.b, self.c, self.d, self.r, self.tmp_s, self.xr,
                  self.i, self.gcmp, self.delay)

    def replot_xr(self, xr):
        self.line0.remove()
        self.line1.remove()
        self.line2.remove()
        self.line3.remove()
        self.tmp_xr = xr
        self.plot(self.a, self.b, self.c, self.d, self.r, self.s, self.tmp_xr,
                  self.i, self.gcmp, self.delay)

    def replot_i(self, i):
        self.line0.remove()
        self.line1.remove()
        self.line2.remove()
        self.line3.remove()
        self.tmp_i = i
        self.plot(self.a, self.b, self.c, self.d, self.r, self.s, self.xr,
                  self.tmp_i, self.gcmp, self.delay)

    def replot_gcmp(self, gcmp):
        self.line0.remove()
        self.line1.remove()
        self.line2.remove()
        self.line3.remove()
        self.tmp_gcmp = gcmp
        self.plot(self.a, self.b, self.c, self.d, self.r, self.s, self.xr,
                  self.i, self.tmp_gcmp, self.delay)

    def replot_delay(self, delay):
        self.line0.remove()
        self.line1.remove()
        self.line2.remove()
        self.line3.remove()
        self.tmp_delay = delay
        self.plot(self.a, self.b, self.c, self.d, self.r, self.s, self.xr,
                  self.i, self.gcmp, self.tmp_delay)

    def plot(self, a, b, c, d, r, s, xr, i, gcmp, delay):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.r = r
        self.s = s
        self.xr = xr
        self.i = i
        self.gcmp = gcmp
        self.delay = delay

        self.number = 4


        self.palm = {"numneu": self.number,
                     "a": self.a,
                     "b": self.b,
                     "c": self.c,
                     "d": self.d,
                     "r": self.r,
                     "s": self.s,
                     "xr": self.xr,
                     "gcmp": self.gcmp,
                     "delay":self.delay,
                     "Iext_amp": self.i,
                     "Iext_width": 10,
                     "Iext_num": 1000,
                     "Syncp": 5,
                     "noise": 1}

        self.nr.__init__(**self.palm)

        
        for i in range(0, self.nr.allsteps-1):
            if (self.nr.curstep * self.nr.dt) > 200:
                for j in range(self.number-1):
                    self.nr.cnct[j, j+1] = 1.0
                    self.nr.cnct[j+1, j] = 1.0
                pass
            self.nr.propagation()

        self.line0, = self.ax0.plot(self.nr.tmhist, self.nr.x[0], "b")
        self.line1, = self.ax1.plot(self.nr.tmhist, self.nr.x[1], "b")
        self.line2, = self.ax2.plot(self.nr.tmhist, self.nr.x[2], "b")
        self.line3, = self.ax3.plot(self.nr.tmhist, self.nr.x[3], "b")

        self.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mw = MainWindow()
    mw.show()
    sys.exit(app.exec_())
