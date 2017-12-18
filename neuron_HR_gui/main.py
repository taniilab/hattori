import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QMenu, QSizePolicy, QMessageBox, QWidget, QPushButton, QLabel, QHBoxLayout, QVBoxLayout, QGroupBox, QSlider, QLineEdit
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


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

        # HR model palameter
        layout1_2 = QHBoxLayout()
        layout1 = QVBoxLayout()
        self.slider1 = QSlider(Qt.Horizontal)
        self.slider1.setRange(1, 5)
        self.label1 = QLabel('a :')
        layout1_2.addWidget(self.label1)
        layout1_2.addWidget(self.slider1)
        self.textbox1 = QLineEdit()
        layout1.addWidget(self.textbox1)
        layout1.addLayout(layout1_2)

        layout2_2 = QHBoxLayout()
        layout2 = QVBoxLayout()
        self.slider2 = QSlider(Qt.Horizontal)
        self.slider2.setRange(1, 5)
        self.label2 = QLabel('b :')
        layout2_2.addWidget(self.label2)
        layout2_2.addWidget(self.slider2)
        self.textbox2 = QLineEdit()
        layout2.addWidget(self.textbox2)
        layout2.addLayout(layout2_2)

        layout3_2 = QHBoxLayout()
        layout3 = QVBoxLayout()
        self.slider3 = QSlider(Qt.Horizontal)
        self.slider3.setRange(1, 5)
        self.label3 = QLabel('c :')
        layout3_2.addWidget(self.label3)
        layout3_2.addWidget(self.slider3)
        self.textbox3 = QLineEdit()
        layout3.addWidget(self.textbox3)
        layout3.addLayout(layout3_2)

        layout4_2 = QHBoxLayout()
        layout4 = QVBoxLayout()
        self.slider4 = QSlider(Qt.Horizontal)
        self.slider4.setRange(1, 10)
        self.label4 = QLabel('d :')
        layout4_2.addWidget(self.label4)
        layout4_2.addWidget(self.slider4)
        self.textbox4 = QLineEdit()
        layout4.addWidget(self.textbox4)
        layout4.addLayout(layout4_2)

        layout5_2 = QHBoxLayout()
        layout5 = QVBoxLayout()
        self.slider5 = QSlider(Qt.Horizontal)
        self.slider5.setRange(0.0001, 0.001)
        self.label5 = QLabel('r :')
        layout5_2.addWidget(self.label5)
        layout5_2.addWidget(self.slider5)
        self.textbox5 = QLineEdit()
        layout5.addWidget(self.textbox5)
        layout5.addLayout(layout5_2)

        layout6_2 = QHBoxLayout()
        layout6 = QVBoxLayout()
        self.slider6 = QSlider(Qt.Horizontal)
        self.slider6.setRange(-4, 4)
        self.label6 = QLabel('s :')
        layout6_2.addWidget(self.label6)
        layout6_2.addWidget(self.slider6)
        self.textbox6 = QLineEdit()
        layout6.addWidget(self.textbox6)
        layout6.addLayout(layout6_2)

        layout7_2 = QHBoxLayout()
        layout7 = QVBoxLayout()
        self.slider7 = QSlider(Qt.Horizontal)
        self.label7 = QLabel('xr :')
        self.slider7.setRange(-3, 3)
        layout7_2.addWidget(self.label7)
        layout7_2.addWidget(self.slider7)
        self.textbox7 = QLineEdit()
        layout7.addWidget(self.textbox7)
        layout7.addLayout(layout7_2)

        layout8_2 = QHBoxLayout()
        layout8 = QVBoxLayout()
        self.slider8 = QSlider(Qt.Horizontal)
        self.slider8.setRange(0, 15)
        self.label8 = QLabel('I :')
        layout8_2.addWidget(self.label8)
        layout8_2.addWidget(self.slider8)
        self.textbox8 = QLineEdit()
        layout8.addWidget(self.textbox8)
        layout8.addLayout(layout8_2)

        layout_palm = QVBoxLayout()
        layout_palm.addLayout(layout1)
        layout_palm.addLayout(layout2)
        layout_palm.addLayout(layout3)
        layout_palm.addLayout(layout4)
        layout_palm.addLayout(layout5)
        layout_palm.addLayout(layout6)
        layout_palm.addLayout(layout7)
        layout_palm.addLayout(layout8)

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
        self.slider1.valueChanged.connect(self.slider1_changed)
        self.slider2.valueChanged.connect(self.slider2_changed)
        self.slider3.valueChanged.connect(self.slider3_changed)
        self.slider4.valueChanged.connect(self.slider4_changed)
        self.slider5.valueChanged.connect(self.slider5_changed)
        self.slider6.valueChanged.connect(self.slider6_changed)
        self.slider7.valueChanged.connect(self.slider7_changed)
        self.slider8.valueChanged.connect(self.slider8_changed)

    # slot
    def text1_changed(self):
        self.glaph.replot_a(float(self.textbox1.text()))

    def text2_changed(self):
        self.glaph.replot_b(float(self.textbox2.text()))

    def text3_changed(self):
        self.glaph.replot_c(float(self.textbox3.text()))

    def text4_changed(self):
        self.glaph.replot_d(float(self.textbox4.text()))

    def text5_changed(self):
        self.glaph.replot_r(float(self.textbox5.text()))

    def text6_changed(self):
        self.glaph.replot_s(float(self.textbox6.text()))

    def text7_changed(self):
        self.glaph.replot_xr(float(self.textbox7.text()))

    def text8_changed(self):
        self.glaph.replot_i(float(self.textbox8.text()))

    def slider1_changed(self):
        self.textbox1.setText(str(self.slider1.value()))

    def slider2_changed(self):
        self.textbox2.setText(str(self.slider2.value()))

    def slider3_changed(self):
        self.textbox3.setText(str(self.slider3.value()))

    def slider4_changed(self):
        self.textbox4.setText(str(self.slider4.value()))

    def slider5_changed(self):
        self.textbox5.setText(str(self.slider5.value()))

    def slider6_changed(self):
        self.textbox6.setText(str(self.slider6.value()))

    def slider7_changed(self):
        self.textbox7.setText(str(self.slider7.value()))

    def slider8_changed(self):
        self.textbox8.setText(str(self.slider8.value()))


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.left = 10
        self.top = 10
        self.title = 'Hindmarsh-Rose model - simulater'
        self.width = 1024
        self.height = 768
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.setCentralWidget(CentralWidget(self))


class PlotCanvas(FigureCanvas):

    def __init__(self, parent=None, width=10, height=8, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.plot_init()

    def plot_init(self):
        self.a = 1
        self.b = 3
        self.c = 1
        self.d = 5
        self.r = 0.003
        self.s = 4
        self.i = 2.3
        self.xr = -1.56
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title('Hindmarsh-Rose model Example')
        
        self.dt = 0.05
        self.t = np.arange(0, 2000, self.dt)
        self.dx = 0
        self.dy = 0
        self.dz = 0
        self.x = [-1.5] * np.size(self.t)
        self.y = [-10.2] * np.size(self.t)
        self.z = [-1.6] * np.size(self.t)
        
        self.plot(self.a, self.b, self.c, self.d, self.r, self.s, self.xr, self.i)

    def replot_a(self, a):
        self.line.remove()
        self.tmp_a = a
        self.plot(self.tmp_a, self.b, self.c, self.d, self.r, self.s, self.xr, self.i)

    def replot_b(self, b):
        self.line.remove()     
        self.tmp_b = b
        self.plot(self.a, self.tmp_b, self.c, self.d, self.r, self.s, self.xr, self.i)

    def replot_c(self, c):
        self.line.remove()     
        self.tmp_c = c 
        self.plot(self.a, self.b, self.tmp_c, self.d, self.r, self.s, self.xr, self.i)

    def replot_d(self, d):
        self.line.remove()     
        self.tmp_d = d
        self.plot(self.a, self.b, self.c, self.tmp_d, self.r, self.s, self.xr, self.i)

    def replot_r(self, r):
        self.line.remove()     
        self.tmp_r = r
        self.plot(self.a, self.b, self.c, self.d, self.tmp_r, self.s, self.xr, self.i)

    def replot_s(self, s):
        self.line.remove()     
        self.tmp_s = s
        self.plot(self.a, self.b, self.c, self.d, self.r, self.tmp_s, self.xr, self.i)

    def replot_xr(self, xr):
        self.line.remove()     
        self.tmp_xr = xr
        self.plot(self.a, self.b, self.c, self.d, self.r, self.s, self.tmp_xr, self.i)

    def replot_i(self, i):
        self.line.remove()     
        self.tmp_i = i 
        self.plot(self.a, self.b, self.c, self.d, self.r, self.s, self.xr, self.tmp_i)

    def plot(self, a, b, c, d, r, s, xr, i):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.r = r
        self.s = s
        self.xr = xr
        self.i = i

        for i in range(0, np.size(self.t)-1):
            self.dx = self.y[i] - self.a * self.x[i]**3 + self.b * self.x[i]**2 -self.z[i] + self.i
            self.dy = self.c - self.d * self.x[i]**2 - self.y[i]
            self.dz = self.r*(self.s*(self.x[i] - self.xr) - self.z[i])

            self.x[i+1] = self.x[i] + self.dt * self.dx
            self.y[i+1] = self.y[i] + self.dt * self.dy
            self.z[i+1] = self.z[i] + self.dt * self.dz

        self.line, = self.ax.plot(self.t, self.x)
        self.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mw = MainWindow()
    mw.show()
    sys.exit(app.exec_())
