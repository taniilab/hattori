import sys
import os
from PyQt5.QtWidgets import *
import pyqtgraph as pg
import pandas as pd
from pyqtgraph.Qt import QtGui, QtCore




class DropWidget(QWidget):
    def __init__(self, parent=None):
        super(DropWidget, self).__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        event.accept()
        mimeData = event.mimeData()
        print('dragEnterEvent')
        for mimetype in mimeData.formats():
            print('MIMEType:', mimetype)
            print('Data:', mimeData.data(mimetype))
            print()
        print()

    def dropEvent(self, event):
        event.accept()
        mimeData = event.mimeData()
        print('dropEvent')
        """
        for mimetype in mimeData.formats():
            print('MIMEType:', mimetype)
            print('Data:', mimeData.data(mimetype))
            print()
        """
        print(mimeData.urls())
        print(os.path.basename(mimeData.text()))
        filename, ext = os.path.splitext(os.path.basename(mimeData.text()))
        print(filename)
        print(ext)

        path = "//192.168.13.10/Public/ishida/simulation/dynamic_synapse_exp17__Mg=0.5~1.0/tmp/2018_10_3_5_26_41/" + \
               filename + ".csv"

        """
        path = "//192.168.13.10/Public/hattori/simulation/HH/rawdata/2018_9_22_9_50_8/" + \
               filename + ".csv"
        """
        print(path)
        df = pd.read_csv(path, delimiter=',', skiprows=1)
        df.fillna(0)

        pg.setConfigOption('background', (255, 255, 255))
        pg.setConfigOption('foreground', (0, 0, 0))

        self.subWindow = SubWindow(df)

        print()


class SubWindow(QMainWindow):
    def __init__(self, df, parent=None):
        QMainWindow.__init__(self, parent)
        self.df = df
        self.pw = pg.PlotWidget(name='Plot1')
        self.pw.plot(self.df['T [ms]'], self.df['V [mV]'], pen=(0, 0, 0))
        self.pw.showGrid(True, True, 0.2)
        self.pw2 = pg.PlotWidget(name='Plot2')
        self.pw2.plot(self.df['T [ms]'], self.df['I_AMPA [uA]'], pen=(200, 0, 0))
        self.pw2.plot(self.df['T [ms]'], self.df['I_NMDA [uA]'], pen=(0, 100, 100))
        self.pw2.showGrid(True, True, 0.2)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.pw)
        self.layout.addWidget(self.pw2)


        self.cw = QWidget()
        self.cw.setLayout(self.layout)

        self.setCentralWidget(self.cw)
        self.show()

def main():
    app = QApplication(sys.argv)
    w = DropWidget()
    w.resize(250, 200)
    w.show()
    w.raise_()
    app.exec_()

if __name__ == '__main__':
    main()