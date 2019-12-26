import sys
import os
from PyQt5.QtWidgets import *
import pyqtgraph as pg
import pandas as pd
from pyqtgraph.Qt import QtGui, QtCore
import pandas as pd



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
        print(mimeData.text())
        print(os.path.basename(mimeData.text()))
        filename, ext = os.path.splitext(os.path.basename(mimeData.text()))
        df = pd.read_csv(mimeData.text())
        print(df)
        print(filename)
        print(ext)


        #self.subWindow = SubWindow(df)

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