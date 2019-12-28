import sys
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from ddplot import Ui_MainWindow
from PyQt5.QtGui import QIcon


class Main(QMainWindow):
    def __init__(self, parent=None):
        super(Main, self).__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Main()
    window.show()
    sys.exit(app.exec_())