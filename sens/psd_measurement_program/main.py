import sys
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from picpsd_gui2 import Ui_MainWindow

class Test(QMainWindow):
    def __init__(self, parent=None, port= "COM0"):
        super(Test, self).__init__(parent)
        self.port = port
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self, self.port)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Test(port="COM10")
    window.show()
    sys.exit(app.exec_())
