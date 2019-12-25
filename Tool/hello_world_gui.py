import sys

from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMainWindow

from hello_world_ui import Ui_MainWindow


class HelloWorldGui(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(HelloWorldGui, self).__init__(parent)
        self.setupUi(self)

    @pyqtSlot()
    def setTextHelloWorld(self):
        self.label.setText("Hello World")


if __name__ == '__main__':
    argvs = sys.argv
    app = QApplication(argvs)
    hello_world_gui = HelloWorldGui()
    hello_world_gui.show()
    sys.exit(app.exec_())
