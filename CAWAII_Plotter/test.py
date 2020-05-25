import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *


class MainWindow(QWidget):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        # ボタンの作成
        makeWindowButton = QPushButton("&make window")
        # クリックイベント(makeWindow())の追加
        makeWindowButton.clicked.connect(self.makeWindow)
        # レイアウトの作成
        layout = QHBoxLayout()
        # レイアウトにボタンの追加
        layout.addWidget(makeWindowButton)
        # ウィンドウにレイアウトを追加
        self.setLayout(layout)

    def makeWindow(self):
        # サブウィンドウの作成
        subWindow = SubWindow()
        # サブウィンドウの表示
        subWindow.show()

class SubWindow(QWidget):
    def __init__(self, parent=None):
        # こいつがサブウィンドウの実体？的な。ダイアログ
        self.w = QDialog(parent)
        label = QLabel()
        label.setText('Sub Window')
        layout = QHBoxLayout()
        layout.addWidget(label)
        self.w.setLayout(layout)

    def show(self):
        self.w.exec_()

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    main_window = MainWindow()

    main_window.show()
    sys.exit(app.exec_())