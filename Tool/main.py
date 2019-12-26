import sys
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QIcon


class Example(QWidget):
    # コンストラクタ
    def __init__(self):
        super().__init__()

        self.initUI()


    def initUI(self):

        # setGeometry(x,y,横幅,高さ)
        self.setGeometry(300, 300, 300, 220)
        # タイトルの設定
        self.setWindowTitle('Icon')
        # 画面左上のアイコンを設定
        self.setWindowIcon(QIcon('maria.png'))

        # 画面表示
        self.show()


if __name__ == '__main__':

    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())