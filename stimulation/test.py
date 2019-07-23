# -*- coding: utf-8 -*-
import sys
import keyboard
from PyQt5.QtWidgets import QWidget, QApplication, QSystemTrayIcon
from PyQt5.QtGui import QIcon
from PyQt5.Qt import Qt


class Window(QWidget):
    tray = QSystemTrayIcon()

    def __init__(self):
        super().__init__()
        iconPath = 'sample_icon.png'

        self.setWindowFlags(Qt.WindowStaysOnTopHint)  # 常に最前面に表示
        self.setWindowTitle('PyQt5 Window Show or Hide')  # ウィンドウタイトル
        self.setWindowIcon(QIcon(iconPath))  # ウィンドウアイコン

        self.tray.setIcon(QIcon(iconPath))  # トレイアイコン
        self.tray.activated.connect(self.ShowOrHide)  # トレイクリック時
        self.tray.show()

        # グローバルホットキー：Ctrl+Shift+Altを押した場合
        keyboard.add_hotkey('shift+F', lambda: self.ShowOrHide())

    def ShowOrHide(self):
        # ウィンドウの表示or非表示
        """
        if self.isHidden():
            self.show()
        else:
            self.hide()
        """
        print("kanopero")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    execute = Window()
    execute.show()
    sys.exit(app.exec_())