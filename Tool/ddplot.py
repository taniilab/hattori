import sys
from PyQt5.QtWidgets import *
import pandas as pd

class DropWidget(QWidget):
    def __init__(self, parent=None):
        super(DropWidget, self).__init__(parent)
        self.setAcceptDrops(True)


    def dragEnterEvent(self, event):
        event.accept()
        mimeData = event.mimeData()
        """
        print('dragEnterEvent')
        for mimetype in mimeData.formats():
            print('MIMEType:', mimetype)
            print('Data:', mimeData.data(mimetype))
            print()
        print()
        """

    def dropEvent(self, event):
        event.accept()
        mimeData = event.mimeData()
        print(mimeData.data('application/x-qt-windows-mime;value="FileName"'))
        filename = str(mimeData.data('application/x-qt-windows-mime;value="FileName"'))
        filename = filename.replace("b'", "")
        filename = filename.replace("\\x00'", "")

        df = pd.read_csv(filename, delimiter=',')
        print(df)
        print('dropEvent')
        """
        for mimetype in mimeData.formats():
            print('MIMEType:', mimetype)
            print('Data:', mimeData.data(mimetype))
            print()
        """
        print()

def main():
    app = QApplication(sys.argv)
    w = DropWidget()
    w.resize(800, 600)
    w.show()
    w.raise_()
    app.exec_()

if __name__ == '__main__':
    main()