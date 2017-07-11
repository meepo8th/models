import os.path

from qtpy import QtGui

from object_detection.demo.lib.lib import struct, newAction, newIcon, addActions, fmtShortcut, downLoadPic
import sys

try:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *
except ImportError:
    # needed for py3+qt4
    # Ref:
    # http://pyqt.sourceforge.net/Docs/PyQt4/incompatible_apis.html
    # http://stackoverflow.com/questions/21217399/pyqt4-qtcore-qvariant-object-instead-of-a-string
    if sys.version_info.major >= 3:
        import sip

        sip.setapi('QVariant', 2)
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *

import object_detection.demo.lib.lib_tensor as libTensor
import object_detection.demo.resources as resources

__appname__ = "智能识别演示程序"


def get_main_app(argv=[]):
    app = QApplication(argv)
    app.setApplicationName(__appname__)
    app.setWindowIcon(newIcon(__appname__))
    win = MainWindow()
    win.showMaximized()
    return app, win


class MainWindow(QMainWindow):
    def open(self):
        picPath = QFileDialog.getOpenFileName(self, "open file dialog", "C:",
                                              "picture files(*.png,*.jpg,*.bmp)")
        self.openWithPath(picPath)

    def openWithPath(self, picPath):
        data = libTensor.detect(picPath)
        if not data.keys().__len__ == 0:
            print(data)
        else:
            print("no object")

    def goWithUrl(self, url):
        picPath = downLoadPic(url)
        if '' == picPath:
            msg_box = QMessageBox(QMessageBox.Warning, u"提示", u"输入路径不正确")
            msg_box.exec_()
        else:
            self.openWithPath(picPath)

    def go(self):
        self.goWithUrl(self.url.text())

    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle(__appname__)
        self.toolbar = self.addToolBar("toolBar")
        libTensor.loadGraph()
        self.setGeometry(100, 100, 800, 600)
        self.statusBar()
        self.url = QLineEdit()
        self.toolbar.addWidget(QLabel("请输入图片地址或者选择本地图片:"))
        self.toolbar.addWidget(self.url)
        self.goButton = QToolButton()
        self.goButton.setArrowType(Qt.RightArrow)
        self.goButton.clicked.connect(self.go)
        self.toolbar.addWidget(self.goButton)
        self.openButton = QToolButton()
        self.openButton.setToolButtonStyle(Qt.ToolButtonIconOnly)
        self.openButton.setIcon(newIcon('file'))
        self.openButton.clicked.connect(self.open)
        self.toolbar.addWidget(self.openButton)


def main(argv=[]):
    app, _win = get_main_app(argv)
    return app.exec_()


if __name__ == '__main__':
    sys.exit(main(sys.argv))
