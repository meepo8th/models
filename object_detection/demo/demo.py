import os.path
import traceback
from qtpy import QtGui

from object_detection.demo.lib.lib import struct, newAction, newIcon, addActions, fmtShortcut, downLoadPic
import sys
from object_detection.demo.lib.canvas import Canvas
from object_detection.demo.lib.shape import Shape

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
                                              "picture files(*.png;*.jpg;*.bmp)")
        self.openWithPath(picPath[0])

    def scrollRequest(self, delta, orientation):
        units = - delta / (8 * 15)
        bar = self.scrollBars[orientation]
        bar.setValue(bar.value() + bar.singleStep() * units)

    def openWithPath(self, picPath):
        self.showTips(u"图片识别中...")
        infoData = (0, 0)
        try:
            self.showPic(picPath)
            data = libTensor.detect(picPath)
            if data:
                self.loadData(data, picPath)
                infoData = libTensor.countPersonWithElement(data)
        except Exception:
            self.showTips(u"图片识别失败")
            print(traceback.format_exc())
        msg = u"识别到%s个人，其中%s个人没有佩戴安全帽" % (infoData[0], infoData[1])
        self.showTips(msg)
        print(msg)

    def loadData(self, data, picPath):
        image = QImage()
        image.load(picPath)
        width = image.width()
        imageScaled = image.scaled(self.canvas.width(), self.canvas.height(), Qt.KeepAspectRatio)
        scaleWidth = imageScaled.width()
        scale = (scaleWidth + 0.0) / width
        s = []
        if data:
            for key, values in data.items():
                for value in values:
                    shape = Shape(label=key)
                    shape.addPoint(QPointF(scale * value[0], scale * value[1]))
                    shape.addPoint(QPointF(scale * value[0], scale * value[3]))
                    shape.addPoint(QPointF(scale * value[2], scale * value[3]))
                    shape.addPoint(QPointF(scale * value[2], scale * value[1]))
                    shape.close()
                    s.append(shape)
            self.canvas.loadShapes(s)

    def showPic(self, picPath):
        image = QImage()
        if image.load(picPath):
            image = image.scaled(self.canvas.width(), self.canvas.height(), Qt.KeepAspectRatio)
            self.canvas.loadPixmap(QPixmap.fromImage(image))

    def showTips(self, tips):
        self.statusBar().showMessage(tips)
        pass

    def goWithUrl(self, url):
        picPath = downLoadPic(url)
        if '' == picPath:
            self.showTips(u"获取图片失败")
        else:
            self.openWithPath(picPath)

    def go(self):
        self.showTips(u"获取图片中...")
        self.goWithUrl(self.url.text())

    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle(__appname__)
        self.toolbar = self.addToolBar("toolBar")
        self.tipsBar = self.statusTip()
        libTensor.loadGraph()
        self.setGeometry(100, 100, 800, 600)
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
        self.canvas = Canvas()
        scroll = QScrollArea()
        scroll.setWidget(self.canvas)
        scroll.setWidgetResizable(True)
        self.scrollBars = {
            Qt.Vertical: scroll.verticalScrollBar(),
            Qt.Horizontal: scroll.horizontalScrollBar()
        }
        self.canvas.scrollRequest.connect(self.scrollRequest)
        self.setCentralWidget(scroll)
        self.showTips("请输入图片网络地址或者选择本地图片")


def main(argv=[]):
    app, _win = get_main_app(argv)
    return app.exec_()


if __name__ == '__main__':
    sys.exit(main(sys.argv))
