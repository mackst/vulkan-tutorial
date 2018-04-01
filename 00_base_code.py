# -*- coding: UTF-8 -*-

from PySide2 import (QtGui, QtCore)


class HelloTriangleApplication(QtGui.QWindow):

    def __init__(self):
        super(HelloTriangleApplication, self).__init__()

        self.setWidth(1280)
        self.setHeight(720)

        self.setTitle("Vulkan Python - PySide2")

        #self.setSurfaceType(self.OpenGLSurface)

    def __del__(self):
        print('closed')
        pass


if __name__ == '__main__':
    import sys

    app = QtGui.QGuiApplication(sys.argv)

    win = HelloTriangleApplication()
    win.show()

    # def clenaup():
    #     global win
    #     del win
    #
    # app.aboutToQuit.connect(clenaup)

    sys.exit(app.exec_())
