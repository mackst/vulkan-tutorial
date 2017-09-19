from vulkan import *

from PyQt5 import QtGui


WIDTH = 800
HEIGHT = 600


class HelloTriangleApplication(QtGui.QWindow):

    def __init__(self):
        super(HelloTriangleApplication, self).__init__(None)

    def __initWindow(self):
        self.setSurfaceType(self.OpenGLSurface)
        self.setTitle("Vulkan")
        self.resize(WIDTH, HEIGHT)

    def __initVulkan(self):
        pass

    def __mainLoop(self):
        pass

    def show(self):
        self.__initWindow()
        self.__initVulkan()
        self.__mainLoop()

        super(HelloTriangleApplication, self).show()


if __name__ == '__main__':
    import sys

    app = QtGui.QGuiApplication(sys.argv)

    win = HelloTriangleApplication()
    win.show()

    sys.exit(app.exec_())

