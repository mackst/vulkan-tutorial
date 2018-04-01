# -*- coding: UTF-8 -*-

from vulkan import *
from PySide2 import (QtGui, QtCore)


class HelloTriangleApplication(QtGui.QWindow):

    def __init__(self):
        super(HelloTriangleApplication, self).__init__()

        self.setWidth(1280)
        self.setHeight(720)

        self.setTitle("Vulkan Python - PySide2")

        #self.setSurfaceType(self.OpenGLSurface)

        self.__instance = None

        self.initVulkan()

    def __del__(self):
        if self.__instance:
            vkDestroyInstance(self.__instance, None)
            print('instance destroyed')

    def initVulkan(self):
        self.__cretaeInstance()

    def __cretaeInstance(self):
        appInfo = VkApplicationInfo(
            # sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName='Python VK',
            applicationVersion=VK_MAKE_VERSION(1, 0, 0),
            pEngineName='pyvulkan',
            engineVersion=VK_MAKE_VERSION(1, 0, 0),
            apiVersion=VK_API_VERSION
        )

        extenstions = [e.extensionName for e in vkEnumerateInstanceExtensionProperties(None)]
        instanceInfo = VkInstanceCreateInfo(
            pApplicationInfo=appInfo,
            enabledLayerCount=0,
            enabledExtensionCount=len(extenstions),
            ppEnabledExtensionNames=extenstions
        )

        self.__instance = vkCreateInstance(instanceInfo, None)


if __name__ == '__main__':
    import sys

    app = QtGui.QGuiApplication(sys.argv)

    win = HelloTriangleApplication()
    win.show()

    def clenaup():
        global win
        del win

    app.aboutToQuit.connect(clenaup)

    sys.exit(app.exec_())
