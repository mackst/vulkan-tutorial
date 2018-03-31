from vulkan import *

from PyQt5 import QtGui


WIDTH = 800
HEIGHT = 600


class HelloTriangleApplication(QtGui.QWindow):

    def __init__(self):
        super(HelloTriangleApplication, self).__init__(None)

        self.__instance = None

    def __del__(self):

        if self.__instance:
            vkDestroyInstance(self.__instance, None)

    def __initWindow(self):
        self.setSurfaceType(self.OpenGLSurface)
        self.setTitle("Vulkan")
        self.resize(WIDTH, HEIGHT)

    def __initVulkan(self):
        self.__createInstance()

    def __mainLoop(self):
        pass

    def __createInstance(self):
        appInfo = VkApplicationInfo(
            sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName='Hello Triangle',
            applicationVersion=VK_MAKE_VERSION(1, 0, 0),
            pEngineName='No Engine',
            engineVersion=VK_MAKE_VERSION(1, 0, 0),
            apiVersion=VK_API_VERSION
        )

        extensions = [i.extensionName for i in vkEnumerateInstanceExtensionProperties(None)]
        print(extensions)
        createInfo = VkInstanceCreateInfo(
            sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            pApplicationInfo=appInfo,
            enabledLayerCount=0,
            enabledExtensionCount=len(extensions),
            ppEnabledExtensionNames=extensions
        )

        self.__instance = vkCreateInstance(createInfo, None)

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

    def clenaup():
        global win
        del win

    app.aboutToQuit.connect(clenaup)

    sys.exit(app.exec_())

