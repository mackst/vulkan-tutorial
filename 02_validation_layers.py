# -*- coding: UTF-8 -*-

from vulkan import *
from PySide2 import (QtGui, QtCore)


validationLayers = [
    'VK_LAYER_LUNARG_standard_validation'
]

enableValidationLayers = True



def createDebugReportCallbackEXT(instance, pCreateInfo, pAllocator=None):
    func = vkGetInstanceProcAddr(instance, "vkCreateDebugReportCallbackEXT")
    if func:
        return func(instance, pCreateInfo, pAllocator)
    else:
        return VK_ERROR_EXTENSION_NOT_PRESENT


def destroyDebugReportCallbackEXT(instance, callback, pAllocator=None):
    func = vkGetInstanceProcAddr(instance, "vkDestroyDebugReportCallbackEXT")
    if func:
        return func(instance, callback, pAllocator)
    else:
        return VK_ERROR_EXTENSION_NOT_PRESENT


def debugCallback(*args):
    print('DEBUG: {} {}'.format(args[5], args[6]))
    return 0


class HelloTriangleApplication(QtGui.QWindow):

    def __init__(self):
        super(HelloTriangleApplication, self).__init__()

        self.setWidth(1280)
        self.setHeight(720)

        self.setTitle("Vulkan Python - PySide2")

        #self.setSurfaceType(self.OpenGLSurface)

        self.__instance = None
        self.__callbcak = None

        self.initVulkan()

    def __del__(self):
        if self.__callbcak:
            destroyDebugReportCallbackEXT(self.__instance, self.__callbcak)

        if self.__instance:
            vkDestroyInstance(self.__instance, None)
            print('instance destroyed')

    def initVulkan(self):
        self.__cretaeInstance()
        self.__setupDebugCallback()

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
        if enableValidationLayers:
            instanceInfo = VkInstanceCreateInfo(
                pApplicationInfo=appInfo,
                enabledLayerCount=len(validationLayers),
                ppEnabledLayerNames=validationLayers,
                enabledExtensionCount=len(extenstions),
                ppEnabledExtensionNames=extenstions
            )
        else:
            instanceInfo = VkInstanceCreateInfo(
                pApplicationInfo=appInfo,
                enabledLayerCount=0,
                enabledExtensionCount=len(extenstions),
                ppEnabledExtensionNames=extenstions
            )

        self.__instance = vkCreateInstance(instanceInfo, None)

    def __setupDebugCallback(self):
        if not enableValidationLayers:
            return

        createInfo = VkDebugReportCallbackCreateInfoEXT(
            flags=VK_DEBUG_REPORT_WARNING_BIT_EXT | VK_DEBUG_REPORT_ERROR_BIT_EXT,
            pfnCallback=debugCallback
        )

        self.__callbcak = createDebugReportCallbackEXT(self.__instance, createInfo)


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
