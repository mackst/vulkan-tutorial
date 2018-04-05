# -*- coding: UTF-8 -*-

from vulkan import *
from PySide2 import (QtGui, QtCore)


validationLayers = [
    'VK_LAYER_LUNARG_standard_validation'
]

enableValidationLayers = True


class InstanceProcAddr(object):

    def __init__(self, func):
        self.__func = func

    def __call__(self, *args, **kwargs):
        funcName = self.__func.__name__
        func = vkGetInstanceProcAddr(args[0], funcName)
        if func:
            return func(*args, **kwargs)
        else:
            return VK_ERROR_EXTENSION_NOT_PRESENT

@InstanceProcAddr
def vkCreateDebugReportCallbackEXT(instance, pCreateInfo, pAllocator):
    pass

@InstanceProcAddr
def vkDestroyDebugReportCallbackEXT(instance, pCreateInfo, pAllocator):
    pass


def debugCallback(*args):
    print('DEBUG: {} {}'.format(args[5], args[6]))
    return 0


class QueueFamilyIndices(object):

    def __init__(self):
        self.graphicsFamily = -1

    @property
    def isComplete(self):
        return self.graphicsFamily >= 0


class HelloTriangleApplication(QtGui.QWindow):

    def __init__(self):
        super(HelloTriangleApplication, self).__init__()

        self.setWidth(1280)
        self.setHeight(720)

        self.setTitle("Vulkan Python - PySide2")

        #self.setSurfaceType(self.OpenGLSurface)

        self.__instance = None
        self.__callbcak = None
        self.__physicalDevice = None
        self.__device = None
        self.__graphicQueue = None

        self.initVulkan()

    def __del__(self):
        if self.__device:
            vkDestroyDevice(self.__device, None)

        if self.__callbcak:
            vkDestroyDebugReportCallbackEXT(self.__instance, self.__callbcak, None)

        if self.__instance:
            vkDestroyInstance(self.__instance, None)
            print('instance destroyed')

    def initVulkan(self):
        self.__cretaeInstance()
        self.__setupDebugCallback()
        self.__pickPhysicalDevice()
        self.__createLogicalDevice()

    def __cretaeInstance(self):
        if enableValidationLayers and not self.__checkValidationLayerSupport():
            raise Exception("validation layers requested, but not available!")

        appInfo = VkApplicationInfo(
            # sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName='Python VK',
            applicationVersion=VK_MAKE_VERSION(1, 0, 0),
            pEngineName='pyvulkan',
            engineVersion=VK_MAKE_VERSION(1, 0, 0),
            apiVersion=VK_API_VERSION
        )

        extenstions = self.__getRequiredExtensions()
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

        self.__callbcak = vkCreateDebugReportCallbackEXT(self.__instance, createInfo, None)

    def __pickPhysicalDevice(self):
        physicalDevices = vkEnumeratePhysicalDevices(self.__instance)

        for device in physicalDevices:
            if self.__isDeviceSuitable(device):
                self.__physicalDevice = device
                break

        assert self.__physicalDevice != None

    def __createLogicalDevice(self):
        indices = self.__findQueueFamilies(self.__physicalDevice)
        queueCreateInfo = VkDeviceQueueCreateInfo(
            queueFamilyIndex=indices.graphicsFamily,
            queueCount=1,
            pQueuePriorities=[1.0]
        )

        deviceFeatures = VkPhysicalDeviceFeatures()
        if enableValidationLayers:
            createInfo = VkDeviceCreateInfo(
                queueCreateInfoCount=1,
                pQueueCreateInfos=queueCreateInfo,
                enabledExtensionCount=0,
                enabledLayerCount=len(validationLayers),
                ppEnabledLayerNames=validationLayers,
                pEnabledFeatures=deviceFeatures
            )
        else:
            createInfo = VkDeviceCreateInfo(
                queueCreateInfoCount=1,
                pQueueCreateInfos=queueCreateInfo,
                enabledExtensionCount=0,
                enabledLayerCount=0,
                pEnabledFeatures=deviceFeatures
            )

        self.__device = vkCreateDevice(self.__physicalDevice, createInfo, None)

    def __isDeviceSuitable(self, device):
        indices = self.__findQueueFamilies(device)

        return indices.isComplete

    def __findQueueFamilies(self, device):
        indices = QueueFamilyIndices()

        familyProperties = vkGetPhysicalDeviceQueueFamilyProperties(device)
        for i, prop in enumerate(familyProperties):
            if prop.queueCount > 0 and prop.queueFlags & VK_QUEUE_GRAPHICS_BIT:
                indices.graphicsFamily = i

            if indices.isComplete:
                break

        return indices

    def __getRequiredExtensions(self):
        extenstions = [e.extensionName for e in vkEnumerateInstanceExtensionProperties(None)]

        if enableValidationLayers:
            extenstions.append(VK_EXT_DEBUG_REPORT_EXTENSION_NAME)

        return extenstions

    def __checkValidationLayerSupport(self):
        availableLayers = vkEnumerateInstanceLayerProperties()

        for layer in validationLayers:
            layerfound = False

            for layerProp in availableLayers:
                if layer == layerProp.layerName:
                    layerfound = True
                    break
            return layerfound

        return False

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
