from vulkan import *

from PyQt5 import QtGui


WIDTH = 800
HEIGHT = 600

validationLayers = ["VK_LAYER_LUNARG_standard_validation"]

enableValidationLayers = True


def debugCallback(*args):
    print('DEBUG: {} {}'.format(args[5], args[6]))
    return 0

def createDebugReportCallbackEXT(instance, pCreateInfo, pAllocator):
    func = vkGetInstanceProcAddr(instance, 'vkCreateDebugReportCallbackEXT')
    if func:
        return func(instance, pCreateInfo, pAllocator)
    else:
        return VK_ERROR_EXTENSION_NOT_PRESENT

def destroyDebugReportCallbackEXT(instance, callback, pAllocator):
    func = vkGetInstanceProcAddr(instance, 'vkDestroyDebugReportCallbackEXT')
    if func:
        func(instance, callback, pAllocator)


class HelloTriangleApplication(QtGui.QWindow):

    def __init__(self):
        super(HelloTriangleApplication, self).__init__(None)

        self.__instance = None
        self.__callback = None

    def __del__(self):
        if self.__callback:
            destroyDebugReportCallbackEXT(self.__instance, self.__callback, None)

        if self.__instance:
            vkDestroyInstance(self.__instance, None)

    def __initWindow(self):
        self.setSurfaceType(self.OpenGLSurface)
        self.setTitle("Vulkan")
        self.resize(WIDTH, HEIGHT)

    def __initVulkan(self):
        self.__createInstance()
        self.__setupDebugCallback()

    def __mainLoop(self):
        pass

    def __createInstance(self):
        if enableValidationLayers and not self.__checkValidationLayerSupport():
            raise Exception("validation layers requested, but not available!")

        appInfo = VkApplicationInfo(
            sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName='Hello Triangle',
            applicationVersion=VK_MAKE_VERSION(1, 0, 0),
            pEngineName='No Engine',
            engineVersion=VK_MAKE_VERSION(1, 0, 0),
            apiVersion=VK_API_VERSION
        )

        extensions = self.__getRequiredExtensions()

        if enableValidationLayers:
            createInfo = VkInstanceCreateInfo(
                sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                pApplicationInfo=appInfo,
                enabledLayerCount=len(validationLayers),
                ppEnabledLayerNames=validationLayers,
                enabledExtensionCount=len(extensions),
                ppEnabledExtensionNames=extensions
            )
        else:
            createInfo = VkInstanceCreateInfo(
                sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                pApplicationInfo=appInfo,
                enabledLayerCount=0,
                enabledExtensionCount=len(extensions),
                ppEnabledExtensionNames=extensions
            )

        self.__instance = vkCreateInstance(createInfo, None)

    def __setupDebugCallback(self):
        if not enableValidationLayers:
            return

        createInfo = VkDebugReportCallbackCreateInfoEXT(
            sType=VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT,
            flags=VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT,
            pfnCallback=debugCallback
        )

        self.__callback = createDebugReportCallbackEXT(self.__instance, createInfo, None)
        if not self.__callback:
            raise Exception("failed to set up debug callback!")


    def __getRequiredExtensions(self):
        extensions = [i.extensionName for i in vkEnumerateInstanceExtensionProperties(None)]

        if enableValidationLayers:
            extensions.append(VK_EXT_DEBUG_REPORT_EXTENSION_NAME)

        return extensions

    def __checkValidationLayerSupport(self):
        availableLayers = vkEnumerateInstanceLayerProperties()
        for layerName in validationLayers:
            layerFound = False

            for layerProperties in availableLayers:
                if layerName == layerProperties.layerName:
                    layerFound = True
                    break
            if not layerFound:
                return False

        return True

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

