from pyVulkan import *

import PyGlfwCffi as glfw


WIDTH = 800
HEIGHT = 600

validationLayers = ["VK_LAYER_LUNARG_standard_validation"]

enableValidationLayers = True

@vkDebugReportCallbackEXT
def debugCallback(*args):
    print (ffi.string(args[6]))
    return True

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


class QueueFamilyIndices(object):

    def __init__(self):
        self.graphicsFamily = -1

    def isComplete(self):
        return self.graphicsFamily >= 0


class HelloTriangleApplication(object):

    def __init__(self):
        self.__window = None
        self.__instance = None
        self.__callback = None
        self.__surface = None
        self.__physicalDevice = None
        self.__device = None
        self.__graphicsQueue = None

    def __del__(self):
        if self.__device:
            vkDestroyDevice(self.__device, None)

        if self.__callback:
            destroyDebugReportCallbackEXT(self.__instance, self.__callback, None)

        if self.__instance:
            vkDestroyInstance(self.__instance, None)

    def __initWindow(self):
        glfw.init()

        glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
        glfw.window_hint(glfw.RESIZABLE, False)

        self.__window = glfw.create_window(WIDTH, HEIGHT, "Vulkan")

    def __initVulkan(self):
        self.__createInstance()
        self.__setupDebugCallback()
        self.__pickPhysicalDevice()
        self.__createLogicalDevice()

    def __mainLoop(self):
        while not glfw.window_should_close(self.__window):
            glfw.poll_events()

    def __createInstance(self):
        if enableValidationLayers and not self.__checkValidationLayerSupport():
            raise Exception("validation layers requested, but not available!")

        appInfo = VkApplicationInfo(
            pApplicationName='Hello Triangle',
            applicationVersion=VK_MAKE_VERSION(1, 0, 0),
            pEngineName='No Engine',
            engineVersion=VK_MAKE_VERSION(1, 0, 0),
            apiVersion=VK_API_VERSION
        )

        createInfo = VkInstanceCreateInfo(pApplicationInfo=appInfo)
        extensions = self.__getRequiredExtensions()
        ext = [ffi.new('char[]', i) for i in extensions]
        extArray = ffi.new('char*[]', ext)

        createInfo.enabledExtensionCount = len(extensions)
        createInfo.ppEnabledExtensionNames = extArray

        if enableValidationLayers:
            createInfo.enabledLayerCount = len(validationLayers)
            layers = [ffi.new('char[]', i) for i in validationLayers]
            vlayers = ffi.new('char*[]', layers)
            createInfo.ppEnabledLayerNames = vlayers
        else:
            createInfo.enabledLayerCount = 0

        self.__instance = vkCreateInstance(createInfo, None)

    def __setupDebugCallback(self):
        if not enableValidationLayers:
            return

        createInfo = VkDebugReportCallbackCreateInfoEXT(
            flags=VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT,
            pfnCallback=debugCallback
        )
        self.__callback = createDebugReportCallbackEXT(self.__instance, createInfo, None)
        if not self.__callback:
            raise Exception("failed to set up debug callback!")

    def __pickPhysicalDevice(self):
        devices = vkEnumeratePhysicalDevices(self.__instance)

        for device in devices:
            if self.__isDeviceSuitable(device):
                self.__physicalDevice = device
                break

        if self.__physicalDevice is None:
            raise Exception("failed to find a suitable GPU!")

    def __createLogicalDevice(self):
        indices = self.__findQueueFamilies(self.__physicalDevice)

        queueCreateInfo = VkDeviceQueueCreateInfo(
            queueFamilyIndex=indices.graphicsFamily,
            queueCount=1,
            pQueuePriorities=[1.0]
        )
        deviceFeatures = VkPhysicalDeviceFeatures()
        # deviceFeatures = VkPhysicalDeviceFeatures(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
        createInfo = VkDeviceCreateInfo(
            flags=0,
            pQueueCreateInfos=queueCreateInfo,
            queueCreateInfoCount=1,
            pEnabledFeatures=[deviceFeatures],
            enabledExtensionCount=0
        )

        if enableValidationLayers:
            createInfo.enabledLayerCount = len(validationLayers)
            layers = [ffi.new('char[]', i) for i in validationLayers]
            vlayers = ffi.new('char*[]', layers)
            createInfo.ppEnabledLayerNames = vlayers
        else:
            createInfo.enabledLayerCount = 0

        self.__device = vkCreateDevice(self.__physicalDevice, createInfo, None)
        if self.__device is None:
            raise Exception("failed to create logical device!")
        self.__graphicsQueue = vkGetDeviceQueue(self.__device, indices.graphicsFamily, 0)

    def __isDeviceSuitable(self, device):
        indices = self.__findQueueFamilies(device)
        return indices.isComplete()

    def __findQueueFamilies(self, device):
        indices = QueueFamilyIndices()

        queueFamilies = vkGetPhysicalDeviceQueueFamilyProperties(device)

        for i, queueFamily in enumerate(queueFamilies):
            if queueFamily.queueCount > 0 and queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT:
                indices.graphicsFamily = i
            if indices.isComplete():
                break

        return indices

    def __getRequiredExtensions(self):
        extensions = []

        glfwExtensions, glfwExtensionCount = glfw.getRequiredInstanceExtensions()
        for i in range(glfwExtensionCount[0]):
            extensions.append(ffi.string(glfwExtensions[i]))

        if enableValidationLayers:
            extensions.append(VK_EXT_DEBUG_REPORT_EXTENSION_NAME)

        return extensions

    def __checkValidationLayerSupport(self):
        availableLayers = vkEnumerateInstanceLayerProperties()
        for layerName in validationLayers:
            layerFound = False

            for layerProperties in availableLayers:
                if layerName == ffi.string(layerProperties.layerName):
                    layerFound = True
                    break
            if not layerFound:
                return False

        return True

    def run(self):
        self.__initWindow()
        self.__initVulkan()
        self.__mainLoop()


if __name__ == '__main__':

    app = HelloTriangleApplication()

    app.run()

    del app
    glfw.terminate()

