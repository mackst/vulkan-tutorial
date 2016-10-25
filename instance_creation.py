from pyVulkan import *

import PyGlfwCffi as glfw


WIDTH = 800
HEIGHT = 600


class HelloTriangleApplication(object):

    def __init__(self):
        self.__window = None
        self.__instance = None

    def __del__(self):

        if self.__instance:
            vkDestroyInstance(self.__instance, None)

    def __initWindow(self):
        glfw.init()

        glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
        glfw.window_hint(glfw.RESIZABLE, False)

        self.__window = glfw.create_window(WIDTH, HEIGHT, "Vulkan")

    def __initVulkan(self):
        self.__createInstance()

    def __mainLoop(self):
        while not glfw.window_should_close(self.__window):
            glfw.poll_events()

    def __createInstance(self):
        appInfo = VkApplicationInfo(
            pApplicationName='Hello Triangle',
            applicationVersion=VK_MAKE_VERSION(1, 0, 0),
            pEngineName='No Engine',
            engineVersion=VK_MAKE_VERSION(1, 0, 0),
            apiVersion=VK_API_VERSION
        )

        createInfo = VkInstanceCreateInfo(pApplicationInfo=appInfo)
        glfwExtensions, glfwExtensionCount = glfw.getRequiredInstanceExtensions()

        # for i in range(glfwExtensionCount[0]): print ffi.string(glfwExtensions[i])

        createInfo.enabledExtensionCount = glfwExtensionCount[0]
        createInfo.ppEnabledExtensionNames = glfwExtensions

        createInfo.enabledLayerCount = 0

        self.__instance = vkCreateInstance(createInfo, None)

    def run(self):
        self.__initWindow()
        self.__initVulkan()
        self.__mainLoop()


if __name__ == '__main__':

    app = HelloTriangleApplication()

    app.run()

    del app
    glfw.terminate()

