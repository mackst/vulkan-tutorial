from pyVulkan import *

import PyGlfwCffi as glfw


WIDTH = 800
HEIGHT = 600


class HelloTriangleApplication(object):

    def __init__(self):
        self.__window = None

    def __initWindow(self):
        glfw.init()

        glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
        glfw.window_hint(glfw.RESIZABLE, False)

        self.__window = glfw.create_window(WIDTH, HEIGHT, "Vulkan")

    def __initVulkan(self):
        pass

    def __mainLoop(self):
        while not glfw.window_should_close(self.__window):
            glfw.poll_events()

    def run(self):
        self.__initWindow()
        self.__initVulkan()
        self.__mainLoop()


if __name__ == '__main__':

    app = HelloTriangleApplication()

    app.run()

