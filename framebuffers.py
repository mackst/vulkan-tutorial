from pyVulkan import *

import PyGlfwCffi as glfw


WIDTH = 800
HEIGHT = 600

validationLayers = ["VK_LAYER_LUNARG_standard_validation"]
deviceExtensions = [VK_KHR_SWAPCHAIN_EXTENSION_NAME]

enableValidationLayers = True


@vkDebugReportCallbackEXT
def debugCallback(*args):
    print (ffi.string(args[6]))
    return True

def CreateDebugReportCallbackEXT(instance, pCreateInfo, pAllocator):
    func = vkGetInstanceProcAddr(instance, 'vkCreateDebugReportCallbackEXT')
    if func:
        return func(instance, pCreateInfo, pAllocator)
    else:
        return VK_ERROR_EXTENSION_NOT_PRESENT

def DestroyDebugReportCallbackEXT(instance, callback, pAllocator):
    func = vkGetInstanceProcAddr(instance, 'vkDestroyDebugReportCallbackEXT')
    if func:
        func(instance, callback, pAllocator)

def destroySurface(instance, surface, pAllocator=None):
    func = vkGetInstanceProcAddr(instance, 'vkDestroySurfaceKHR')
    if func:
        func(instance, surface, pAllocator)

def destroySwapChain(device, swapChain, pAllocator=None):
    func = vkGetDeviceProcAddr(device, 'vkDestroySwapchainKHR')
    if func:
        func(device, swapChain, pAllocator)


class QueueFamilyIndices(object):

    def __init__(self):
        self.graphicsFamily = -1
        self.presentFamily = -1

    def isComplete(self):
        return self.graphicsFamily >= 0 and self.presentFamily >= 0


class SwapChainSupportDetails(object):
    def __init__(self):
        self.capabilities = None
        self.formats = None
        self.presentModes = None

class HelloTriangleApplication(object):

    def __init__(self):
        self.__window = None
        self.__instance = None
        self.__callback = None
        self.__surface = None
        self.__physicalDevice = None
        self.__device = None
        self.__graphicsQueue = None
        self.__presentQueue = None

        self.__swapChain = None
        self.__swapChainImages = None
        self.__swapChainImageFormat = None
        self.__swapChainExtent = None

        self.__swapChainImageViews = None
        self.__swapChainFramebuffers = None

        self.__renderPass = None
        self.__pipelineLayout = None
        self.__graphicsPipeline = None

    def __del__(self):
        if self.__swapChainFramebuffers:
            for i in self.__swapChainFramebuffers:
                vkDestroyFramebuffer(self.__device, i, None)
            self.__swapChainFramebuffers = None

        if self.__pipelineLayout:
            vkDestroyPipelineLayout(self.__device, self.__pipelineLayout, ffi.NULL)

        if self.__graphicsPipeline:
            vkDestroyPipeline(self.__device, self.__graphicsPipeline, None)

        if self.__renderPass:
            vkDestroyRenderPass(self.__device, self.__renderPass, None)

        for i in self.__swapChainImageViews:
            vkDestroyImageView(self.__device, i, None)

        if self.__swapChain:
            destroySwapChain(self.__device, self.__swapChain, None)

        if self.__device:
            vkDestroyDevice(self.__device, None)

        destroySurface(self.__instance, self.__surface, None)

        DestroyDebugReportCallbackEXT(self.__instance, self.__callback, None)

        vkDestroyInstance(self.__instance, None)

    def __initWindow(self):
        glfw.init()

        glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
        glfw.window_hint(glfw.RESIZABLE, False)

        self.__window = glfw.create_window(WIDTH, HEIGHT, "Vulkan")

    def __initVulkan(self):
        self.__createInstance()
        self.__setupDebugCallback()
        self.__createSurface()
        self.__pickPhysicalDevice()
        self.__createLogicalDevice()
        self.__createSwapChain()
        self.__createImageViews()
        self.__createRenderPass()
        self.__createGraphicsPipeline()
        self.__createFramebuffers()

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
            apiVersion=VK_MAKE_VERSION(1, 0, 1)
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
        # CreateDebugReportCallback = vkGetInstanceProcAddr(self.__instance, 'vkCreateDebugReportCallbackEXT')
        # self.__callback = CreateDebugReportCallback(self.__instance, createInfo, None)
        self.__callback = CreateDebugReportCallbackEXT(self.__instance, createInfo, None)
        if not self.__callback:
            raise Exception("failed to set up debug callback!")

    def __createSurface(self):
        surface = glfw.createWindowSurface(self.__instance, self.__window)
        self.__surface = ffi.cast('VkSurfaceKHR', surface)
        if self.__surface is None:
            raise Exception("failed to create window surface!")

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
        uniqueQueueFamilies = {}.fromkeys((indices.graphicsFamily, indices.presentFamily))
        queueCreateInfos = []
        for queueFamily in uniqueQueueFamilies:
            queueCreateInfo = VkDeviceQueueCreateInfo(
                queueFamilyIndex=queueFamily,
                queueCount=1,
                pQueuePriorities=[1.0]
            )
            queueCreateInfos.append(queueCreateInfo)

        f = [0 for i in range(55)]
        deviceFeatures = VkPhysicalDeviceFeatures()
        # deviceFeatures = VkPhysicalDeviceFeatures(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
        deArray = [ffi.new('char[]', i) for i in deviceExtensions]
        deviceExtensions_c = ffi.new('char*[]', deArray)
        createInfo = VkDeviceCreateInfo(
            flags=0,
            pQueueCreateInfos=queueCreateInfos,
            queueCreateInfoCount=len(queueCreateInfos),
            pEnabledFeatures=[deviceFeatures],
            enabledExtensionCount=len(deviceExtensions),
            ppEnabledExtensionNames=deviceExtensions_c
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
        self.__presentQueue = vkGetDeviceQueue(self.__device, indices.presentFamily, 0)

    def __createSwapChain(self):
        swapChainSupport = self.__querySwapChainSupport(self.__physicalDevice)

        surfaceFormat = self.__chooseSwapSurfaceFormat(swapChainSupport.formats)
        presentMode = self.__chooseSwapPresentMode(swapChainSupport.presentModes)
        extent = self.__chooseSwapExtent(swapChainSupport.capabilities)

        imageCount = swapChainSupport.capabilities.minImageCount + 1
        if swapChainSupport.capabilities.maxImageCount > 0 and imageCount > swapChainSupport.capabilities.maxImageCount:
            imageCount = swapChainSupport.capabilities.maxImageCount

        createInfo = VkSwapchainCreateInfoKHR(
            flags=0,
            surface=self.__surface,
            minImageCount=imageCount,
            imageFormat=surfaceFormat.format,
            imageColorSpace=surfaceFormat.colorSpace,
            imageExtent=extent,
            imageArrayLayers=1,
            imageUsage=VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
        )

        indices = self.__findQueueFamilies(self.__physicalDevice)
        if indices.graphicsFamily != indices.presentFamily:
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT
            createInfo.queueFamilyIndexCount = 2
            createInfo.pQueueFamilyIndices = [indices.graphicsFamily, indices.presentFamily]
        else:
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE

        createInfo.preTransform = swapChainSupport.capabilities.currentTransform
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR
        createInfo.presentMode = presentMode
        createInfo.clipped = True

        vkCreateSwapchainKHR = vkGetDeviceProcAddr(self.__device, 'vkCreateSwapchainKHR')
        self.__swapChain = vkCreateSwapchainKHR(self.__device, createInfo, None)

        vkGetSwapchainImagesKHR = vkGetDeviceProcAddr(self.__device, 'vkGetSwapchainImagesKHR')
        self.__swapChainImages = vkGetSwapchainImagesKHR(self.__device, self.__swapChain)

        self.__swapChainImageFormat = surfaceFormat.format
        self.__swapChainExtent = extent

    def __createImageViews(self):
        self.__swapChainImageViews = []
        components = VkComponentMapping(VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                                        VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY)
        subresourceRange = VkImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT,
                                                   0, 1, 0, 1)
        for i, image in enumerate(self.__swapChainImages):
            createInfo = VkImageViewCreateInfo(
                flags=0,
                image=image,
                viewType=VK_IMAGE_VIEW_TYPE_2D,
                format=self.__swapChainImageFormat,
                components=components,
                subresourceRange=subresourceRange
            )
            self.__swapChainImageViews.append(vkCreateImageView(self.__device, createInfo, None))

    def __createRenderPass(self):
        colorAttachment = VkAttachmentDescription(
            format=self.__swapChainImageFormat,
            samples=VK_SAMPLE_COUNT_1_BIT,
            loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR,
            storeOp=VK_ATTACHMENT_STORE_OP_STORE,
            stencilLoadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            stencilStoreOp=VK_ATTACHMENT_STORE_OP_DONT_CARE,
            initialLayout=VK_IMAGE_LAYOUT_UNDEFINED,
            finalLayout=VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
        )

        colorAttachmentRef = VkAttachmentReference(
            attachment=0,
            layout=VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
        )

        subPass = VkSubpassDescription(
            pipelineBindPoint=VK_PIPELINE_BIND_POINT_GRAPHICS,
            colorAttachmentCount=1,
            pColorAttachments=colorAttachmentRef
        )

        renderPassInfo = VkRenderPassCreateInfo(
            attachmentCount=1,
            pAttachments=colorAttachment,
            subpassCount=1,
            pSubpasses=subPass
        )

        self.__renderPass = vkCreateRenderPass(self.__device, renderPassInfo, ffi.NULL)

    def __createGraphicsPipeline(self):
        vertShaderModule = self.__createShaderModule('shaders/vert.spv')
        fragShaderModule = self.__createShaderModule('shaders/frag.spv')

        vertShaderStageInfo = VkPipelineShaderStageCreateInfo(
            flags=0,
            stage=VK_SHADER_STAGE_VERTEX_BIT,
            module=vertShaderModule,
            pName='main'
        )

        fragShaderStageInfo = VkPipelineShaderStageCreateInfo(
            flags=0,
            stage=VK_SHADER_STAGE_FRAGMENT_BIT,
            module=fragShaderModule,
            pName='main'
        )

        shaderStages = [vertShaderStageInfo, fragShaderStageInfo]

        vertexInputInfo = VkPipelineVertexInputStateCreateInfo(
            vertexBindingDescriptionCount=0,
            vertexAttributeDescriptionCount=0
        )

        inputAssembly = VkPipelineInputAssemblyStateCreateInfo(
            topology=VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            primitiveRestartEnable=True
        )

        viewport = VkViewport(0.0, 0.0,
                              float(self.__swapChainExtent.width),
                              float(self.__swapChainExtent.height),
                              0.0, 1.0)
        scissor = VkRect2D([0, 0], self.__swapChainExtent)
        viewportState = VkPipelineViewportStateCreateInfo(
            viewportCount=1,
            pViewports=viewport,
            scissorCount=1,
            pScissors=scissor
        )

        rasterizer = VkPipelineRasterizationStateCreateInfo(
            depthClampEnable=False,
            rasterizerDiscardEnable=False,
            polygonMode=VK_POLYGON_MODE_FILL,
            lineWidth=1.0,
            cullMode=VK_CULL_MODE_BACK_BIT,
            frontFace=VK_FRONT_FACE_CLOCKWISE,
            depthBiasEnable=False
        )

        multisampling = VkPipelineMultisampleStateCreateInfo(
            sampleShadingEnable=False,
            rasterizationSamples=VK_SAMPLE_COUNT_1_BIT
        )

        colorBlendAttachment = VkPipelineColorBlendAttachmentState(
            colorWriteMask=VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
            blendEnable=False
        )

        colorBlending = VkPipelineColorBlendStateCreateInfo(
            logicOpEnable=False,
            logicOp=VK_LOGIC_OP_COPY,
            attachmentCount=1,
            pAttachments=colorBlendAttachment,
            blendConstants=[0.0, 0.0, 0.0, 0.0]
        )

        pipelineLayoutInfo = VkPipelineLayoutCreateInfo(
            setLayoutCount=0,
            pushConstantRangeCount=0
        )

        self.__pipelineLayout = vkCreatePipelineLayout(self.__device, pipelineLayoutInfo, ffi.NULL)

        pipelineInfo = VkGraphicsPipelineCreateInfo(
            stageCount=2,
            pStages=shaderStages,
            pVertexInputState=vertexInputInfo,
            pInputAssemblyState=inputAssembly,
            pViewportState=viewportState,
            pRasterizationState=rasterizer,
            pMultisampleState=multisampling,
            pColorBlendState=colorBlending,
            layout=self.__pipelineLayout,
            renderPass=self.__renderPass,
            subpass=0
        )

        self.__graphicsPipeline = vkCreateGraphicsPipelines(self.__device, VK_NULL_HANDLE, 1, pipelineInfo, ffi.NULL)[0]

        vkDestroyShaderModule(self.__device, vertShaderModule, None)
        vkDestroyShaderModule(self.__device, fragShaderModule, None)

    def __createFramebuffers(self):
        self.__swapChainFramebuffers = []

        for imageView in self.__swapChainImageViews:
            attachments = [imageView,]

            framebufferInfo = VkFramebufferCreateInfo(
                renderPass=self.__renderPass,
                attachmentCount=1,
                pAttachments=attachments,
                width=self.__swapChainExtent.width,
                height=self.__swapChainExtent.height,
                layers=1
            )
            framebuffer = vkCreateFramebuffer(self.__device, framebufferInfo, None)
            self.__swapChainFramebuffers.append(framebuffer)

    def __createShaderModule(self, shaderFile):
        with open(shaderFile, 'rb') as sf:
            code = sf.read()
            codeSize = len(code)
            c_code = ffi.new('unsigned char []', code)
            pcode = ffi.cast('uint32_t*', c_code)

            createInfo = VkShaderModuleCreateInfo(codeSize=codeSize,pCode=pcode)

            return vkCreateShaderModule(self.__device, createInfo, None)


    def __chooseSwapSurfaceFormat(self, availableFormats):
        if len(availableFormats) == 1 and availableFormats[0].format == VK_FORMAT_UNDEFINED:
            return VkSurfaceFormatKHR(VK_FORMAT_B8G8R8A8_UNORM, 0)

        for availableFormat in availableFormats:
            if availableFormat.format == VK_FORMAT_B8G8R8A8_UNORM and availableFormat.colorSpace == 0:
                return availableFormat

        return availableFormats[0]

    def __chooseSwapPresentMode(self, availablePresentModes):
        for availablePresentMode in availablePresentModes:
            if availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR:
                return availablePresentMode

        return VK_PRESENT_MODE_FIFO_KHR

    def __chooseSwapExtent(self, capabilities):
        width = max(capabilities.minImageExtent.width, min(capabilities.maxImageExtent.width, WIDTH))
        height = max(capabilities.minImageExtent.height, min(capabilities.maxImageExtent.height, HEIGHT))
        return VkExtent2D(width, height)

    def __querySwapChainSupport(self, device):
        details = SwapChainSupportDetails()

        vkGetPhysicalDeviceSurfaceCapabilitiesKHR = vkGetInstanceProcAddr(self.__instance, 'vkGetPhysicalDeviceSurfaceCapabilitiesKHR')
        details.capabilities = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, self.__surface)

        vkGetPhysicalDeviceSurfaceFormatsKHR = vkGetInstanceProcAddr(self.__instance, 'vkGetPhysicalDeviceSurfaceFormatsKHR')
        details.formats = vkGetPhysicalDeviceSurfaceFormatsKHR(device, self.__surface)

        vkGetPhysicalDeviceSurfacePresentModesKHR = vkGetInstanceProcAddr(self.__instance, 'vkGetPhysicalDeviceSurfacePresentModesKHR')
        details.presentModes = vkGetPhysicalDeviceSurfacePresentModesKHR(device, self.__surface)

        return details

    def __isDeviceSuitable(self, device):
        indices = self.__findQueueFamilies(device)
        extensionsSupported = self.__checkDeviceExtensionSupport(device)
        swapChainAdequate = False
        if extensionsSupported:
            swapChainSupport = self.__querySwapChainSupport(device)
            swapChainAdequate = (not swapChainSupport.formats is None) and (not swapChainSupport.presentModes is None)
        return indices.isComplete() and extensionsSupported and swapChainAdequate

    def __checkDeviceExtensionSupport(self, device):
        availableExtensions = vkEnumerateDeviceExtensionProperties(device, None)

        for extension in availableExtensions:
            if ffi.string(extension.extensionName) in deviceExtensions:
                return True

        return False

    def __findQueueFamilies(self, device):
        vkGetPhysicalDeviceSurfaceSupportKHR = vkGetInstanceProcAddr(self.__instance,
                                                                   'vkGetPhysicalDeviceSurfaceSupportKHR')
        indices = QueueFamilyIndices()

        queueFamilies = vkGetPhysicalDeviceQueueFamilyProperties(device)

        for i, queueFamily in enumerate(queueFamilies):
            if queueFamily.queueCount > 0 and queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT:
                indices.graphicsFamily = i

            presentSupport = vkGetPhysicalDeviceSurfaceSupportKHR(device, i, self.__surface)

            if queueFamily.queueCount > 0 and presentSupport:
                indices.presentFamily = i

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

