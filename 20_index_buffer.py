# -*- coding: UTF-8 -*-

import sys
import array

from vulkan import *
from PySide2 import (QtGui, QtCore)

validationLayers = [
    'VK_LAYER_LUNARG_standard_validation'
]

deviceExtensions = [
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
]

enableValidationLayers = True


class InstanceProcAddr(object):
    T = None

    def __init__(self, func):
        self.__func = func

    def __call__(self, *args, **kwargs):
        funcName = self.__func.__name__
        func = InstanceProcAddr.procfunc(funcName)
        if func:
            return func(*args, **kwargs)
        else:
            return VK_ERROR_EXTENSION_NOT_PRESENT

    @staticmethod
    def procfunc(funcName):
        return vkGetInstanceProcAddr(InstanceProcAddr.T, funcName)


class DeviceProcAddr(InstanceProcAddr):

    @staticmethod
    def procfunc(funcName):
        return vkGetDeviceProcAddr(InstanceProcAddr.T, funcName)

# instance ext functions
@InstanceProcAddr
def vkCreateDebugReportCallbackEXT(instance, pCreateInfo, pAllocator):
    pass


@InstanceProcAddr
def vkDestroyDebugReportCallbackEXT(instance, pCreateInfo, pAllocator):
    pass


@InstanceProcAddr
def vkCreateWin32SurfaceKHR(instance, pCreateInfo, pAllocator):
    pass


@InstanceProcAddr
def vkDestroySurfaceKHR(instance, surface, pAllocator):
    pass


@InstanceProcAddr
def vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, queueFamilyIndex, surface):
    pass


@InstanceProcAddr
def vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface):
    pass


@InstanceProcAddr
def vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface):
    pass


@InstanceProcAddr
def vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface):
    pass


# device ext functions
@DeviceProcAddr
def vkCreateSwapchainKHR(device, pCreateInfo, pAllocator):
    pass

@DeviceProcAddr
def vkDestroySwapchainKHR(device, swapchain, pAllocator):
    pass

@DeviceProcAddr
def vkGetSwapchainImagesKHR(device, swapchain):
    pass

@DeviceProcAddr
def vkAcquireNextImageKHR(device, swapchain, timeout, semaphore, fence):
    pass

@DeviceProcAddr
def vkQueuePresentKHR(queue, pPresentInfo):
    pass

def debugCallback(*args):
    print('DEBUG: {} {}'.format(args[5], args[6]))
    return 0


class Win32misc(object):
    @staticmethod
    def getInstance(hWnd):
        from cffi import FFI as _FFI
        _ffi = _FFI()
        _ffi.cdef('long __stdcall GetWindowLongA(void* hWnd, int nIndex);')
        _lib = _ffi.dlopen('User32.dll')
        return _lib.GetWindowLongA(_ffi.cast('void*', hWnd), -6)  # GWL_HINSTANCE


class QueueFamilyIndices(object):

    def __init__(self):
        self.graphicsFamily = -1
        self.presentFamily = -1

    @property
    def isComplete(self):
        return self.graphicsFamily >= 0 and self.presentFamily >= 0


class SwapChainSupportDetails(object):

    def __init__(self):
        self.capabilities = None
        self.formats = None
        self.presentModes = None


class Vertex(object):

    POS = array.array('f', [0, 0])
    COLOR = array.array('f', [0, 0, 0])

    # def __init__(self):
    #     self.pos = []
    #     self.color = []

    @staticmethod
    def getBindingDescription():
        bindingDescription = VkVertexInputBindingDescription(
            binding=0,
            # stride=len(Vertex.POS) * Vertex.POS.itemsize,
            stride=len(Vertex.POS) * Vertex.POS.itemsize + len(Vertex.COLOR) * Vertex.COLOR.itemsize,
            inputRate=VK_VERTEX_INPUT_RATE_VERTEX
        )

        return bindingDescription

    @staticmethod
    def getAttributeDescriptions():
        pos = VkVertexInputAttributeDescription(
            location=0,
            binding=0,
            format=VK_FORMAT_R32G32_SFLOAT,
            offset=0
        )

        color = VkVertexInputAttributeDescription(
            location=1,
            binding=0,
            format=VK_FORMAT_R32G32B32_SFLOAT,
            offset=len(Vertex.POS) * Vertex.POS.itemsize,
        )
        return [pos, color]

class HelloTriangleApplication(QtGui.QWindow):

    def __init__(self):
        super(HelloTriangleApplication, self).__init__()

        self.setWidth(1280)
        self.setHeight(720)
        self.setMinimumWidth(40)
        self.setMinimumHeight(40)

        self.setTitle("Vulkan Python - PySide2")

        # self.setSurfaceType(self.OpenGLSurface)

        self.__instance = None
        self.__callbcak = None
        self.__surface = None

        self.__physicalDevice = None
        self.__device = None
        self.__graphicQueue = None
        self.__presentQueue = None

        self.__swapChain = None
        self.__swapChainImages = []
        self.__swapChainImageFormat = None
        self.__swapChainExtent = None
        self.__swapChainImageViews = []
        self.__swapChainFramebuffers = []

        self.__renderpass = None
        self.__pipeline = None
        self.__pipelineLayout = None

        self.__commandPool = None
        self.__commandBuffers = []

        self.__imageAvailableSemaphore = None
        self.__renderFinishedSemaphore = None

        self.__vertexBuffer = None
        self.__vertexBufferMemory = None

        self.__indexBuffer = None
        self.__indexBufferMemory = None

        self.__vertices = array.array('f', [
            # pos    color
            -.5, -.5, 1, 0, 0,
            .5, -.5, 0, 1, 0,
            .5, .5, 0, 0, 1,
            -.5, .5, 1, 1, 1
        ])

        self.__indices = array.array('H', [0, 1, 2, 2, 3, 0])
        # self.__indices = ffi.new('uint16_t[]', [0, 1, 2, 2, 3, 0])
        # self.__indices = numpy.array([0, 1, 2, 2, 3, 0], numpy.uint16)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.drawFrame)

        self.initVulkan()
        self.timer.start()

    def __del__(self):
        vkDeviceWaitIdle(self.__device)

        if self.__vertexBuffer:
            vkDestroyBuffer(self.__device, self.__vertexBuffer, None)

        if self.__vertexBufferMemory:
            vkFreeMemory(self.__device, self.__vertexBufferMemory, None)

        if self.__indexBuffer:
            vkDestroyBuffer(self.__device, self.__indexBuffer, None)

        if self.__indexBufferMemory:
            vkFreeMemory(self.__device, self.__indexBufferMemory, None)

        if self.__imageAvailableSemaphore:
            vkDestroySemaphore(self.__device, self.__imageAvailableSemaphore, None)
        if self.__renderFinishedSemaphore:
            vkDestroySemaphore(self.__device, self.__renderFinishedSemaphore, None)

        self.__cleanupSwapChain()

        if self.__commandPool:
            vkDestroyCommandPool(self.__device, self.__commandPool, None)

        if self.__device:
            vkDestroyDevice(self.__device, None)

        if self.__callbcak:
            vkDestroyDebugReportCallbackEXT(self.__instance, self.__callbcak, None)

        if self.__surface:
            vkDestroySurfaceKHR(self.__instance, self.__surface, None)

        if self.__instance:
            vkDestroyInstance(self.__instance, None)
            print('instance destroyed')

        self.destroy()

    def __cleanupSwapChain(self):
        [vkDestroyFramebuffer(self.__device, i, None) for i in self.__swapChainFramebuffers]
        self.__swapChainFramebuffers = []

        vkFreeCommandBuffers(self.__device, self.__commandPool, len(self.__commandBuffers), self.__commandBuffers)
        self.__swapChainFramebuffers = []

        vkDestroyPipeline(self.__device, self.__pipeline, None)
        vkDestroyPipelineLayout(self.__device, self.__pipelineLayout, None)
        vkDestroyRenderPass(self.__device, self.__renderpass, None)

        [vkDestroyImageView(self.__device, i, None) for i in self.__swapChainImageViews]
        self.__swapChainImageViews = []
        vkDestroySwapchainKHR(self.__device, self.__swapChain, None)

    def __recreateSwapChain(self):
        vkDeviceWaitIdle(self.__device)

        self.__cleanupSwapChain()
        self.__createSwapChain()
        self.__createImageViews()
        self.__createRenderPass()
        self.__createGraphicsPipeline()
        self.__createFrambuffers()
        self.__createCommandBuffers()

    def initVulkan(self):
        self.__cretaeInstance()
        self.__setupDebugCallback()
        self.__createSurface()
        self.__pickPhysicalDevice()
        self.__createLogicalDevice()
        self.__createSwapChain()
        self.__createImageViews()
        self.__createRenderPass()
        self.__createGraphicsPipeline()
        self.__createFrambuffers()
        self.__createCommandPool()
        self.__createVertexBuffer()
        self.__createIndexBuffer()
        self.__createCommandBuffers()
        self.__createSemaphores()

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
                # enabledLayerCount=len(validationLayers),
                ppEnabledLayerNames=validationLayers,
                # enabledExtensionCount=len(extenstions),
                ppEnabledExtensionNames=extenstions
            )
        else:
            instanceInfo = VkInstanceCreateInfo(
                pApplicationInfo=appInfo,
                enabledLayerCount=0,
                # enabledExtensionCount=len(extenstions),
                ppEnabledExtensionNames=extenstions
            )

        self.__instance = vkCreateInstance(instanceInfo, None)

        InstanceProcAddr.T = self.__instance

    def __setupDebugCallback(self):
        if not enableValidationLayers:
            return

        createInfo = VkDebugReportCallbackCreateInfoEXT(
            flags=VK_DEBUG_REPORT_WARNING_BIT_EXT | VK_DEBUG_REPORT_ERROR_BIT_EXT,
            pfnCallback=debugCallback
        )

        self.__callbcak = vkCreateDebugReportCallbackEXT(self.__instance, createInfo, None)

    def __createSurface(self):
        if sys.platform == 'win32':
            hwnd = self.winId()
            hinstance = Win32misc.getInstance(hwnd)
            createInfo = VkWin32SurfaceCreateInfoKHR(
                hinstance=hinstance,
                hwnd=hwnd
            )

            self.__surface = vkCreateWin32SurfaceKHR(self.__instance, createInfo, None)
        # elif sys.platform == 'linux':
        #     pass

    def __pickPhysicalDevice(self):
        physicalDevices = vkEnumeratePhysicalDevices(self.__instance)

        for device in physicalDevices:
            if self.__isDeviceSuitable(device):
                self.__physicalDevice = device
                break

        assert self.__physicalDevice != None

    def __createLogicalDevice(self):
        indices = self.__findQueueFamilies(self.__physicalDevice)

        uniqueQueueFamilies = {}.fromkeys([indices.graphicsFamily, indices.presentFamily])
        queueCreateInfos = []
        for i in uniqueQueueFamilies:
            queueCreateInfo = VkDeviceQueueCreateInfo(
                queueFamilyIndex=i,
                queueCount=1,
                pQueuePriorities=[1.0]
            )
            queueCreateInfos.append(queueCreateInfo)

        deviceFeatures = VkPhysicalDeviceFeatures()
        if enableValidationLayers:
            createInfo = VkDeviceCreateInfo(
                # queueCreateInfoCount=len(queueCreateInfos),
                pQueueCreateInfos=queueCreateInfos,
                # enabledExtensionCount=len(deviceExtensions),
                ppEnabledExtensionNames=deviceExtensions,
                # enabledLayerCount=len(validationLayers),
                ppEnabledLayerNames=validationLayers,
                pEnabledFeatures=deviceFeatures
            )
        else:
            createInfo = VkDeviceCreateInfo(
                queueCreateInfoCount=1,
                pQueueCreateInfos=queueCreateInfo,
                # enabledExtensionCount=len(deviceExtensions),
                ppEnabledExtensionNames=deviceExtensions,
                enabledLayerCount=0,
                pEnabledFeatures=deviceFeatures
            )

        self.__device = vkCreateDevice(self.__physicalDevice, createInfo, None)

        DeviceProcAddr.T = self.__device

        self.__graphicQueue = vkGetDeviceQueue(self.__device, indices.graphicsFamily, 0)
        self.__presentQueue = vkGetDeviceQueue(self.__device, indices.presentFamily, 0)

    def __createSwapChain(self):
        swapChainSupport = self.__querySwapChainSupport(self.__physicalDevice)

        surfaceFormat = self.__chooseSwapSurfaceFormat(swapChainSupport.formats)
        presentMode = self.__chooseSwapPresentMode(swapChainSupport.presentModes)
        extent = self.__chooseSwapExtent(swapChainSupport.capabilities)

        imageCount = swapChainSupport.capabilities.minImageCount + 1
        if swapChainSupport.capabilities.maxImageCount > 0 and imageCount > swapChainSupport.capabilities.maxImageCount:
            imageCount = swapChainSupport.capabilities.maxImageCount

        indices = self.__findQueueFamilies(self.__physicalDevice)
        queueFamily = {}.fromkeys([indices.graphicsFamily, indices.presentFamily])
        queueFamilies = list(queueFamily.keys())
        if len(queueFamilies) > 1:
            createInfo = VkSwapchainCreateInfoKHR(
                surface=self.__surface,
                minImageCount=imageCount,
                imageFormat=surfaceFormat.format,
                imageColorSpace=surfaceFormat.colorSpace,
                imageExtent=extent,
                imageArrayLayers=1,
                imageUsage=VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                # queueFamilyIndexCount=len(queueFamilies),
                pQueueFamilyIndices=queueFamilies,
                imageSharingMode=VK_SHARING_MODE_CONCURRENT,
                preTransform=swapChainSupport.capabilities.currentTransform,
                compositeAlpha=VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
                presentMode=presentMode,
                clipped=True
            )
        else:
            createInfo = VkSwapchainCreateInfoKHR(
                surface=self.__surface,
                minImageCount=imageCount,
                imageFormat=surfaceFormat.format,
                imageColorSpace=surfaceFormat.colorSpace,
                imageExtent=extent,
                imageArrayLayers=1,
                imageUsage=VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                # queueFamilyIndexCount=len(queueFamilies),
                pQueueFamilyIndices=queueFamilies,
                imageSharingMode=VK_SHARING_MODE_EXCLUSIVE,
                preTransform=swapChainSupport.capabilities.currentTransform,
                compositeAlpha=VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
                presentMode=presentMode,
                clipped=True
            )

        self.__swapChain = vkCreateSwapchainKHR(self.__device, createInfo, None)
        assert self.__swapChain != None

        self.__swapChainImages = vkGetSwapchainImagesKHR(self.__device, self.__swapChain)

        self.__swapChainImageFormat = surfaceFormat.format
        self.__swapChainExtent = extent

    def __createImageViews(self):
        self.__swapChainImageViews = []

        for i, image in enumerate(self.__swapChainImages):
            ssr = VkImageSubresourceRange(
                VK_IMAGE_ASPECT_COLOR_BIT,
                0, 1, 0, 1
            )

            createInfo = VkImageViewCreateInfo(
                image=image,
                viewType=VK_IMAGE_VIEW_TYPE_2D,
                format=self.__swapChainImageFormat,
                components=[VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                            VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY],
                subresourceRange=ssr
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
            0,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
        )

        subpass = VkSubpassDescription(
            pipelineBindPoint=VK_PIPELINE_BIND_POINT_GRAPHICS,
            pColorAttachments=[colorAttachmentRef]
        )

        renderPassInfo = VkRenderPassCreateInfo(
            pAttachments=[colorAttachment],
            pSubpasses=[subpass]
        )

        self.__renderpass = vkCreateRenderPass(self.__device, renderPassInfo, None)

    def __createGraphicsPipeline(self):
        vertexShaderMode = self.__createShaderModule('shader/vert.spv')
        fragmentShaderMode = self.__createShaderModule('shader/frag.spv')

        vertexShaderStageInfo = VkPipelineShaderStageCreateInfo(
            stage=VK_SHADER_STAGE_VERTEX_BIT,
            module=vertexShaderMode,
            pName='main'
        )
        fragmentShaderStageInfo = VkPipelineShaderStageCreateInfo(
            stage=VK_SHADER_STAGE_FRAGMENT_BIT,
            module=fragmentShaderMode,
            pName='main'
        )

        shaderStageInfos = [vertexShaderStageInfo, fragmentShaderStageInfo]

        bindingDescription = Vertex.getBindingDescription()
        attributeDescription = Vertex.getAttributeDescriptions()

        vertexInputInfo = VkPipelineVertexInputStateCreateInfo(
            # vertexBindingDescriptionCount=0,
            pVertexBindingDescriptions=[bindingDescription],
            # vertexAttributeDescriptionCount=0,
            pVertexAttributeDescriptions=attributeDescription,
        )

        inputAssembly = VkPipelineInputAssemblyStateCreateInfo(
            topology=VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            primitiveRestartEnable=False
        )

        viewport = VkViewport(0.0, 0.0,
                              float(self.__swapChainExtent.width),
                              float(self.__swapChainExtent.height),
                              0.0, 1.0)

        scissor = VkRect2D([0, 0], self.__swapChainExtent)
        viewportStage = VkPipelineViewportStateCreateInfo(
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

        colorBending = VkPipelineColorBlendStateCreateInfo(
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

        self.__pipelineLayout = vkCreatePipelineLayout(self.__device, pipelineLayoutInfo, None)

        pipelineInfo = VkGraphicsPipelineCreateInfo(
            # stageCount=len(shaderStageInfos),
            pStages=shaderStageInfos,
            pVertexInputState=vertexInputInfo,
            pInputAssemblyState=inputAssembly,
            pViewportState=viewportStage,
            pRasterizationState=rasterizer,
            pMultisampleState=multisampling,
            pColorBlendState=colorBending,
            layout=self.__pipelineLayout,
            renderPass=self.__renderpass,
            subpass=0,
            basePipelineHandle=VK_NULL_HANDLE
        )

        self.__pipeline = vkCreateGraphicsPipelines(self.__device, VK_NULL_HANDLE, 1, pipelineInfo, None)#[0]

        vkDestroyShaderModule(self.__device, vertexShaderMode, None)
        vkDestroyShaderModule(self.__device, fragmentShaderMode, None)

    def __createFrambuffers(self):
        self.__swapChainFramebuffers = []
        for i, iv in enumerate(self.__swapChainImageViews):
            framebufferInfo = VkFramebufferCreateInfo(
                renderPass=self.__renderpass,
                pAttachments=[iv],
                width=self.__swapChainExtent.width,
                height=self.__swapChainExtent.height,
                layers=1
            )

            self.__swapChainFramebuffers.append(vkCreateFramebuffer(self.__device, framebufferInfo, None))

    def __createCommandPool(self):
        queueFamilyIndices = self.__findQueueFamilies(self.__physicalDevice)

        createInfo = VkCommandPoolCreateInfo(
            queueFamilyIndex=queueFamilyIndices.graphicsFamily
        )

        self.__commandPool = vkCreateCommandPool(self.__device, createInfo, None)

    def __createVertexBuffer(self):
        bufferSize = len(self.__vertices) * self.__vertices.itemsize

        stagingBuffer, stagingMemory = self.__createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)

        data = vkMapMemory(self.__device, stagingMemory, 0, bufferSize, 0)
        vertePtr = ffi.cast('float *', self.__vertices.buffer_info()[0])
        ffi.memmove(data, vertePtr, bufferSize)
        vkUnmapMemory(self.__device, stagingMemory)

        self.__vertexBuffer, self.__vertexBufferMemory = self.__createBuffer(bufferSize,
                                                                             VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                                                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
        self.__copyBuffer(stagingBuffer, self.__vertexBuffer, bufferSize)

        vkDestroyBuffer(self.__device, stagingBuffer, None)
        vkFreeMemory(self.__device, stagingMemory, None)

    def __createIndexBuffer(self):
        bufferSize = len(self.__indices) * self.__indices.itemsize
        # bufferSize = ffi.sizeof(self.__indices)

        stagingBuffer, stagingMemory = self.__createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
        data = vkMapMemory(self.__device, stagingMemory, 0, bufferSize, 0)
        indicesPtr = ffi.cast('uint16_t *', self.__indices.buffer_info()[0])
        # vertePtr = ffi.cast('uint16_t*', self.__indices.ctypes.data)
        # ffi.memmove(data, ffi.addressof(self.__indices), bufferSize)
        ffi.memmove(data, indicesPtr, bufferSize)
        vkUnmapMemory(self.__device, stagingMemory)

        self.__indexBuffer, self.__indexBufferMemory = self.__createBuffer(bufferSize,
                                                                           VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                                                                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)

        self.__copyBuffer(stagingBuffer, self.__indexBuffer, bufferSize)

        vkDestroyBuffer(self.__device, stagingBuffer, None)
        vkFreeMemory(self.__device, stagingMemory, None)

    def __createBuffer(self, size, usage, properties):
        buffer = None
        bufferMemory = None

        bufferInfo = VkBufferCreateInfo(
            size=size,
            usage=usage,
            sharingMode=VK_SHARING_MODE_EXCLUSIVE
        )

        buffer = vkCreateBuffer(self.__device, bufferInfo, None)

        memRequirements = vkGetBufferMemoryRequirements(self.__device, buffer)
        allocInfo = VkMemoryAllocateInfo(
            allocationSize=memRequirements.size,
            memoryTypeIndex=self.__findMemoryType(memRequirements.memoryTypeBits, properties)
        )
        bufferMemory = vkAllocateMemory(self.__device, allocInfo, None)

        vkBindBufferMemory(self.__device, buffer, bufferMemory, 0)

        return (buffer, bufferMemory)

    def __copyBuffer(self, src, dst, bufferSize):
        allocInfo = VkCommandBufferAllocateInfo(
            level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandPool=self.__commandPool,
            commandBufferCount=1
        )

        commandBuffer = vkAllocateCommandBuffers(self.__device, allocInfo)[0]
        beginInfo = VkCommandBufferBeginInfo(flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT)
        vkBeginCommandBuffer(commandBuffer, beginInfo)

        # copyRegion = VkBufferCopy(size=bufferSize)
        copyRegion = VkBufferCopy(0, 0, bufferSize)
        vkCmdCopyBuffer(commandBuffer, src, dst, 1, [copyRegion])

        vkEndCommandBuffer(commandBuffer)

        submitInfo = VkSubmitInfo(pCommandBuffers=[commandBuffer])

        vkQueueSubmit(self.__graphicQueue, 1, [submitInfo], VK_NULL_HANDLE)
        vkQueueWaitIdle(self.__graphicQueue)

        vkFreeCommandBuffers(self.__device, self.__commandPool, 1, [commandBuffer])

    def __findMemoryType(self, typeFilter, properties):
        memProperties = vkGetPhysicalDeviceMemoryProperties(self.__physicalDevice)

        for i, prop in enumerate(memProperties.memoryTypes):
            if (typeFilter & (1 << i)) and ((prop.propertyFlags & properties) == properties):
                return i

        return -1

    def __createCommandBuffers(self):
        self.__commandBuffers = []

        allocInfo = VkCommandBufferAllocateInfo(
            commandPool=self.__commandPool,
            level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=len(self.__swapChainFramebuffers)
        )

        self.__commandBuffers = vkAllocateCommandBuffers(self.__device, allocInfo)

        for i, buffer in enumerate(self.__commandBuffers):
            beginInfo = VkCommandBufferBeginInfo(flags=VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT)
            vkBeginCommandBuffer(buffer, beginInfo)

            renderArea = VkRect2D([0, 0], self.__swapChainExtent)
            clearColor = VkClearValue(color=[[0.0, 0.0, 0.0, 1.0]])
            renderPassInfo = VkRenderPassBeginInfo(
                renderPass=self.__renderpass,
                framebuffer=self.__swapChainFramebuffers[i],
                renderArea=renderArea,
                pClearValues=[clearColor]
            )

            vkCmdBeginRenderPass(buffer, renderPassInfo, VK_SUBPASS_CONTENTS_INLINE)

            vkCmdBindPipeline(buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, self.__pipeline)

            vkCmdBindVertexBuffers(buffer, 0, 1, [self.__vertexBuffer], [0])

            vkCmdBindIndexBuffer(buffer, self.__indexBuffer, 0, VK_INDEX_TYPE_UINT16)

            vkCmdDrawIndexed(buffer, len(self.__indices), 1, 0, 0, 0)

            vkCmdEndRenderPass(buffer)

            vkEndCommandBuffer(buffer)

    def __createSemaphores(self):
        semaphoreInfo = VkSemaphoreCreateInfo()

        self.__imageAvailableSemaphore = vkCreateSemaphore(self.__device, semaphoreInfo, None)
        self.__renderFinishedSemaphore = vkCreateSemaphore(self.__device, semaphoreInfo, None)

    def drawFrame(self):
        if not self.isExposed():
            return

        try:
            imageIndex = vkAcquireNextImageKHR(self.__device, self.__swapChain, 18446744073709551615,
                                               self.__imageAvailableSemaphore, VK_NULL_HANDLE)
        except VkErrorSurfaceLostKhr:
            self.__recreateSwapChain()
            return
        # else:
        #     raise Exception('faild to acquire next image.')

        waitSemaphores = [self.__imageAvailableSemaphore]
        signalSemaphores = [self.__renderFinishedSemaphore]
        waitStages = [VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT]
        submit = VkSubmitInfo(
            pWaitSemaphores=waitSemaphores,
            pWaitDstStageMask=waitStages,
            pCommandBuffers=[self.__commandBuffers[imageIndex]],
            pSignalSemaphores=signalSemaphores
        )

        vkQueueSubmit(self.__graphicQueue, 1, submit, VK_NULL_HANDLE)

        presenInfo = VkPresentInfoKHR(
            pWaitSemaphores=signalSemaphores,
            pSwapchains=[self.__swapChain],
            pImageIndices=[imageIndex]
        )

        try:
            vkQueuePresentKHR(self.__presentQueue, presenInfo)
        except VkErrorOutOfDateKhr:
            self.__recreateSwapChain()

        if enableValidationLayers:
            vkQueueWaitIdle(self.__presentQueue)

    def __createShaderModule(self, shaderFile):
        with open(shaderFile, 'rb') as sf:
            code = sf.read()

            createInfo = VkShaderModuleCreateInfo(
                codeSize=len(code),
                pCode=code
            )

            return vkCreateShaderModule(self.__device, createInfo, None)

    def __chooseSwapSurfaceFormat(self, formats):
        if len(formats) == 1 and formats[0].format == VK_FORMAT_UNDEFINED:
            return [VK_FORMAT_B8G8R8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR]

        for i in formats:
            if i.format == VK_FORMAT_B8G8R8_UNORM and i.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR:
                return i

        return formats[0]

    def __chooseSwapPresentMode(self, presentModes):
        bestMode = VK_PRESENT_MODE_FIFO_KHR

        for i in presentModes:
            if i == VK_PRESENT_MODE_FIFO_KHR:
                return i
            elif i == VK_PRESENT_MODE_MAILBOX_KHR:
                return i
            elif i == VK_PRESENT_MODE_IMMEDIATE_KHR:
                return i

        return bestMode

    def __chooseSwapExtent(self, capabilities):
        width = max(capabilities.minImageExtent.width, min(capabilities.maxImageExtent.width, self.width()))
        height = max(capabilities.minImageExtent.height, min(capabilities.maxImageExtent.height, self.height()))
        return VkExtent2D(width, height)

    def __querySwapChainSupport(self, device):
        detail = SwapChainSupportDetails()

        detail.capabilities = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, self.__surface)
        detail.formats = vkGetPhysicalDeviceSurfaceFormatsKHR(device, self.__surface)
        detail.presentModes = vkGetPhysicalDeviceSurfacePresentModesKHR(device, self.__surface)
        return detail

    def __isDeviceSuitable(self, device):
        indices = self.__findQueueFamilies(device)

        return indices.isComplete

    def __findQueueFamilies(self, device):
        indices = QueueFamilyIndices()

        familyProperties = vkGetPhysicalDeviceQueueFamilyProperties(device)
        for i, prop in enumerate(familyProperties):
            if prop.queueCount > 0 and prop.queueFlags & VK_QUEUE_GRAPHICS_BIT:
                indices.graphicsFamily = i

            presentSupport = vkGetPhysicalDeviceSurfaceSupportKHR(device, i, self.__surface)

            if prop.queueCount > 0 and presentSupport:
                indices.presentFamily = i

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

    def resizeEvent(self, event):
        if event.size() != event.oldSize():
            self.__recreateSwapChain()
        super(HelloTriangleApplication, self).resizeEvent(event)


if __name__ == '__main__':
    import sys

    app = QtGui.QGuiApplication(sys.argv)

    win = HelloTriangleApplication()
    win.show()

    def clenaup():
        global win
        win.timer.stop()
        del win


    app.aboutToQuit.connect(clenaup)

    sys.exit(app.exec_())
