import sys
import time

from vulkan import *

from PyQt5 import (QtGui, QtCore)
from PIL import Image
import numpy as np
import tinyobjloader as tol

import glm


WIDTH = 800
HEIGHT = 600

MODEL_PATH = "models/chalet.obj"
TEXTURE_PATH = "textures/chalet.jpg"

validationLayers = ["VK_LAYER_LUNARG_standard_validation"]
deviceExtensions = [VK_KHR_SWAPCHAIN_EXTENSION_NAME]

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

def destroySurface(instance, surface, pAllocator=None):
    func = vkGetInstanceProcAddr(instance, 'vkDestroySurfaceKHR')
    if func:
        func(instance, surface, pAllocator)

def destroySwapChain(device, swapChain, pAllocator=None):
    func = vkGetDeviceProcAddr(device, 'vkDestroySwapchainKHR')
    if func:
        func(device, swapChain, pAllocator)


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

    def isComplete(self):
        return self.graphicsFamily >= 0 and self.presentFamily >= 0


class SwapChainSupportDetails(object):
    def __init__(self):
        self.capabilities = None
        self.formats = None
        self.presentModes = None

class Vertex(object):
    POS = np.array([0, 0, 0], dtype=np.float32)
    COLOR = np.array([0, 0, 0], dtype=np.float32)
    TEXCOORD = np.array([0, 0], dtype=np.float32)

    @staticmethod
    def getBindingDescription():
        bindingDescription = VkVertexInputBindingDescription(
            binding=0,
            stride=Vertex.numBytes(),
            inputRate=VK_VERTEX_INPUT_RATE_VERTEX
        )
        return bindingDescription

    @staticmethod
    def getAttributeDescriptions():
        attributeDescription1 = VkVertexInputAttributeDescription(
            binding=0,
            location=0,
            format=VK_FORMAT_R32G32B32_SFLOAT,
            offset=0
        )
        attributeDescription2 = VkVertexInputAttributeDescription(
            binding=0,
            location=1,
            format=VK_FORMAT_R32G32B32_SFLOAT,
            offset=Vertex.POS.nbytes
        )
        attributeDescription3 = VkVertexInputAttributeDescription(
            binding=0,
            location=2,
            format=VK_FORMAT_R32G32_SFLOAT,
            offset=Vertex.POS.nbytes+Vertex.COLOR.nbytes
        )

        return [attributeDescription1, attributeDescription2, attributeDescription3]

    @staticmethod
    def numBytes():
        return Vertex.POS.nbytes+Vertex.COLOR.nbytes+Vertex.TEXCOORD.nbytes

class UniformBufferObject(object):

    def __init__(self, model=None, view=None, proj=None):
        self.model = model
        self.view = view
        self.proj = proj

    @property
    def nbytes(self):
        return self.model.nbytes + self.view.nbytes + self.proj.nbytes

    @property
    def to_c_ptr(self):
        a = np.concatenate((self.model, self.view, self.proj))

        return ffi.cast('float*', a.ctypes.data)

class HelloTriangleApplication(QtGui.QWindow):

    def __init__(self):
        super(HelloTriangleApplication, self).__init__(None)
        
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
        self.__descriptorSetLayout = None
        self.__pipelineLayout = None
        self.__graphicsPipeline = None

        self.__commandPool = None

        self.__depthImage = None
        self.__depthImageMemory = None
        self.__depthImageView = None

        self.__textureImage = None
        self.__textureImageMemory = None
        self.__textureImageView = None
        self.__textureSampler = None

        self.__vertexBuffer = None
        self.__vertexBufferMemory = None
        self.__indexBuffer = None
        self.__indexBufferMemory = None

        self.__uniformBuffer = None
        self.__uniformBufferMemory = None

        self.__descriptorPool = None
        self.__descriptorSet = None

        self.__commandBuffers = None

        self.__imageAvailableSemaphore = None
        self.__renderFinishedSemaphore = None

        self.vertices = None
        self.indices = None

        self.__timer = QtCore.QTimer(self)
        self.__timer.timeout.connect(self.__mainLoop)
        self.__startTime = time.time()

    def __del__(self):
        vkDeviceWaitIdle(self.__device)

        self.__cleanupSwapChain()

        if self.__descriptorPool:
            vkDestroyDescriptorPool(self.__device, self.__descriptorPool, None)

        if self.__descriptorSetLayout:
            vkDestroyDescriptorSetLayout(self.__device, self.__descriptorSetLayout, None)

        if self.__imageAvailableSemaphore:
            vkDestroySemaphore(self.__device, self.__imageAvailableSemaphore, None)

        if self.__renderFinishedSemaphore:
            vkDestroySemaphore(self.__device, self.__renderFinishedSemaphore, None)

        if self.__textureSampler:
            vkDestroySampler(self.__device, self.__textureSampler, None)

        if self.__textureImageView:
            vkDestroyImageView(self.__device, self.__textureImageView, None)

        if self.__textureImageMemory:
            vkFreeMemory(self.__device, self.__textureImageMemory, None)

        if self.__textureImage:
            vkDestroyImage(self.__device, self.__textureImage, None)

        if self.__uniformBufferMemory:
            vkFreeMemory(self.__device, self.__uniformBufferMemory, None)

        if self.__uniformBuffer:
            vkDestroyBuffer(self.__device, self.__uniformBuffer, None)

        if self.__indexBufferMemory:
            vkFreeMemory(self.__device, self.__indexBufferMemory, None)

        if self.__indexBuffer:
            vkDestroyBuffer(self.__device, self.__indexBuffer, None)

        if self.__vertexBufferMemory:
            vkFreeMemory(self.__device, self.__vertexBufferMemory, None)

        if self.__vertexBuffer:
            vkDestroyBuffer(self.__device, self.__vertexBuffer, None)

        if self.__commandPool:
            vkDestroyCommandPool(self.__device, self.__commandPool, None)

        if self.__device:
            vkDestroyDevice(self.__device, None)

        if self.__surface:
            destroySurface(self.__instance, self.__surface, None)

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
        self.__createSurface()
        self.__pickPhysicalDevice()
        self.__createLogicalDevice()
        self.__createSwapChain()
        self.__createImageViews()
        self.__createRenderPass()
        self.__createDescriptorSetLayout()
        self.__createGraphicsPipeline()
        self.__createCommandPool()
        self.__createDepthResources()
        self.__createFramebuffers()
        self.__createTextureImage()
        self.__createTextureImageView()
        self.__createTextureSampler()
        self.__loadModel()
        self.__createVertexBuffer()
        self.__createIndexBuffer()
        self.__createUniformBuffer()
        self.__createDescriptorPool()
        self.__createDescriptorSet()
        self.__createCommandBuffers()
        self.__createSemaphores()

    def __mainLoop(self):
        self.__updateUniformBuffer()
        self.__drawFrame()

    def __cleanupSwapChain(self):
        vkDestroyImageView(self.__device, self.__depthImageView, None)
        vkFreeMemory(self.__device, self.__depthImageMemory, None)
        vkDestroyImage(self.__device, self.__depthImage, None)

        for buf in self.__swapChainFramebuffers:
            vkDestroyFramebuffer(self.__device, buf, None)
        self.__swapChainFramebuffers = None

        vkFreeCommandBuffers(self.__device, self.__commandPool, len(self.__commandBuffers), self.__commandBuffers)
        self.__commandBuffers = None

        vkDestroyPipeline(self.__device, self.__graphicsPipeline, None)
        vkDestroyPipelineLayout(self.__device, self.__pipelineLayout, None)
        vkDestroyRenderPass(self.__device, self.__renderPass, None)

        for iv in self.__swapChainImageViews:
            vkDestroyImageView(self.__device, iv, None)
        self.__swapChainImageViews = None

        destroySwapChain(self.__device, self.__swapChain)

    def recreateSwapChain(self):
        vkDeviceWaitIdle(self.__device)

        self.__cleanupSwapChain()

        self.__createSwapChain()
        self.__createImageViews()
        self.__createRenderPass()
        self.__createGraphicsPipeline()
        self.__createDepthResources()
        self.__createFramebuffers()
        self.__createCommandBuffers()

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

    def __createSurface(self):
        if sys.platform == 'win32':
            vkCreateWin32SurfaceKHR = vkGetInstanceProcAddr(self.__instance, 'vkCreateWin32SurfaceKHR')

            hwnd = self.winId()
            hinstance = Win32misc.getInstance(hwnd)
            createInfo = VkWin32SurfaceCreateInfoKHR(
                sType=VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR,
                hinstance=hinstance,
                hwnd=hwnd
            )
            self.__surface = vkCreateWin32SurfaceKHR(self.__instance, createInfo, None)
        elif sys.platform == 'linux' or sys.platform == 'linux2':
            from PyQt5 import QtX11Extras
            import sip

            vkCreateXcbSurfaceKHR = vkGetInstanceProcAddr(self.__instance, 'vkCreateXcbSurfaceKHR')

            connection = sip.unwrapinstance(QtX11Extras.QX11Info.connection())
            createInfo = VkXcbSurfaceCreateInfoKHR(
                sType=VK_STRUCTURE_TYPE_XCB_SURFACE_CREATE_INFO_KHR,
                connection=connection,
                window=self.winId()
            )
            self.__surface = vkCreateXcbSurfaceKHR(self.__instance, createInfo, None)
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
                sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                queueFamilyIndex=queueFamily,
                queueCount=1,
                pQueuePriorities=[1.0]
            )
            queueCreateInfos.append(queueCreateInfo)

        deviceFeatures = VkPhysicalDeviceFeatures()
        deviceFeatures.samplerAnisotropy = VK_TRUE

        if enableValidationLayers:
            createInfo = VkDeviceCreateInfo(
                sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                pQueueCreateInfos=queueCreateInfos,
                queueCreateInfoCount=1,
                pEnabledFeatures=[deviceFeatures],
                enabledExtensionCount=len(deviceExtensions),
                ppEnabledExtensionNames=deviceExtensions,
                enabledLayerCount=len(validationLayers),
                ppEnabledLayerNames=validationLayers
            )
        else:
            createInfo = VkDeviceCreateInfo(
                sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                pQueueCreateInfos=queueCreateInfos,
                queueCreateInfoCount=1,
                pEnabledFeatures=[deviceFeatures],
                enabledExtensionCount=len(deviceExtensions),
                ppEnabledExtensionNames=deviceExtensions,
                enabledLayerCount=0
            )

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
            sType=VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
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
        self.__swapChainImageViews = [
            self.__createImageView(image, self.__swapChainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT) for image
            in self.__swapChainImages]

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

        depthAttachment = VkAttachmentDescription(
            format=self.__findDepthFormat(),
            samples=VK_SAMPLE_COUNT_1_BIT,
            loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR,
            storeOp=VK_ATTACHMENT_STORE_OP_DONT_CARE,
            stencilLoadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            stencilStoreOp=VK_ATTACHMENT_STORE_OP_DONT_CARE,
            initialLayout=VK_IMAGE_LAYOUT_UNDEFINED,
            finalLayout=VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        )

        colorAttachmentRef = VkAttachmentReference(
            attachment=0,
            layout=VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
        )

        depthAttachmentRef = VkAttachmentReference(
            attachment=1,
            layout=VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        )

        subPass = VkSubpassDescription(
            pipelineBindPoint=VK_PIPELINE_BIND_POINT_GRAPHICS,
            colorAttachmentCount=1,
            pColorAttachments=[colorAttachmentRef],
            pDepthStencilAttachment=depthAttachmentRef
        )

        dependency = VkSubpassDependency(
            srcSubpass=VK_SUBPASS_EXTERNAL,
            dstSubpass=0,
            srcStageMask=VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            srcAccessMask=VK_ACCESS_MEMORY_READ_BIT,
            dstStageMask=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            dstAccessMask=VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
        )

        attachments = [colorAttachment, depthAttachment]
        renderPassInfo = VkRenderPassCreateInfo(
            sType=VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            attachmentCount=len(attachments),
            pAttachments=attachments,
            subpassCount=1,
            pSubpasses=[subPass],
            dependencyCount=1,
            pDependencies=[dependency]
        )

        self.__renderPass = vkCreateRenderPass(self.__device, renderPassInfo, None)

    def __createDescriptorSetLayout(self):
        uboLayoutBinding = VkDescriptorSetLayoutBinding(
            binding=0,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            stageFlags=VK_SHADER_STAGE_VERTEX_BIT
        )

        samplerLayoutBinding = VkDescriptorSetLayoutBinding(
            binding=1,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            stageFlags=VK_SHADER_STAGE_FRAGMENT_BIT
        )

        bindings = [uboLayoutBinding, samplerLayoutBinding]
        layoutInfo = VkDescriptorSetLayoutCreateInfo(
            sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            bindingCount=len(bindings),
            pBindings=bindings
        )

        self.__descriptorSetLayout = vkCreateDescriptorSetLayout(self.__device, layoutInfo, None)

    def __createGraphicsPipeline(self):
        vertShaderModule = self.__createShaderModule('shaders/vert5.spv')
        fragShaderModule = self.__createShaderModule('shaders/frag4.spv')

        vertShaderStageInfo = VkPipelineShaderStageCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage=VK_SHADER_STAGE_VERTEX_BIT,
            module=vertShaderModule,
            pName='main'
        )

        fragShaderStageInfo = VkPipelineShaderStageCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage=VK_SHADER_STAGE_FRAGMENT_BIT,
            module=fragShaderModule,
            pName='main'
        )

        shaderStages = [vertShaderStageInfo, fragShaderStageInfo]

        bindingDescription = Vertex.getBindingDescription()
        attributeDescriptions = Vertex.getAttributeDescriptions()
        vertexInputInfo = VkPipelineVertexInputStateCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            vertexBindingDescriptionCount=1,
            vertexAttributeDescriptionCount=len(attributeDescriptions),
            pVertexBindingDescriptions=bindingDescription,
            pVertexAttributeDescriptions=attributeDescriptions
        )

        inputAssembly = VkPipelineInputAssemblyStateCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            topology=VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            primitiveRestartEnable=False
        )

        viewport = VkViewport(0.0, 0.0,
                              float(self.__swapChainExtent.width),
                              float(self.__swapChainExtent.height),
                              0.0, 1.0)
        scissor = VkRect2D([0, 0], self.__swapChainExtent)
        viewportState = VkPipelineViewportStateCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            viewportCount=1,
            pViewports=viewport,
            scissorCount=1,
            pScissors=scissor
        )

        rasterizer = VkPipelineRasterizationStateCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            depthClampEnable=False,
            rasterizerDiscardEnable=False,
            polygonMode=VK_POLYGON_MODE_FILL,
            lineWidth=1.0,
            cullMode=VK_CULL_MODE_BACK_BIT,
            frontFace=VK_FRONT_FACE_COUNTER_CLOCKWISE,
            depthBiasEnable=False
        )

        multisampling = VkPipelineMultisampleStateCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            sampleShadingEnable=False,
            rasterizationSamples=VK_SAMPLE_COUNT_1_BIT
        )

        depthStencil = VkPipelineDepthStencilStateCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            depthTestEnable=VK_TRUE,
            depthWriteEnable=VK_TRUE,
            depthCompareOp=VK_COMPARE_OP_LESS,
            depthBoundsTestEnable=VK_FALSE,
            stencilTestEnable=VK_FALSE
        )

        colorBlendAttachment = VkPipelineColorBlendAttachmentState(
            colorWriteMask=VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
            blendEnable=False
        )

        colorBlending = VkPipelineColorBlendStateCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            logicOpEnable=False,
            logicOp=VK_LOGIC_OP_COPY,
            attachmentCount=1,
            pAttachments=colorBlendAttachment,
            blendConstants=[0.0, 0.0, 0.0, 0.0]
        )

        setLayouts = [self.__descriptorSetLayout]
        pipelineLayoutInfo = VkPipelineLayoutCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            setLayoutCount=1,
            pSetLayouts=setLayouts,
        )

        self.__pipelineLayout = vkCreatePipelineLayout(self.__device, pipelineLayoutInfo, None)

        pipelineInfo = VkGraphicsPipelineCreateInfo(
            sType=VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            stageCount=2,
            pStages=shaderStages,
            pVertexInputState=vertexInputInfo,
            pInputAssemblyState=inputAssembly,
            pViewportState=viewportState,
            pRasterizationState=rasterizer,
            pMultisampleState=multisampling,
            pDepthStencilState=depthStencil,
            pColorBlendState=colorBlending,
            layout=self.__pipelineLayout,
            renderPass=self.__renderPass,
            subpass=0,
            basePipelineHandle=VK_NULL_HANDLE
        )

        self.__graphicsPipeline = vkCreateGraphicsPipelines(self.__device, VK_NULL_HANDLE, 1, pipelineInfo, None)

        vkDestroyShaderModule(self.__device, vertShaderModule, None)
        vkDestroyShaderModule(self.__device, fragShaderModule, None)

    def __createFramebuffers(self):
        self.__swapChainFramebuffers = []

        for imageView in self.__swapChainImageViews:
            attachments = [imageView, self.__depthImageView]

            framebufferInfo = VkFramebufferCreateInfo(
                sType=VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                renderPass=self.__renderPass,
                attachmentCount=len(attachments),
                pAttachments=attachments,
                width=self.__swapChainExtent.width,
                height=self.__swapChainExtent.height,
                layers=1
            )
            framebuffer = vkCreateFramebuffer(self.__device, framebufferInfo, None)
            self.__swapChainFramebuffers.append(framebuffer)

    def __createCommandPool(self):
        queueFamilyIndices = self.__findQueueFamilies(self.__physicalDevice)

        poolInfo = VkCommandPoolCreateInfo(
            sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            queueFamilyIndex=queueFamilyIndices.graphicsFamily
        )

        self.__commandPool = vkCreateCommandPool(self.__device, poolInfo, None)

    def __createDepthResources(self):
        depthFormat = self.__findDepthFormat()

        self.__depthImage, self.__depthImageMemory = self.__createImage(self.__swapChainExtent.width,
                                                                        self.__swapChainExtent.height,
                                                                        depthFormat,
                                                                        VK_IMAGE_TILING_OPTIMAL,
                                                                        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                                                                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
        self.__depthImageView = self.__createImageView(self.__depthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT)

        self.__transitionImageLayout(self.__depthImage, depthFormat, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)

    def __findSupportedFormat(self, candidates, tiling, features):
        for i in candidates:
            props = vkGetPhysicalDeviceFormatProperties(self.__physicalDevice, i)

            if tiling == VK_IMAGE_TILING_LINEAR and ((props.linearTilingFeatures & features) == features):
                return i
            elif tiling == VK_IMAGE_TILING_OPTIMAL and ((props.optimalTilingFeatures & features) == features):
                return i

        raise Exception("failed to find supported format!")

    def __findDepthFormat(self):
        return self.__findSupportedFormat((VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT),
                                          VK_IMAGE_TILING_OPTIMAL,
                                          VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)

    def __hasStencilComponent(self, f):
        return f == VK_FORMAT_D32_SFLOAT_S8_UINT or f == VK_FORMAT_D24_UNORM_S8_UINT

    def __createTextureImage(self):
        im = Image.open(TEXTURE_PATH)
        im.putalpha(1)
        imageSize = im.width * im.height * 4

        stagingBuffer, stagingBufferMemory = self.__createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)

        data = vkMapMemory(self.__device, stagingBufferMemory, 0, imageSize, 0)
        ffi.memmove(data, im.tobytes(), imageSize)
        vkUnmapMemory(self.__device, stagingBufferMemory)

        self.__textureImage, self.__textureImageMemory = self.__createImage(im.width, im.height,
                                                                            VK_FORMAT_R8G8B8A8_UNORM,
                                                                            VK_IMAGE_TILING_OPTIMAL,
                                                                            VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                                                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)

        self.__transitionImageLayout(self.__textureImage, VK_FORMAT_R8G8B8A8_UNORM,
                                     VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
        self.__copyBufferToImage(stagingBuffer, self.__textureImage, im.width, im.height)

        self.__transitionImageLayout(self.__textureImage, VK_FORMAT_R8G8B8A8_UNORM,
                                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)

        vkFreeMemory(self.__device, stagingBufferMemory, None)
        vkDestroyBuffer(self.__device, stagingBuffer, None)

    def __createTextureImageView(self):
        self.__textureImageView = self.__createImageView(self.__textureImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT)

    def __createTextureSampler(self):
        samplerInfo = VkSamplerCreateInfo(
            sType=VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            magFilter=VK_FILTER_LINEAR,
            minFilter=VK_FILTER_LINEAR,
            addressModeU=VK_SAMPLER_ADDRESS_MODE_REPEAT,
            addressModeV=VK_SAMPLER_ADDRESS_MODE_REPEAT,
            addressModeW=VK_SAMPLER_ADDRESS_MODE_REPEAT,
            anisotropyEnable=VK_TRUE,
            maxAnisotropy=16,
            borderColor=VK_BORDER_COLOR_INT_OPAQUE_BLACK,
            unnormalizedCoordinates=VK_FALSE,
            compareEnable=VK_FALSE,
            compareOp=VK_COMPARE_OP_ALWAYS,
            mipmapMode=VK_SAMPLER_MIPMAP_MODE_LINEAR
        )

        self.__textureSampler = vkCreateSampler(self.__device, samplerInfo, None)

    def __createImageView(self, image, im_format, aspectFlags):
        viewInfo = VkImageViewCreateInfo(
            sType=VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            image=image,
            viewType=VK_IMAGE_VIEW_TYPE_2D,
            format=im_format,
            subresourceRange=[aspectFlags, 0, 1, 0, 1]
        )

        imageView = vkCreateImageView(self.__device, viewInfo, None)

        return imageView

    def __createImage(self, width, height, im_format, tiling, usage, properties):
        imageInfo = VkImageCreateInfo(
            sType=VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            imageType=VK_IMAGE_TYPE_2D,
            extent=VkExtent3D(width, height, 1),
            mipLevels=1,
            arrayLayers=1,
            format=im_format,
            tiling=tiling,
            initialLayout=VK_IMAGE_LAYOUT_PREINITIALIZED,
            usage=usage,
            samples=VK_SAMPLE_COUNT_1_BIT,
            sharingMode=VK_SHARING_MODE_EXCLUSIVE
        )

        image = vkCreateImage(self.__device, imageInfo, None)

        memRequirements = vkGetImageMemoryRequirements(self.__device, image)

        allocInfo = VkMemoryAllocateInfo(
            sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize=memRequirements.size,
            memoryTypeIndex=self.__findMemoryType(memRequirements.memoryTypeBits, properties)
        )
        imageMemory = vkAllocateMemory(self.__device, allocInfo, None)
        vkBindImageMemory(self.__device, image, imageMemory, 0)
        return (image, imageMemory)

    def __transitionImageLayout(self, image, im_format, oldLayout, newLayout):
        commandBuffer = self.__beginSingleTimeCommands()

        familyIndiex = ffi.cast('uint32_t', VK_QUEUE_FAMILY_IGNORED)
        barrier = VkImageMemoryBarrier(
            sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            oldLayout=oldLayout,
            newLayout=newLayout,
            srcQueueFamilyIndex=familyIndiex,
            dstQueueFamilyIndex=familyIndiex,
            image=image,
        )

        subresourceRange = VkImageSubresourceRange(
            aspectMask=VK_IMAGE_ASPECT_COLOR_BIT,
            baseMipLevel=0,
            levelCount=1,
            baseArrayLayer=0,
            layerCount=1
        )
        if newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
            subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT

            if self.__hasStencilComponent(im_format):
                subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT
        # else:
        #     subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT
        barrier.subresourceRange = subresourceRange

        sourceStage = -1
        destinationStage = -1
        if oldLayout == VK_IMAGE_LAYOUT_UNDEFINED and newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
            barrier.srcAccessMask = 0
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT

            sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
            destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT
        elif oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL and newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT

            sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT
            destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
        elif oldLayout == VK_IMAGE_LAYOUT_UNDEFINED and newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
            barrier.srcAccessMask = 0
            barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT

            sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
            destinationStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT
        else:
            raise Exception("unsupported layout transition!")

        vkCmdPipelineBarrier(
            commandBuffer,
            sourceStage,
            destinationStage,
            0,
            0, None,
            0, None,
            1, [barrier]
        )

        self.__endSingleTimeCommands(commandBuffer)

    def __copyBufferToImage(self, buffer, image, width, height):
        commandBuffer = self.__beginSingleTimeCommands()

        region = VkBufferImageCopy(
            bufferOffset=0,
            bufferRowLength=0,
            bufferImageHeight=0,
            imageSubresource=[VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1],
            imageOffset=[0, 0, 0],
            imageExtent=[width, height, 1]
        )

        vkCmdCopyBufferToImage(commandBuffer, buffer, image,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                               1, [region])

        self.__endSingleTimeCommands(commandBuffer)

    def __loadModel(self):
        # st = time.time()
        model = tol.LoadObj(MODEL_PATH)
        attrib = model['attribs']
        vertices = attrib['vertices']
        texcoords = attrib['texcoords']
        shapes = model['shapes']

        uniqueVertices = {}
        vertexData = []
        indexData = []
        for shape in shapes:
            allIndices = np.array(shapes[shape]['indices'])
            for idx in range(0, len(allIndices), 3):
                vid = allIndices[idx]
                tid = allIndices[idx+2]
                data = (
                    # vertex pos
                    vertices[3 * vid + 0],
                    vertices[3 * vid + 1],
                    vertices[3 * vid + 2],
                    # color
                    1.0, 1.0, 1.0,
                    # texture coord
                    texcoords[2 * tid + 0],
                    1.0 - texcoords[2 * tid + 1]
                )

                if not data in uniqueVertices:
                    uniqueVertices[data] = len(vertexData)
                    vertexData.append(data)
                indexData.append(uniqueVertices[data])

        self.vertices = np.array(vertexData, np.float32)
        self.indices = np.array(indexData, np.uint32)
        # print('load model time: {} seconds'.format(time.time() - st))

    def __createVertexBuffer(self):
        bufferSize = self.vertices.nbytes
        vertex_ptr = ffi.cast('float *', self.vertices.ctypes.data)

        stagingBuffer, stagingBufferMemory = self.__createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)

        data = vkMapMemory(self.__device, stagingBufferMemory, 0, bufferSize, 0)
        ffi.memmove(data, vertex_ptr, bufferSize)
        vkUnmapMemory(self.__device, stagingBufferMemory)

        self.__vertexBuffer, self.__vertexBufferMemory = self.__createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                                                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)

        self.__copyBuffer(stagingBuffer, self.__vertexBuffer, bufferSize)

        vkFreeMemory(self.__device, stagingBufferMemory, None)
        vkDestroyBuffer(self.__device, stagingBuffer, None)

    def __createIndexBuffer(self):
        bufferSize = self.indices.nbytes
        indices_ptr = ffi.cast('uint16_t*', self.indices.ctypes.data)

        stagingBuffer, stagingBufferMemory = self.__createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)

        data = vkMapMemory(self.__device, stagingBufferMemory, 0, bufferSize, 0)
        ffi.memmove(data, indices_ptr, bufferSize)
        vkUnmapMemory(self.__device, stagingBufferMemory)

        self.__indexBuffer, self.__indexBufferMemory = self.__createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                                                                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)

        self.__copyBuffer(stagingBuffer, self.__indexBuffer, bufferSize)

        vkFreeMemory(self.__device, stagingBufferMemory, None)
        vkDestroyBuffer(self.__device, stagingBuffer, None)

    def __createUniformBuffer(self):
        uniformBufObj = UniformBufferObject(np.identity(4, np.float32), np.identity(4, np.float32),
                                            np.identity(4, np.float32))
        bufferSize = uniformBufObj.nbytes

        self.__uniformBuffer, self.__uniformBufferMemory = self.__createBuffer(bufferSize,
                                                                               VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                                                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)

    def __createDescriptorPool(self):
        poolSize1 = VkDescriptorPoolSize(
            type=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            descriptorCount=1
        )
        poolSize2 = VkDescriptorPoolSize(
            type=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            descriptorCount=1
        )
        poolSizes = [poolSize1, poolSize2]

        poolInfo = VkDescriptorPoolCreateInfo(
            sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            poolSizeCount=len(poolSizes),
            pPoolSizes=poolSizes,
            maxSets=1
        )

        self.__descriptorPool = vkCreateDescriptorPool(self.__device, poolInfo, None)

    def __createDescriptorSet(self):
        layouts = [self.__descriptorSetLayout]
        allocInfo = VkDescriptorSetAllocateInfo(
            sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            descriptorPool=self.__descriptorPool,
            descriptorSetCount=1,
            pSetLayouts=layouts
        )

        descriptorSets = vkAllocateDescriptorSets(self.__device, allocInfo)
        self.__descriptorSet = descriptorSets[0]

        ubo = UniformBufferObject(np.identity(4, np.float32), np.identity(4, np.float32), np.identity(4, np.float32))
        bufferInfo = VkDescriptorBufferInfo(
            buffer=self.__uniformBuffer,
            offset=0,
            range=ubo.nbytes
        )

        imageInfo = VkDescriptorImageInfo(
            imageLayout=VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            imageView=self.__textureImageView,
            sampler=self.__textureSampler
        )

        descriptorWrite1 = VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=self.__descriptorSet,
            dstBinding=0,
            dstArrayElement=0,
            descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            descriptorCount=1,
            pBufferInfo=bufferInfo
        )
        descriptorWrite2 = VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=self.__descriptorSet,
            dstBinding=1,
            dstArrayElement=0,
            descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            descriptorCount=1,
            pImageInfo=imageInfo
        )

        descriptorWrites = [descriptorWrite1, descriptorWrite2]
        vkUpdateDescriptorSets(self.__device, len(descriptorWrites), descriptorWrites, 0, None)

    def __createBuffer(self, size, usage, properties):
        buf = None
        bufMemory = None

        bufferInfo = VkBufferCreateInfo(
            sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            size=size,
            usage=usage,
            sharingMode=VK_SHARING_MODE_EXCLUSIVE
        )
        buf = vkCreateBuffer(self.__device, bufferInfo, None)

        memRequirements = vkGetBufferMemoryRequirements(self.__device, buf)

        allocInfo = VkMemoryAllocateInfo(
            sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize=memRequirements.size,
            memoryTypeIndex=self.__findMemoryType(memRequirements.memoryTypeBits, properties)
        )

        bufMemory = vkAllocateMemory(self.__device, allocInfo, None)

        vkBindBufferMemory(self.__device, buf, bufMemory, 0)

        return (buf, bufMemory)

    def __beginSingleTimeCommands(self):
        allocInfo = VkCommandBufferAllocateInfo(
            sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandPool=self.__commandPool,
            commandBufferCount=1
        )

        commandBuffers = vkAllocateCommandBuffers(self.__device, allocInfo)

        beginInfo = VkCommandBufferBeginInfo(
            sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
        )

        vkBeginCommandBuffer(commandBuffers[0], beginInfo)

        return commandBuffers[0]

    def __endSingleTimeCommands(self, commandBuffer):
        vkEndCommandBuffer(commandBuffer)

        commandBuffers = [commandBuffer]
        submitInfo = VkSubmitInfo(
            sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,
            commandBufferCount=1,
            pCommandBuffers=commandBuffers
        )

        vkQueueSubmit(self.__graphicsQueue, 1, submitInfo, VK_NULL_HANDLE)
        vkQueueWaitIdle(self.__graphicsQueue)

        vkFreeCommandBuffers(self.__device, self.__commandPool, 1, commandBuffers)

    def __copyBuffer(self, srcBuffer, dstBuffer, size):
        commandBuffer = self.__beginSingleTimeCommands()

        copyRegion = VkBufferCopy(size=size)
        vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, ffi.addressof(copyRegion))

        self.__endSingleTimeCommands(commandBuffer)

    def __findMemoryType(self, typeFilter, properties):
        memProperties = vkGetPhysicalDeviceMemoryProperties(self.__physicalDevice)

        for i, memType in enumerate(memProperties.memoryTypes):
            if (typeFilter & (1 << i)) and (memType.propertyFlags & properties == properties):
                return i

        raise Exception("failed to find suitable memory type!")

    def __createCommandBuffers(self):
        allocInfo = VkCommandBufferAllocateInfo(
            sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            commandPool=self.__commandPool,
            level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=len(self.__swapChainFramebuffers)
        )

        self.__commandBuffers = vkAllocateCommandBuffers(self.__device, allocInfo)

        for i, cmdBuffer in enumerate(self.__commandBuffers):
            beginInfo = VkCommandBufferBeginInfo(
                sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                flags=VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT
            )

            vkBeginCommandBuffer(cmdBuffer, beginInfo)

            clearValues = [VkClearValue(), VkClearValue()]
            clearValues[0].color = [[0.0, 0.0, 0.0, 1.0]]
            clearValues[1].depthStencil = [1.0, 0]
            renderPassInfo = VkRenderPassBeginInfo(
                sType=VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                renderPass=self.__renderPass,
                framebuffer=self.__swapChainFramebuffers[i],
                renderArea=[[0, 0], self.__swapChainExtent],
                clearValueCount=len(clearValues),
                pClearValues=clearValues
            )

            vkCmdBeginRenderPass(cmdBuffer, renderPassInfo, VK_SUBPASS_CONTENTS_INLINE)

            vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, self.__graphicsPipeline)

            vkCmdBindVertexBuffers(cmdBuffer, 0, 1, [self.__vertexBuffer, ], [0, ])

            vkCmdBindIndexBuffer(cmdBuffer, self.__indexBuffer, 0, VK_INDEX_TYPE_UINT32)

            vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, self.__pipelineLayout,
                                    0, 1, [self.__descriptorSet], 0, None)

            vkCmdDrawIndexed(cmdBuffer, len(self.indices), 1, 0, 0, 0)

            vkCmdEndRenderPass(cmdBuffer)

            vkEndCommandBuffer(cmdBuffer)

    def __createSemaphores(self):
        semaphoreInfo = VkSemaphoreCreateInfo(sType=VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO)

        self.__imageAvailableSemaphore = vkCreateSemaphore(self.__device, semaphoreInfo, None)
        self.__renderFinishedSemaphore = vkCreateSemaphore(self.__device, semaphoreInfo, None)

    def __updateUniformBuffer(self):
        end = time.time()

        t = (end - self.__startTime)

        ubo = UniformBufferObject()
        ubo.model = glm.rotate(np.identity(4, np.float32), t * 90.0, 0.0, 0.0, 1.0)
        ubo.view = glm.lookAt(np.array([2, 2, 2], np.float32), np.array([0, 0, 0], np.float32),
                              np.array([0, 0, 1], np.float32))
        ubo.proj = glm.perspective(45, self.__swapChainExtent.width / float(self.__swapChainExtent.height), 0.1, 10.0)
        ubo.proj[1][1] *= -1

        data = vkMapMemory(self.__device, self.__uniformBufferMemory, 0, ubo.nbytes, 0)
        ffi.memmove(data, ubo.to_c_ptr, ubo.nbytes)
        vkUnmapMemory(self.__device, self.__uniformBufferMemory)

    def __drawFrame(self):
        vkAcquireNextImageKHR = vkGetDeviceProcAddr(self.__device, 'vkAcquireNextImageKHR')
        vkQueuePresentKHR = vkGetDeviceProcAddr(self.__device, 'vkQueuePresentKHR')

        try:
            imageIndex = vkAcquireNextImageKHR(self.__device, self.__swapChain, 18446744073709551615,
                                               self.__imageAvailableSemaphore, VK_NULL_HANDLE)
        except VkErrorOutOfDateKhr:
            self.recreateSwapChain()
            return

        waitSemaphore = [self.__imageAvailableSemaphore]
        signalSemaphore = [self.__renderFinishedSemaphore]
        submitInfo = VkSubmitInfo(
            sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,
            waitSemaphoreCount=len(waitSemaphore),
            pWaitSemaphores=waitSemaphore,
            pWaitDstStageMask=[VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, ],
            commandBufferCount=1,
            pCommandBuffers=[self.__commandBuffers[imageIndex], ],
            signalSemaphoreCount=len(signalSemaphore),
            pSignalSemaphores=signalSemaphore
        )

        vkQueueSubmit(self.__graphicsQueue, 1, submitInfo, VK_NULL_HANDLE)

        swapChains = [self.__swapChain]
        presentInfo = VkPresentInfoKHR(
            sType=VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            waitSemaphoreCount=len(signalSemaphore),
            pWaitSemaphores=signalSemaphore,
            swapchainCount=1,
            pSwapchains=swapChains,
            pImageIndices=[imageIndex]
        )

        try:
            vkQueuePresentKHR(self.__presentQueue, presentInfo)
        except VkErrorOutOfDateKhr:
            self.recreateSwapChain()

        vkQueueWaitIdle(self.__presentQueue)

    def __createShaderModule(self, shaderFile):
        with open(shaderFile, 'rb') as sf:
            code = sf.read()

            createInfo = VkShaderModuleCreateInfo(
                sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                codeSize=len(code),
                pCode=code)

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
        width = max(capabilities.minImageExtent.width, min(capabilities.maxImageExtent.width, self.width()))
        height = max(capabilities.minImageExtent.height, min(capabilities.maxImageExtent.height, self.height()))
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
            swapChainAdequate = (swapChainSupport.formats is not None) and (
                swapChainSupport.presentModes is not None)
        return indices.isComplete() and extensionsSupported and swapChainAdequate

    def __checkDeviceExtensionSupport(self, device):
        availableExtensions = vkEnumerateDeviceExtensionProperties(device, None)

        for extension in availableExtensions:
            if extension.extensionName in deviceExtensions:
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

        self.__timer.start()

        super(HelloTriangleApplication, self).show()

    def resizeEvent(self, event):
        # only recreate swapChain when window size got changed
        if event.size() != event.oldSize():
            self.__timer.stop()
            self.recreateSwapChain()
            self.__timer.start()

        super(HelloTriangleApplication, self).resizeEvent(event)

if __name__ == '__main__':
    app = QtGui.QGuiApplication(sys.argv)

    win = HelloTriangleApplication()
    win.show()

    def clenaup():
        global win
        del win

    app.aboutToQuit.connect(clenaup)

    sys.exit(app.exec_())

