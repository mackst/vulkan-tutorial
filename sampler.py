import math

from pyVulkan import *
from PIL import Image

import PyGlfwCffi as glfw
import numpy as np

import glm


WIDTH = 800
HEIGHT = 600

validationLayers = ["VK_LAYER_LUNARG_standard_validation"]
deviceExtensions = [VK_KHR_SWAPCHAIN_EXTENSION_NAME]

enableValidationLayers = True


@vkDebugReportCallbackEXT
def debugCallback(*args):
    print (ffi.string(args[6]))
    return True

@glfw.window_size_callback
def onWindowResized(window, width, height):
    if width == 0 and height == 0:
        return

    app = HelloTriangleApplication.INSTANCE
    app.recreateSwapChain()

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
    POS = np.array([0, 0], dtype=np.float32)
    COLOR = np.array([0, 0, 0], dtype=np.float32)

    @staticmethod
    def getBindingDescription():
        bindingDescription = VkVertexInputBindingDescription(
            binding=0,
            stride=Vertex.POS.nbytes+Vertex.COLOR.nbytes,
            inputRate=VK_VERTEX_INPUT_RATE_VERTEX
        )
        return bindingDescription

    @staticmethod
    def getAttributeDescriptions():
        attributeDescription1 = VkVertexInputAttributeDescription(
            binding=0,
            location=0,
            format=VK_FORMAT_R32G32_SFLOAT,
            offset=0
        )
        attributeDescription2 = VkVertexInputAttributeDescription(
            binding=0,
            location=1,
            format=VK_FORMAT_R32G32B32_SFLOAT,
            offset=Vertex.POS.nbytes
        )

        return ffi.new('VkVertexInputAttributeDescription[]', [attributeDescription1, attributeDescription2])

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

class HelloTriangleApplication(object):

    INSTANCE = None

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
        self.__descriptorSetLayout = None
        self.__pipelineLayout = None
        self.__graphicsPipeline = None

        self.__commandPool = None

        self.__textureImage = None
        self.__textureImageMemory = None
        self.__textureImageView = None
        self.__textureSampler = None

        self.__vertexBuffer = None
        self.__vertexBufferMemory = None
        self.__indexBuffer = None
        self.__indexBufferMemory = None

        self.__uniformStagingBuffer = None
        self.__uniformStagingBufferMemory = None
        self.__uniformBuffer = None
        self.__uniformBufferMemory = None

        self.__descriptorPool = None
        self.__descriptorSet = None

        self.__commandBuffers = None

        self.__imageAvailableSemaphore = None
        self.__renderFinishedSemaphore = None

        self.vertices = np.array([
            -0.5, -0.5, 1.0, 0.0, 0.0,
            0.5, -0.5, 0.0, 1.0, 0.0,
            0.5, 0.5, 0.0, 0.0, 1.0,
            -0.5, 0.5, 1.0, 1.0, 1.0
        ], np.float32)

        self.indices = np.array([0, 1, 2, 2, 3, 0], np.uint16)

    def __del__(self):
        vkDeviceWaitIdle(self.__device)

        if self.__imageAvailableSemaphore:
            vkDestroySemaphore(self.__device, self.__imageAvailableSemaphore, None)

        if self.__renderFinishedSemaphore:
            vkDestroySemaphore(self.__device, self.__renderFinishedSemaphore, None)

        if self.__textureImageView:
            vkDestroyImageView(self.__device, self.__textureImageView, None)

        if self.__textureSampler:
            vkDestroySampler(self.__device, self.__textureSampler, None)

        if self.__textureImageMemory:
            vkFreeMemory(self.__device, self.__textureImageMemory, None)

        if self.__textureImage:
            vkDestroyImage(self.__device, self.__textureImage, None)

        if self.__indexBufferMemory:
            vkFreeMemory(self.__device, self.__indexBufferMemory, None)

        if self.__indexBuffer:
            vkDestroyBuffer(self.__device, self.__indexBuffer, None)

        if self.__vertexBufferMemory:
            vkFreeMemory(self.__device, self.__vertexBufferMemory, None)

        if self.__vertexBuffer:
            vkDestroyBuffer(self.__device, self.__vertexBuffer, None)

        if self.__commandBuffers:
            self.__commandBuffers = None

        if self.__commandPool:
            vkDestroyCommandPool(self.__device, self.__commandPool, None)

        if self.__descriptorPool:
            vkDestroyDescriptorPool(self.__device, self.__descriptorPool, None)

        if self.__swapChainFramebuffers:
            for i in self.__swapChainFramebuffers:
                vkDestroyFramebuffer(self.__device, i, None)
            self.__swapChainFramebuffers = None

        if self.__renderPass:
            vkDestroyRenderPass(self.__device, self.__renderPass, None)

        if self.__descriptorSetLayout:
            vkDestroyDescriptorSetLayout(self.__device, self.__descriptorSetLayout, None)

        if self.__pipelineLayout:
            vkDestroyPipelineLayout(self.__device, self.__pipelineLayout, ffi.NULL)

        if self.__graphicsPipeline:
            vkDestroyPipeline(self.__device, self.__graphicsPipeline, None)

        if self.__swapChainImageViews:
            for i in self.__swapChainImageViews:
                vkDestroyImageView(self.__device, i, None)

        if self.__swapChain:
            destroySwapChain(self.__device, self.__swapChain, None)

        if self.__device:
            vkDestroyDevice(self.__device, None)

        if self.__surface:
            destroySurface(self.__instance, self.__surface, None)

        if self.__callback:
            destroyDebugReportCallbackEXT(self.__instance, self.__callback, None)

        if self.__instance:
            vkDestroyInstance(self.__instance, None)

    def __initWindow(self):
        glfw.init()

        glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)

        self.__window = glfw.create_window(WIDTH, HEIGHT, "Vulkan")

        HelloTriangleApplication.INSTANCE = self
        glfw.set_window_size_callback(self.__window, onWindowResized)

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
        self.__createFramebuffers()
        self.__createCommandPool()
        self.__createTextureImage()
        self.__createTextureImageView()
        self.__createTextureSampler()
        self.__createVertexBuffer()
        self.__createIndexBuffer()
        self.__createUniformBuffer()
        self.__createDescriptorPool()
        self.__createDescriptorSet()
        self.__createCommandBuffers()
        self.__createSemaphores()

    def __mainLoop(self):
        while not glfw.window_should_close(self.__window):
            glfw.poll_events()

            self.__updateUniformBuffer()
            self.__drawFrame()

        vkDeviceWaitIdle(self.__device)

    def recreateSwapChain(self):
        vkDeviceWaitIdle(self.__device)

        self.__createSwapChain()
        self.__createImageViews()
        self.__createRenderPass()
        self.__createGraphicsPipeline()
        self.__createFramebuffers()
        self.__createCommandBuffers()

    def __createInstance(self):
        if enableValidationLayers and not self.__checkValidationLayerSupport():
            raise Exception("validation layers requested, but not available!")

        appInfo = VkApplicationInfo(
            pApplicationName='Hello Triangle',
            applicationVersion=VK_MAKE_VERSION(1, 0, 0),
            pEngineName='No Engine',
            engineVersion=VK_MAKE_VERSION(1, 0, 0),
            apiVersion=VK_MAKE_VERSION(1, 0, 3)
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

        deviceFeatures = VkPhysicalDeviceFeatures()
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

        oldSwapchain = None
        if self.__swapChain:
            oldSwapchain = self.__swapChain
            createInfo.oldSwapchain = oldSwapchain

        vkCreateSwapchainKHR = vkGetDeviceProcAddr(self.__device, 'vkCreateSwapchainKHR')
        self.__swapChain = vkCreateSwapchainKHR(self.__device, createInfo, None)

        if oldSwapchain:
            destroySwapChain(self.__device, oldSwapchain, None)

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

    def __createDescriptorSetLayout(self):
        uboLayoutBinding = VkDescriptorSetLayoutBinding(
            binding=0,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            stageFlags=VK_SHADER_STAGE_VERTEX_BIT
        )

        layoutInfo = VkDescriptorSetLayoutCreateInfo(
            bindingCount=1,
            pBindings=uboLayoutBinding
        )

        self.__descriptorSetLayout = vkCreateDescriptorSetLayout(self.__device, layoutInfo, None)

    def __createGraphicsPipeline(self):
        vertShaderModule = self.__createShaderModule('shaders/vert3.spv')
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

        bindingDescription = Vertex.getBindingDescription()
        attributeDescriptions = Vertex.getAttributeDescriptions()
        vertexInputInfo = VkPipelineVertexInputStateCreateInfo(
            vertexBindingDescriptionCount=1,
            vertexAttributeDescriptionCount=len(attributeDescriptions),
            pVertexBindingDescriptions=bindingDescription,
            pVertexAttributeDescriptions=attributeDescriptions
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
            frontFace=VK_FRONT_FACE_COUNTER_CLOCKWISE,
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

        setLayouts = ffi.new('VkDescriptorSetLayout[]', [self.__descriptorSetLayout])
        pipelineLayoutInfo = VkPipelineLayoutCreateInfo(
            setLayoutCount=1,
            pSetLayouts=setLayouts,
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

    def __createCommandPool(self):
        queueFamilyIndices = self.__findQueueFamilies(self.__physicalDevice)

        poolInfo = VkCommandPoolCreateInfo(
            queueFamilyIndex=queueFamilyIndices.graphicsFamily
        )

        self.__commandPool = vkCreateCommandPool(self.__device, poolInfo, None)

    def __createTextureImage(self):
        im = Image.open('textures/texture.jpg')
        im.putalpha(1)
        imageSize = im.width * im.height * 4

        stagingImage, stagingImageMemory = self.__createImage(im.width, im.height,
                                                              VK_FORMAT_R8G8B8A8_UNORM,
                                                              VK_IMAGE_TILING_LINEAR,
                                                              VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                                                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)

        data = vkMapMemory(self.__device, stagingImageMemory, 0, imageSize, 0)
        ffi.memmove(data, im.tobytes(), imageSize)
        vkUnmapMemory(self.__device, stagingImageMemory)

        self.__textureImage, self.__textureImageMemory = self.__createImage(im.width, im.height,
                                                                            VK_FORMAT_R8G8B8A8_UNORM,
                                                                            VK_IMAGE_TILING_OPTIMAL,
                                                                            VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                                                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
        self.__transitionImageLayout(stagingImage, VK_FORMAT_R8G8B8A8_UNORM,
                                     VK_IMAGE_LAYOUT_PREINITIALIZED, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
        self.__transitionImageLayout(self.__textureImage, VK_FORMAT_R8G8B8A8_UNORM,
                                     VK_IMAGE_LAYOUT_PREINITIALIZED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
        self.__copyImage(stagingImage, self.__textureImage, im.width, im.height)

        self.__transitionImageLayout(self.__textureImage, VK_FORMAT_R8G8B8A8_UNORM,
                                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)

        vkFreeMemory(self.__device, stagingImageMemory, None)
        vkDestroyImage(self.__device, stagingImage, None)

    def __createTextureImageView(self):
        self.__textureImageView = self.__createImageView(self.__textureImage, VK_FORMAT_R8G8B8A8_UNORM)

    def __createTextureSampler(self):
        samplerInfo = VkSamplerCreateInfo(
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

    def __createImageView(self, image, im_format):
        viewInfo = VkImageViewCreateInfo(
            image=image,
            viewType=VK_IMAGE_VIEW_TYPE_2D,
            format=im_format,
            subresourceRange=[VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1]
        )

        imageView = vkCreateImageView(self.__device, viewInfo, None)

        return imageView

    def __createImage(self, width, height, im_format, tiling, usage, properties):
        imageInfo = VkImageCreateInfo(
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
            oldLayout=oldLayout,
            newLayout=newLayout,
            srcQueueFamilyIndex=familyIndiex,
            dstQueueFamilyIndex=familyIndiex,
            image=image,
            subresourceRange=[VK_IMAGE_ASPECT_COLOR_BIT,
                              0, 1, 0, 1]
        )

        if oldLayout == VK_IMAGE_LAYOUT_PREINITIALIZED and newLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
            barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT
        elif oldLayout == VK_IMAGE_LAYOUT_PREINITIALIZED and newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
            barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT
        elif oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL and newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT
        else:
            raise Exception("unsupported layout transition!")

        vkCmdPipelineBarrier(
            commandBuffer,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            0,
            0, ffi.NULL,
            0, ffi.NULL,
            1, barrier
        )

        self.__endSingleTimeCommands(commandBuffer)

    def __copyImage(self, srcImage, dstImage, width, height):
        commandBuffer = self.__beginSingleTimeCommands()

        subResource = VkImageSubresourceLayers(
            aspectMask=VK_IMAGE_ASPECT_COLOR_BIT,
            baseArrayLayer=0,
            mipLevel=0,
            layerCount=1
        )

        region = VkImageCopy(
            srcSubresource=subResource,
            dstSubresource=subResource,
            srcOffset=[0, 0, 0],
            dstOffset=[0, 0, 0],
            extent=VkExtent3D(width, height, 1)
        )

        vkCmdCopyImage(
            commandBuffer,
            srcImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            dstImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1, region
        )

        self.__endSingleTimeCommands(commandBuffer)

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
        uniformBufObj = UniformBufferObject(np.identity(4, np.float32), np.identity(4, np.float32), np.identity(4, np.float32))
        bufferSize = uniformBufObj.nbytes

        self.__uniformStagingBuffer, self.__uniformStagingBufferMemory = self.__createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                                                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
        self.__uniformBuffer, self.__uniformBufferMemory = self.__createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                                                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)

    def __createDescriptorPool(self):
        poolSize = VkDescriptorPoolSize(
            type=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            descriptorCount=1
        )

        poolInfo = VkDescriptorPoolCreateInfo(
            poolSizeCount=1,
            pPoolSizes=poolSize,
            maxSets=1
        )

        self.__descriptorPool = vkCreateDescriptorPool(self.__device, poolInfo, None)

    def __createDescriptorSet(self):
        layouts = [self.__descriptorSetLayout]
        # layouts = ffi.new('VkDescriptorSetLayout[]', [self.__descriptorSetLayout])
        allocInfo = VkDescriptorSetAllocateInfo(
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

        descriptorWrite = VkWriteDescriptorSet(
            dstSet=self.__descriptorSet,
            dstBinding=0,
            dstArrayElement=0,
            descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            descriptorCount=1,
            pBufferInfo=bufferInfo
        )

        vkUpdateDescriptorSets(self.__device, 1, [descriptorWrite], 0, ffi.NULL)

    def __createBuffer(self, size, usage, properties):
        buf = None
        bufMemory = None

        bufferInfo = VkBufferCreateInfo(
            size=size,
            usage=usage,
            sharingMode=VK_SHARING_MODE_EXCLUSIVE
        )
        buf = vkCreateBuffer(self.__device, bufferInfo, None)

        memRequirements = vkGetBufferMemoryRequirements(self.__device, buf)

        allocInfo = VkMemoryAllocateInfo(
            allocationSize=memRequirements.size,
            memoryTypeIndex=self.__findMemoryType(memRequirements.memoryTypeBits, properties)
        )

        bufMemory = vkAllocateMemory(self.__device, allocInfo, None)

        vkBindBufferMemory(self.__device, buf, bufMemory, 0)

        return (buf, bufMemory)

    def __beginSingleTimeCommands(self):
        allocInfo = VkCommandBufferAllocateInfo(
            level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandPool=self.__commandPool,
            commandBufferCount=1
        )

        commandBuffers = vkAllocateCommandBuffers(self.__device, allocInfo)

        beginInfo = VkCommandBufferBeginInfo(
            flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
        )

        vkBeginCommandBuffer(commandBuffers[0], beginInfo)

        return commandBuffers[0]

    def __endSingleTimeCommands(self, commandBuffer):
        vkEndCommandBuffer(commandBuffer)

        commandBuffers = ffi.new('VkCommandBuffer[]', [commandBuffer])
        submitInfo = VkSubmitInfo(
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
        # self.__commandBuffers = []

        allocInfo = VkCommandBufferAllocateInfo(
            commandPool=self.__commandPool,
            level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=len(self.__swapChainFramebuffers)
        )

        commandBuffers = vkAllocateCommandBuffers(self.__device, allocInfo)
        self.__commandBuffers = [ffi.addressof(commandBuffers, i)[0] for i in range(len(self.__swapChainFramebuffers))]

        for i, cmdBuffer in enumerate(self.__commandBuffers):
            beginInfo = VkCommandBufferBeginInfo(flags=VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT)

            vkBeginCommandBuffer(cmdBuffer, beginInfo)

            renderPassInfo = VkRenderPassBeginInfo(
                renderPass=self.__renderPass,
                framebuffer=self.__swapChainFramebuffers[i],
                renderArea=[[0, 0], self.__swapChainExtent]
            )

            clearColor = VkClearValue([[0.0, 0.0, 0.0, 1.0]])
            renderPassInfo.clearValueCount = 1
            renderPassInfo.pClearValues = ffi.addressof(clearColor)

            vkCmdBeginRenderPass(cmdBuffer, renderPassInfo, VK_SUBPASS_CONTENTS_INLINE)

            vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, self.__graphicsPipeline)

            vertexBuffers = ffi.new('VkBuffer[]', [self.__vertexBuffer,])
            # offsets = [0]
            offsets = ffi.new('uint64_t[]', [0, ])
            vkCmdBindVertexBuffers(cmdBuffer, 0, 1, vertexBuffers, offsets)

            vkCmdBindIndexBuffer(cmdBuffer, self.__indexBuffer, 0, VK_INDEX_TYPE_UINT16)

            vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, self.__pipelineLayout,
                                    0, 1, [self.__descriptorSet], 0, ffi.NULL)

            vkCmdDrawIndexed(cmdBuffer, len(self.indices), 1, 0, 0, 0)

            vkCmdEndRenderPass(cmdBuffer)

            vkEndCommandBuffer(cmdBuffer)

    def __createSemaphores(self):
        semaphoreInfo = VkSemaphoreCreateInfo()

        self.__imageAvailableSemaphore = vkCreateSemaphore(self.__device, semaphoreInfo, None)
        self.__renderFinishedSemaphore = vkCreateSemaphore(self.__device, semaphoreInfo, None)

    def __updateUniformBuffer(self):
        time = glfw.get_time()

        ubo = UniformBufferObject()
        ubo.model = glm.rotate(np.identity(4, np.float32), time * 90.0, 0.0, 0.0, 1.0)
        ubo.view = glm.lookAt(np.array([2, 2, 2], np.float32), np.array([0, 0, 0], np.float32),
                              np.array([0, 0, 1], np.float32))
        ubo.proj = glm.perspective(45, self.__swapChainExtent.width / float(self.__swapChainExtent.height), 0.1, 10.0)
        ubo.proj[1][1] *= -1

        data = vkMapMemory(self.__device, self.__uniformStagingBufferMemory, 0, ubo.nbytes, 0)
        ffi.memmove(data, ubo.to_c_ptr, ubo.nbytes)
        vkUnmapMemory(self.__device, self.__uniformStagingBufferMemory)

        self.__copyBuffer(self.__uniformStagingBuffer, self.__uniformBuffer, ubo.nbytes)

    def __drawFrame(self):
        vkAcquireNextImageKHR = vkGetDeviceProcAddr(self.__device, 'vkAcquireNextImageKHR')
        vkQueuePresentKHR = vkGetDeviceProcAddr(self.__device, 'vkQueuePresentKHR')

        try:
            imageIndex = vkAcquireNextImageKHR(self.__device, self.__swapChain, 18446744073709551615,
                                               self.__imageAvailableSemaphore, VK_NULL_HANDLE)
        except VkErrorOutOfDateKHR:
            self.recreateSwapChain()
            return

        submitInfo = VkSubmitInfo()

        waitSemaphores = ffi.new('VkSemaphore[]', [self.__imageAvailableSemaphore])
        waitStages = ffi.new('uint32_t[]', [VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, ])
        submitInfo.waitSemaphoreCount = 1
        submitInfo.pWaitSemaphores = waitSemaphores
        submitInfo.pWaitDstStageMask = waitStages

        cmdBuffers = ffi.new('VkCommandBuffer[]', [self.__commandBuffers[imageIndex], ])
        submitInfo.commandBufferCount = 1
        submitInfo.pCommandBuffers = cmdBuffers

        signalSemaphores = ffi.new('VkSemaphore[]', [self.__renderFinishedSemaphore])
        submitInfo.signalSemaphoreCount = 1
        submitInfo.pSignalSemaphores = signalSemaphores

        vkQueueSubmit(self.__graphicsQueue, 1, submitInfo, VK_NULL_HANDLE)

        swapChains = [self.__swapChain]
        presentInfo = VkPresentInfoKHR(
            waitSemaphoreCount=1,
            pWaitSemaphores=signalSemaphores,
            swapchainCount=1,
            pSwapchains=swapChains,
            pImageIndices=[imageIndex]
        )

        try:
            vkQueuePresentKHR(self.__presentQueue, presentInfo)
        except VkErrorOutOfDateKHR:
            self.recreateSwapChain()

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
        winWH = glfw.get_window_size(self.__window)
        width = max(capabilities.minImageExtent.width, min(capabilities.maxImageExtent.width, winWH[0]))
        height = max(capabilities.minImageExtent.height, min(capabilities.maxImageExtent.height, winWH[1]))
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

