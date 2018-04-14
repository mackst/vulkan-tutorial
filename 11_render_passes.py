# -*- coding: UTF-8 -*-

import sys

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


class HelloTriangleApplication(QtGui.QWindow):

    def __init__(self):
        super(HelloTriangleApplication, self).__init__()

        self.setWidth(1280)
        self.setHeight(720)

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

        self.__renderpass = None
        self.__pipelineLayout = None

        self.__indices = QueueFamilyIndices()

        self.initVulkan()

    def __del__(self):
        if self.__renderpass:
            vkDestroyRenderPass(self.__device, self.__renderpass, None)

        if self.__pipelineLayout:
            vkDestroyPipelineLayout(self.__device, self.__pipelineLayout, None)

        if self.__swapChainImageViews:
            [vkDestroyImageView(self.__device, imv, None) for imv in self.__swapChainImageViews]

        if self.__swapChain:
            vkDestroySwapchainKHR(self.__device, self.__swapChain, None)

        if self.__device:
            vkDestroyDevice(self.__device, None)

        if self.__callbcak:
            vkDestroyDebugReportCallbackEXT(self.__instance, self.__callbcak, None)

        if self.__surface:
            vkDestroySurfaceKHR(self.__instance, self.__surface, None)

        if self.__instance:
            vkDestroyInstance(self.__instance, None)
            print('instance destroyed')

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
        self.__indices = self.__findQueueFamilies(self.__physicalDevice)

        uniqueQueueFamilies = {}.fromkeys([self.__indices.graphicsFamily, self.__indices.presentFamily])
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

        self.__graphicQueue = vkGetDeviceQueue(self.__device, self.__indices.graphicsFamily, 0)
        self.__presentQueue = vkGetDeviceQueue(self.__device, self.__indices.presentFamily, 0)

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

        vertexInputInfo = VkPipelineVertexInputStateCreateInfo(
            vertexBindingDescriptionCount=0,
            vertexAttributeDescriptionCount=0
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

        vkDestroyShaderModule(self.__device, vertexShaderMode, None)
        vkDestroyShaderModule(self.__device, fragmentShaderMode, None)

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
