using System;
using System.Collections.Generic;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Media3D;
using System.Reflection;
using Vulkan;
using Vulkan.Windows;

namespace vertex_buffer
{
    class QueueFamilyIndices
    {
        public uint GraphicsFamily;
        public uint PresentFamily;

        public bool IsComplete
        {
            get
            {
                return GraphicsFamily >= 0 && PresentFamily >= 0;
            }
        }
    }

    struct SwapChainSupportDetails
    {
        public SurfaceCapabilitiesKhr capabilities;
        public SurfaceFormatKhr[] formats;
        public PresentModeKhr[] presentModes;
    }

    struct Vertex
    {
        //public Vector pos;
        //public Vector3D color;

        static public VertexInputBindingDescription[] GetBindingDescription()
        {
            return new VertexInputBindingDescription[]
            {
                new VertexInputBindingDescription
                {
                    Binding = 0,
                    Stride = (2 + 3) * sizeof(float),
                    InputRate = VertexInputRate.Vertex
                }
            };
        }

        static public VertexInputAttributeDescription[] GetAttributeDescriptions()
        {
            return new VertexInputAttributeDescription[]
            {
                new VertexInputAttributeDescription
                {
                    Binding = 0,
                    Location = 0,
                    Format = Format.R32G32Sfloat,
                    Offset = 0
                },

                new VertexInputAttributeDescription
                {
                    Binding = 0,
                    Location = 1,
                    Format = Format.R32G32B32Sfloat,
                    Offset = 2 * sizeof(float)
                }
            };
        }
    }

    /// <summary>
    /// MainWindow.xaml 的交互逻辑
    /// </summary>
    public partial class MainWindow : Window
    {
        public bool enableValidationLayers = true;


        private Instance instance;
        private Instance.DebugReportCallback debugCallback;
        private PhysicalDevice physicalDevice;
        private Device device;
        private SurfaceKhr surface;

        private Queue graphicQueue;
        private Queue presentQueue;

        private SwapchainKhr swapChain;
        private Image[] swapChainImages;
        private Format swapChainImageFormat;
        private Extent2D swapChainextent;
        private List<ImageView> swapChainImageViews = new List<ImageView>();
        private List<Framebuffer> swapChainFramebuffers = new List<Framebuffer>();

        private RenderPass renderPass;
        private PipelineLayout pipelineLayout;
        private Pipeline graphicsPipeline;

        private CommandPool commandPool;

        private Vulkan.Buffer vertexBuffer;
        private DeviceMemory vertexBufferMemory;

        private CommandBuffer[] commandBuffers;

        private Semaphore imageAvailableSemaphore;
        private Semaphore renderFinishedSemaphore;

        private string[] validationLayers = new string[] {
            "VK_LAYER_LUNARG_standard_validation"
        };

        private string[] deviceExtenstions = new string[]
        {
            "VK_KHR_swapchain"
        };

        private float[] vertices = new float[] {
            // pos           color
            0.0f, -0.5f, 1.0f, 0.0f, 0.0f,
            0.5f, 0.5f, 0.0f, 1.0f, 0.0f,
            -0.5f, 0.5f, 0.0f, 0.0f, 1.0f
        };

        public MainWindow()
        {
            InitializeComponent();

            InitVulkan();

            CompositionTarget.Rendering += CompositionTarget_Rendering;

        }

        private void CompositionTarget_Rendering(object sender, EventArgs e)
        {
            DrawFrame();
        }

        protected override void OnRenderSizeChanged(SizeChangedInfo sizeChanged)
        {
            RecreateSwapChain();
            base.OnRenderSizeChanged(sizeChanged);
        }

        private void Window_Closed(object sender, EventArgs e)
        {
            //foreach(var imageView in swapChainImageViews)
            //{
            //    device.DestroyImageView(imageView);
            //}

            //device.DestroySwapchainKHR(swapChain);

            //instance.DestroySurfaceKHR(surface);
            //device.Destroy();

            //instance.Destroy();
            //Console.WriteLine("window closed instance destoryed.");
        }

        public void InitVulkan()
        {
            CreateInstance();
            SetupDebugCallback();
            CreateSurface();
            PickPhysicalDevice();
            CreateLogicalDevice();
            CreateSwapChain();
            CreateImageViews();
            CreateRenderPass();
            CreateGraphicPipeline();
            CreateFrameBuffers();
            CreateCommandPool();
            CreateVertexBuffer();
            CreateCommandBuffers();
            CreateSemaphores();
        }

        public void CleanupSwapChain()
        {
            foreach (var frameBuffer in swapChainFramebuffers)
            {
                device.DestroyFramebuffer(frameBuffer);
            }

            device.FreeCommandBuffers(commandPool, commandBuffers);

            device.DestroyPipeline(graphicsPipeline);
            device.DestroyPipelineLayout(pipelineLayout);
            device.DestroyRenderPass(renderPass);

            foreach (var imageView in swapChainImageViews)
            {
                device.DestroyImageView(imageView);
            }

            device.DestroySwapchainKHR(swapChain);
        }

        public void RecreateSwapChain()
        {
            if (this.Width == 0 || this.Height == 0)
                return;

            device.WaitIdle();

            CleanupSwapChain();

            CreateSwapChain();
            CreateImageViews();
            CreateRenderPass();
            CreateGraphicPipeline();
            CreateFrameBuffers();
            CreateCommandBuffers();
        }

        public void CreateInstance()
        {
            var appInfo = new Vulkan.ApplicationInfo
            {
                ApplicationName = "CSharp Vulkan",
                ApplicationVersion = Vulkan.Version.Make(1, 0, 0),
                ApiVersion = Vulkan.Version.Make(1, 0, 0),
                EngineName = "CSharp Engine",
                EngineVersion = Vulkan.Version.Make(1, 0, 0)
            };

            var extenstions = GetRequiredExtenstions();
            var createInfo = new Vulkan.InstanceCreateInfo
            {
                ApplicationInfo = appInfo,
                EnabledExtensionNames = extenstions
            };
            if (enableValidationLayers)
                createInfo.EnabledLayerNames = validationLayers;

            instance = new Vulkan.Instance(createInfo);
        }

        public void SetupDebugCallback()
        {
            if (enableValidationLayers)
            {
                debugCallback = new Vulkan.Instance.DebugReportCallback(DebugReportCallback);
                instance.EnableDebug(debugCallback, DebugReportFlagsExt.Warning | DebugReportFlagsExt.Error);
            }
        }

        public void CreateSurface()
        {
            var hWnd = new System.Windows.Interop.WindowInteropHelper(this).EnsureHandle();
            var hInstance = System.Runtime.InteropServices.Marshal.GetHINSTANCE(typeof(App).Module);
            surface = instance.CreateWin32SurfaceKHR(new Win32SurfaceCreateInfoKhr { Hwnd = hWnd, Hinstance = hInstance });
        }

        public void PickPhysicalDevice()
        {
            var physicalDevices = instance.EnumeratePhysicalDevices();
            foreach (var device in physicalDevices)
            {
                if (IsDeviceSuitable(device))
                {
                    physicalDevice = device;
                    break;
                }
            }

            if (physicalDevice == null)
                throw new Exception("No suitable GPU found!");
        }

        public void CreateLogicalDevice()
        {
            var indices = FindQueueFamilies(physicalDevice);
            HashSet<uint> uniqueueFamilies = new HashSet<uint> { indices.GraphicsFamily, indices.PresentFamily };

            float[] queuePrioriteise = { 1.0f };
            List<DeviceQueueCreateInfo> queueCreateInfo = new List<DeviceQueueCreateInfo>();
            foreach (var queueFamily in uniqueueFamilies)
            {
                queueCreateInfo.Add(
                    new DeviceQueueCreateInfo
                    {
                        QueueFamilyIndex = queueFamily,
                        QueuePriorities = queuePrioriteise
                    }
                );
            }

            var deviceFeatures = new Vulkan.PhysicalDeviceFeatures();
            var createInfo = new Vulkan.DeviceCreateInfo
            {
                QueueCreateInfos = queueCreateInfo.ToArray(),
                EnabledFeatures = deviceFeatures,
                EnabledExtensionNames = deviceExtenstions
            };

            if (enableValidationLayers)
            {
                createInfo.EnabledLayerNames = validationLayers;
            }

            device = physicalDevice.CreateDevice(createInfo);

            graphicQueue = device.GetQueue(indices.GraphicsFamily, 0);
            presentQueue = device.GetQueue(indices.PresentFamily, 0);
        }

        public void CreateSwapChain()
        {
            var swapChainSupport = QuerySwapChainSupport(physicalDevice);

            var surfaceFormat = ChooseSwapSurfaceFormat(swapChainSupport.formats);
            var presentMode = ChooseSwapPresentMode(swapChainSupport.presentModes);
            var extent = ChooseSwapExten(swapChainSupport.capabilities);

            uint imageCount = swapChainSupport.capabilities.MinImageCount + 1;
            if (swapChainSupport.capabilities.MaxImageCount > 0 && imageCount > swapChainSupport.capabilities.MaxImageCount)
                imageCount = swapChainSupport.capabilities.MaxImageCount;

            var createInfo = new SwapchainCreateInfoKhr
            {
                Surface = surface,
                MinImageCount = imageCount,
                ImageFormat = surfaceFormat.Format,
                ImageColorSpace = surfaceFormat.ColorSpace,
                ImageExtent = extent,
                ImageArrayLayers = 1,
                ImageUsage = ImageUsageFlags.ColorAttachment
            };

            var indices = FindQueueFamilies(physicalDevice);
            uint[] queueFamilyIndices = { indices.GraphicsFamily, indices.PresentFamily };

            if (indices.GraphicsFamily != indices.PresentFamily)
            {
                createInfo.ImageSharingMode = SharingMode.Concurrent;
                createInfo.QueueFamilyIndices = queueFamilyIndices;
            }
            else
            {
                createInfo.ImageSharingMode = SharingMode.Exclusive;
            }

            createInfo.PreTransform = swapChainSupport.capabilities.CurrentTransform;
            createInfo.CompositeAlpha = CompositeAlphaFlagsKhr.Opaque;
            createInfo.PresentMode = presentMode;
            createInfo.Clipped = true;
            createInfo.OldSwapchain = null;

            swapChain = device.CreateSwapchainKHR(createInfo);

            swapChainImages = device.GetSwapchainImagesKHR(swapChain);
            swapChainImageFormat = surfaceFormat.Format;
            swapChainextent = extent;
        }

        public void CreateImageViews()
        {
            swapChainImageViews.Clear();

            foreach (var image in swapChainImages)
            {
                var createInfo = new ImageViewCreateInfo
                {
                    Image = image,
                    ViewType = ImageViewType.View2D,
                    Format = swapChainImageFormat,
                    Components = new ComponentMapping
                    {
                        R = ComponentSwizzle.Identity,
                        G = ComponentSwizzle.Identity,
                        B = ComponentSwizzle.Identity,
                        A = ComponentSwizzle.Identity
                    },
                    SubresourceRange = new ImageSubresourceRange
                    {
                        AspectMask = ImageAspectFlags.Color,
                        BaseMipLevel = 0,
                        LevelCount = 1,
                        BaseArrayLayer = 0,
                        LayerCount = 1
                    }
                };

                swapChainImageViews.Add(device.CreateImageView(createInfo));
            }
        }

        public void CreateRenderPass()
        {
            AttachmentDescription[] colorAttachment = new AttachmentDescription[]
            {
                new AttachmentDescription
                {
                    Format = swapChainImageFormat,
                    Samples = SampleCountFlags.Count1,
                    LoadOp = AttachmentLoadOp.Clear,
                    StoreOp = AttachmentStoreOp.Store,
                    StencilLoadOp = AttachmentLoadOp.DontCare,
                    StencilStoreOp = AttachmentStoreOp.DontCare,
                    InitialLayout = ImageLayout.Undefined,
                    FinalLayout = ImageLayout.PresentSrcKhr
                }
            };

            AttachmentReference[] colorAttachmentRefs = new AttachmentReference[]
            {
                new AttachmentReference
                {
                    Attachment = 0,
                    Layout = ImageLayout.ColorAttachmentOptimal
                }
            };

            SubpassDescription[] subpasses = new SubpassDescription[]
            {
                new SubpassDescription
                {
                    PipelineBindPoint = PipelineBindPoint.Graphics,
                    ColorAttachments = colorAttachmentRefs
                }
            };

            var renderPassInfo = new RenderPassCreateInfo
            {
                Attachments = colorAttachment,
                Subpasses = subpasses
            };

            renderPass = device.CreateRenderPass(renderPassInfo);
        }

        public void CreateGraphicPipeline()
        {
            var vertShaderCode = LoadResource(string.Format("{0}.vert.spv", typeof(MainWindow).Namespace));
            var fragShaderCode = LoadResource(string.Format("{0}.frag.spv", typeof(MainWindow).Namespace));

            var vertShaderModule = device.CreateShaderModule(vertShaderCode);
            var fragShaderModule = device.CreateShaderModule(fragShaderCode);

            PipelineShaderStageCreateInfo[] shaderStageCreateInfos =
            {
                new PipelineShaderStageCreateInfo
                {
                    Stage = ShaderStageFlags.Vertex,
                    Module = vertShaderModule,
                    Name = "main"
                },

                new PipelineShaderStageCreateInfo
                {
                    Stage = ShaderStageFlags.Fragment,
                    Module = fragShaderModule,
                    Name = "main"
                }
            };

            var vertexInputInfo = new PipelineVertexInputStateCreateInfo
            {
                VertexBindingDescriptions = Vertex.GetBindingDescription(),
                VertexAttributeDescriptions = Vertex.GetAttributeDescriptions()
            };

            var inputAssembly = new PipelineInputAssemblyStateCreateInfo
            {
                Topology = PrimitiveTopology.TriangleList,
                PrimitiveRestartEnable = false
            };

            var viewport = new Viewport
            {
                X = 0.0f,
                Y = 0.0f,
                Width = (float)swapChainextent.Width,
                Height = (float)swapChainextent.Height,
                MinDepth = 0.0f,
                MaxDepth = 0.0f
            };

            var scissor = new Rect2D();
            scissor.Offset.X = 0;
            scissor.Offset.Y = 0;
            scissor.Extent = swapChainextent;

            var viewportState = new PipelineViewportStateCreateInfo
            {
                Viewports = new Viewport[] { viewport },
                Scissors = new Rect2D[] { scissor }
            };

            var rasterizer = new PipelineRasterizationStateCreateInfo
            {
                DepthClampEnable = false,
                RasterizerDiscardEnable = false,
                PolygonMode = PolygonMode.Fill,
                LineWidth = 1.0f,
                CullMode = CullModeFlags.Back,
                FrontFace = FrontFace.Clockwise,
                DepthBiasEnable = false
            };

            var multisampling = new PipelineMultisampleStateCreateInfo
            {
                SampleShadingEnable = false,
                RasterizationSamples = SampleCountFlags.Count1
            };

            var colorBlendAttachment = new PipelineColorBlendAttachmentState
            {
                ColorWriteMask = ColorComponentFlags.R | ColorComponentFlags.G | ColorComponentFlags.B | ColorComponentFlags.A,
                BlendEnable = false
            };

            var colorBlending = new PipelineColorBlendStateCreateInfo
            {
                LogicOpEnable = false,
                LogicOp = LogicOp.Copy,
                Attachments = new PipelineColorBlendAttachmentState[] { colorBlendAttachment },
                BlendConstants = new float[] { 0.0f, 0.0f, 0.0f, 0.0f }
            };

            var pipelineLayoutInfo = new PipelineLayoutCreateInfo
            {
                SetLayoutCount = 0,
                PushConstantRangeCount = 0
            };

            pipelineLayout = device.CreatePipelineLayout(pipelineLayoutInfo);

            GraphicsPipelineCreateInfo[] pipelineInfo = new GraphicsPipelineCreateInfo[]
            {
                new GraphicsPipelineCreateInfo
                {
                    Stages = shaderStageCreateInfos,
                    VertexInputState = vertexInputInfo,
                    InputAssemblyState = inputAssembly,
                    ViewportState = viewportState,
                    RasterizationState = rasterizer,
                    MultisampleState = multisampling,
                    ColorBlendState = colorBlending,
                    Layout = pipelineLayout,
                    RenderPass = renderPass,
                    Subpass = 0,
                    BasePipelineHandle = null
                }
            };

            graphicsPipeline = device.CreateGraphicsPipelines(null, pipelineInfo)[0];

            device.DestroyShaderModule(vertShaderModule);
            device.DestroyShaderModule(fragShaderModule);
        }

        public void CreateFrameBuffers()
        {
            swapChainFramebuffers.Clear();

            foreach (var imageView in swapChainImageViews)
            {
                ImageView[] attachments = { imageView };

                var framebufferInfo = new FramebufferCreateInfo
                {
                    RenderPass = renderPass,
                    Attachments = attachments,
                    Width = swapChainextent.Width,
                    Height = swapChainextent.Height,
                    Layers = 1
                };

                swapChainFramebuffers.Add(device.CreateFramebuffer(framebufferInfo));
            }
        }

        public void CreateCommandPool()
        {
            var queueFamilyIndices = FindQueueFamilies(physicalDevice);

            var poolInfo = new CommandPoolCreateInfo
            {
                QueueFamilyIndex = queueFamilyIndices.GraphicsFamily
            };

            commandPool = device.CreateCommandPool(poolInfo);
        }

        public void CreateVertexBuffer()
        {
            var vertSize = vertices.Length * sizeof(float);
            var bufferInfo = new BufferCreateInfo
            {
                Size = vertSize,
                Usage = BufferUsageFlags.VertexBuffer,
                SharingMode = SharingMode.Exclusive
            };

            vertexBuffer = device.CreateBuffer(bufferInfo);

            var memRequirements = device.GetBufferMemoryRequirements(vertexBuffer);

            var allocInfo = new MemoryAllocateInfo
            {
                AllocationSize = memRequirements.Size,
                MemoryTypeIndex = FindMemoryType(memRequirements.MemoryTypeBits, MemoryPropertyFlags.HostVisible | MemoryPropertyFlags.HostCoherent)
            };

            vertexBufferMemory = device.AllocateMemory(allocInfo);

            device.BindBufferMemory(vertexBuffer, vertexBufferMemory, 0);

            var data = device.MapMemory(vertexBufferMemory, 0, vertSize, 0);
            System.Runtime.InteropServices.Marshal.Copy(vertices, 0, data, vertices.Length);
            device.UnmapMemory(vertexBufferMemory);
        }

        private uint FindMemoryType(uint typeFilter, MemoryPropertyFlags propertyFlags)
        {
            var memProperties = physicalDevice.GetMemoryProperties();

            uint i = 0;
            foreach(var memType in memProperties.MemoryTypes)
            {
                if ((((typeFilter >> (int)i) & 1) == 1) && ((memType.PropertyFlags & propertyFlags) == propertyFlags))
                    return i;
                i++;
            }

            throw new Exception("failed to find memory type");
        }

        public void CreateCommandBuffers()
        {
            var allocateInfos = new CommandBufferAllocateInfo
            {
                CommandPool = commandPool,
                Level = CommandBufferLevel.Primary,
                CommandBufferCount = (uint)swapChainFramebuffers.Count
            };

            commandBuffers = device.AllocateCommandBuffers(allocateInfos);

            var i = 0;
            foreach (var cmdBuffer in commandBuffers)
            {
                var beginInfo = new CommandBufferBeginInfo
                {
                    Flags = CommandBufferUsageFlags.SimultaneousUse
                };

                cmdBuffer.Begin(beginInfo);

                var renderPassInfo = new RenderPassBeginInfo
                {
                    RenderPass = renderPass,
                    Framebuffer = swapChainFramebuffers[i],
                    RenderArea = new Rect2D { Offset = new Offset2D { X = 0, Y = 0 }, Extent = swapChainextent },
                    ClearValues = new ClearValue[] { new ClearValue { Color = new ClearColorValue(new float[] { 0.0f, 0.0f, 0.0f, 1.0f }) } }
                };

                cmdBuffer.CmdBeginRenderPass(renderPassInfo, SubpassContents.Inline);
                cmdBuffer.CmdBindPipeline(PipelineBindPoint.Graphics, graphicsPipeline);

                cmdBuffer.CmdBindVertexBuffer(0, vertexBuffer, 0);
                //Vulkan.Buffer[] vertexBuffers = { vertexBuffer };
                //DeviceSize[] offsets = { 0 };
                //cmdBuffer.CmdBindVertexBuffers(0, vertexBuffers, offsets);

                cmdBuffer.CmdDraw(3, 1, 0, 0);

                cmdBuffer.CmdEndRenderPass();

                cmdBuffer.End();
                i++;
            }
        }

        public void CreateSemaphores()
        {
            var semaphoreInfo = new SemaphoreCreateInfo();
            imageAvailableSemaphore = device.CreateSemaphore(semaphoreInfo);
            renderFinishedSemaphore = device.CreateSemaphore(semaphoreInfo);
        }

        public void DrawFrame()
        {
            var imageIndex = device.AcquireNextImageKHR(swapChain, UInt16.MaxValue, imageAvailableSemaphore);

            Semaphore[] waitSemaphores = { imageAvailableSemaphore };
            Semaphore[] signalSemaphores = { renderFinishedSemaphore };
            var submitInfo = new SubmitInfo
            {
                WaitSemaphores = waitSemaphores,
                WaitDstStageMask = new PipelineStageFlags[] { PipelineStageFlags.ColorAttachmentOutput },
                CommandBuffers = new CommandBuffer[] { commandBuffers[imageIndex] },
                SignalSemaphores = signalSemaphores
            };

            graphicQueue.Submit(submitInfo);

            var presentInfo = new PresentInfoKhr
            {
                Swapchains = new SwapchainKhr[] { swapChain },
                WaitSemaphores = signalSemaphores,
                ImageIndices = new uint[] { imageIndex }
            };

            presentQueue.PresentKHR(presentInfo);

            if (enableValidationLayers)
                presentQueue.WaitIdle();
        }

        private SurfaceFormatKhr ChooseSwapSurfaceFormat(SurfaceFormatKhr[] availableFormats)
        {
            if (availableFormats.Length == 1 && availableFormats[0].Format == Format.Undefined)
            {
                return new SurfaceFormatKhr
                {
                    Format = Format.R8G8B8A8Unorm,
                    ColorSpace = ColorSpaceKhr.SrgbNonlinear
                };
            }

            foreach (var availableFormat in availableFormats)
            {
                if (availableFormat.Format == Format.R8G8B8A8Unorm && availableFormat.ColorSpace == ColorSpaceKhr.SrgbNonlinear)
                    return availableFormat;
            }

            return availableFormats[0];
        }

        private PresentModeKhr ChooseSwapPresentMode(PresentModeKhr[] presentModes)
        {
            var bestMode = Vulkan.PresentModeKhr.Fifo;

            foreach (var mode in presentModes)
            {
                if (mode == PresentModeKhr.Mailbox)
                {
                    return mode;
                }
                else if (mode == PresentModeKhr.Immediate)
                    return mode;
            }

            return bestMode;
        }

        private Extent2D ChooseSwapExten(SurfaceCapabilitiesKhr capabilities)
        {
            if (capabilities.CurrentExtent.Width != UInt32.MaxValue)
                return capabilities.CurrentExtent;
            else
            {
                var extent = new Extent2D
                {
                    Width = Math.Max(capabilities.MinImageExtent.Width, Math.Min(capabilities.MaxImageExtent.Width, (uint)this.Width)),
                    Height = Math.Max(capabilities.MinImageExtent.Height, Math.Min(capabilities.MaxImageExtent.Height, (uint)this.Height))
                };

                return extent;
            }
        }

        private SwapChainSupportDetails QuerySwapChainSupport(PhysicalDevice device)
        {
            SwapChainSupportDetails details = new SwapChainSupportDetails
            {
                capabilities = device.GetSurfaceCapabilitiesKHR(surface),
                formats = device.GetSurfaceFormatsKHR(surface),
                presentModes = device.GetSurfacePresentModesKHR(surface)
            };

            return details;
        }

        private bool IsDeviceSuitable(PhysicalDevice physicalDevice)
        {
            var indices = FindQueueFamilies(physicalDevice);

            var extensionSupported = CheckDeviceExtenstionSupported(physicalDevice);

            bool swapChainAdequate = false;
            if (extensionSupported)
            {
                var swapChainSupport = QuerySwapChainSupport(physicalDevice);
                swapChainAdequate = swapChainSupport.formats.Length != 0 && swapChainSupport.presentModes.Length != 0;
            }

            return indices.IsComplete && extensionSupported && swapChainAdequate;
        }

        private bool CheckDeviceExtenstionSupported(PhysicalDevice device)
        {
            var availableExtentions = device.EnumerateDeviceExtensionProperties();
            List<string> eNames = new List<string>();
            foreach (var exten in availableExtentions)
            {
                eNames.Add(exten.ExtensionName);
            }

            foreach (var name in deviceExtenstions)
            {
                if (!eNames.Contains(name))
                    return false;
            }

            return true;
        }

        private QueueFamilyIndices FindQueueFamilies(Vulkan.PhysicalDevice physicalDevice)
        {
            var indices = new QueueFamilyIndices();

            uint i = 0;
            var familyProperties = physicalDevice.GetQueueFamilyProperties();
            foreach (var property in familyProperties)
            {
                if ((property.QueueCount > 0) && ((property.QueueFlags & QueueFlags.Graphics) == QueueFlags.Graphics))
                {
                    indices.GraphicsFamily = i;
                }

                var presentSupport = physicalDevice.GetSurfaceSupportKHR(i, surface);

                if (property.QueueCount > 0 && presentSupport)
                    indices.PresentFamily = i;

                if (indices.IsComplete)
                    break;
                i++;
            }

            return indices;
        }

        private string[] GetRequiredExtenstions()
        {
            var extenstions = Commands.EnumerateInstanceExtensionProperties();
            List<string> extenstionNames = new List<string>();
            foreach (var extension in extenstions)
            {
                extenstionNames.Add(extension.ExtensionName);
            }

            if (enableValidationLayers)
                extenstionNames.Add("VK_EXT_debug_report");

            return extenstionNames.ToArray();
        }

        private bool CheckValidationLayerSupport()
        {
            var avilablelayers = Commands.EnumerateInstanceLayerProperties();
            bool layerFound = false;

            foreach (var layer in validationLayers)
            {
                foreach (var prop in avilablelayers)
                {
                    if (layer == prop.LayerName)
                    {
                        layerFound = true;
                        break;
                    }
                }

                if (!layerFound)
                    return layerFound;
            }

            return layerFound;
        }

        byte[] LoadResource(string name)
        {
            System.IO.Stream stream = typeof(MainWindow).GetTypeInfo().Assembly.GetManifestResourceStream(name);
            byte[] bytes = new byte[stream.Length];
            stream.Read(bytes, 0, (int)stream.Length);

            return bytes;
        }

        static Bool32 DebugReportCallback(DebugReportFlagsExt flags, DebugReportObjectTypeExt objectType, ulong objectHandle, IntPtr location, int messageCode, IntPtr layerPrefix, IntPtr message, IntPtr userData)
        {
            string layerString = System.Runtime.InteropServices.Marshal.PtrToStringAnsi(layerPrefix);
            string messageString = System.Runtime.InteropServices.Marshal.PtrToStringAnsi(message);

            Console.WriteLine("DebugReport layer: {0} message: {1}", layerString, messageString);

            return false;
        }
    }
}
