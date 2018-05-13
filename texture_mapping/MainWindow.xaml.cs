using System;
using System.Collections.Generic;
using System.Windows;
using System.Windows.Media;
using System.Runtime.InteropServices;
using System.Reflection;
using System.Numerics;
using System.Drawing;

using Vulkan;
using Vulkan.Windows;

namespace texture_mapping
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
        static public VertexInputBindingDescription[] GetBindingDescription()
        {
            return new VertexInputBindingDescription[]
            {
                new VertexInputBindingDescription
                {
                    Binding = 0,
                    Stride = (uint)(2 * Marshal.SizeOf<Vector2>() + Marshal.SizeOf<Vector3>()),
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
                    Offset = (uint)Marshal.SizeOf<Vector2>()
                },

                new VertexInputAttributeDescription
                {
                    Binding = 0,
                    Location = 2,
                    Format = Format.R32G32Sfloat,
                    Offset = (uint)(Marshal.SizeOf<Vector2>() + Marshal.SizeOf<Vector3>())
                }
            };
        }
    }

    struct UniformBufferObject
    {
        public Matrix4x4 model;
        public Matrix4x4 view;
        public Matrix4x4 proj;

        public float[] ToArray()
        {
            List<float> outArray = new List<float>();
            var mtype = typeof(Matrix4x4);
            Matrix4x4[] matrices = { model, view, proj };
            foreach (var mat in matrices)
            {
                var fields = mat.GetType().GetFields();
                foreach (var field in fields)
                {
                    outArray.Add((float)field.GetValue(mat));
                }
            }
            return outArray.ToArray();
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
        private Vulkan.Image[] swapChainImages;
        private Format swapChainImageFormat;
        private Extent2D swapChainextent;
        private List<ImageView> swapChainImageViews = new List<ImageView>();
        private List<Framebuffer> swapChainFramebuffers = new List<Framebuffer>();

        private RenderPass renderPass;
        private DescriptorSetLayout descriptorSetLayout;
        private PipelineLayout pipelineLayout;
        private Pipeline graphicsPipeline;

        private CommandPool commandPool;

        private Vulkan.Image textureImage;
        private DeviceMemory textureImageMemory;
        private ImageView textureImageView;
        private Sampler textureSampler;

        private Vulkan.Buffer vertexBuffer;
        private DeviceMemory vertexBufferMemory;
        private Vulkan.Buffer indexBuffer;
        private DeviceMemory indexBufferMemory;

        private Vulkan.Buffer uniformBuffer;
        private DeviceMemory uniformBufferMemory;

        private DescriptorPool descriptorPool;
        private DescriptorSet descriptorSet;

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
            // pos           color            texcoord
            -0.5f, -0.5f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f,
            0.5f, -0.5f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
            0.5f, 0.5f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f,
            -0.5f, 0.5f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        };

        private short[] indices = new short[]
        {
            0, 1, 2, 2, 3, 0,
        };

        private DateTime startTime;
        private DateTime currentTime;

        public MainWindow()
        {
            InitializeComponent();

            startTime = DateTime.Now;

            InitVulkan();

            CompositionTarget.Rendering += CompositionTarget_Rendering;
        }

        private void CompositionTarget_Rendering(object sender, EventArgs e)
        {
            UpdateUniformBuffer();
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
            CreateDescriptorSetLayout();
            CreateGraphicPipeline();
            CreateFrameBuffers();
            CreateCommandPool();
            CreateTextureImage();
            CreateTextureImageView();
            CreateTextureSampler();
            CreateVertexBuffer();
            CreateIndexBuffer();
            CreateUniformBuffer();
            CreateDescriptorPool();
            CreateDescriptorSet();
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
            deviceFeatures.SamplerAnisotropy = true;

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
                swapChainImageViews.Add(CreateImageView(image, swapChainImageFormat));
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

        public void CreateDescriptorSetLayout()
        {
            DescriptorSetLayoutBinding[] bindings = new DescriptorSetLayoutBinding[]
            {
                new DescriptorSetLayoutBinding
                {
                    Binding = 0,
                    DescriptorCount = 1,
                    DescriptorType = DescriptorType.UniformBuffer,
                    StageFlags = ShaderStageFlags.Vertex
                },

                new DescriptorSetLayoutBinding
                {
                    Binding = 1,
                    DescriptorCount = 1,
                    DescriptorType = DescriptorType.CombinedImageSampler,
                    StageFlags = ShaderStageFlags.Fragment
                }
            };

            var layoutInfo = new DescriptorSetLayoutCreateInfo
            {
                Bindings = bindings
            };

            descriptorSetLayout = device.CreateDescriptorSetLayout(layoutInfo);
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
                SetLayouts = new DescriptorSetLayout[] { descriptorSetLayout }
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

        public void CreateTextureImage()
        {
            string textureName = string.Format("{0}.texture.jpg", typeof(MainWindow).Namespace);
            System.IO.Stream stream = typeof(MainWindow).GetTypeInfo().Assembly.GetManifestResourceStream(textureName);
            Bitmap bitmap = new Bitmap(stream);
            bitmap.MakeTransparent();

            int texWidth = bitmap.Width;
            int texHeight = bitmap.Height;

            int imageSize = texWidth * texHeight * 4;

            Rectangle rect = new Rectangle(0, 0, texWidth, texHeight);
            var bmpData = bitmap.LockBits(rect, System.Drawing.Imaging.ImageLockMode.ReadOnly, bitmap.PixelFormat);

            IntPtr pixelPtr = bmpData.Scan0;
            byte[] pixels = new byte[imageSize];

            Marshal.Copy(pixelPtr, pixels, 0, imageSize);

            bitmap.UnlockBits(bmpData);
            stream.Close();
            bitmap.Dispose();

            Vulkan.Buffer stagingBuffer;
            DeviceMemory stagingBufferMemory;
            (stagingBuffer, stagingBufferMemory) = CreateBuffer(imageSize, BufferUsageFlags.TransferSrc, MemoryPropertyFlags.HostVisible | MemoryPropertyFlags.HostCoherent);

            var data = device.MapMemory(stagingBufferMemory, 0, imageSize);
            Marshal.Copy(pixels, 0, data, pixels.Length);
            device.UnmapMemory(stagingBufferMemory);

            (textureImage, textureImageMemory) = CreateImage((uint)texWidth, (uint)texHeight, Format.B8G8R8A8Unorm, ImageTiling.Optimal, ImageUsageFlags.TransferDst | ImageUsageFlags.Sampled, MemoryPropertyFlags.DeviceLocal);

            TransitionImageLayout(textureImage, Format.B8G8R8A8Unorm, ImageLayout.Undefined, ImageLayout.TransferDstOptimal);
            CopyBufferToImage(stagingBuffer, textureImage, (uint)texWidth, (uint)texHeight);
            TransitionImageLayout(textureImage, Format.B8G8R8A8Unorm, ImageLayout.TransferDstOptimal, ImageLayout.ShaderReadOnlyOptimal);

            device.DestroyBuffer(stagingBuffer);
            device.FreeMemory(stagingBufferMemory);
        }

        public ImageView CreateImageView(Vulkan.Image image, Format format)
        {
            var viewInfo = new ImageViewCreateInfo
            {
                Image = image,
                ViewType = ImageViewType.View2D,
                Format = format,
                SubresourceRange = new ImageSubresourceRange
                {
                    AspectMask = ImageAspectFlags.Color,
                    BaseMipLevel = 0,
                    LevelCount = 1,
                    BaseArrayLayer = 0,
                    LayerCount = 1
                }
            };

            return device.CreateImageView(viewInfo);
        }

        public (Vulkan.Image, DeviceMemory) CreateImage(uint width, uint height, Format format, ImageTiling imageTiling, ImageUsageFlags usageFlags, MemoryPropertyFlags memoryProperty)
        {
            var imageInfo = new ImageCreateInfo
            {
                ImageType = ImageType.Image2D,
                Extent = new Extent3D
                {
                    Width = width,
                    Height = height,
                    Depth = 1
                },
                MipLevels = 1,
                ArrayLayers = 1,
                Format = format,
                Tiling = imageTiling,
                InitialLayout = ImageLayout.Undefined,
                Usage = usageFlags,
                Samples = SampleCountFlags.Count1,
                SharingMode = SharingMode.Exclusive
            };

            var image = device.CreateImage(imageInfo);

            var memRequirements = device.GetImageMemoryRequirements(image);

            var allocInfo = new MemoryAllocateInfo
            {
                AllocationSize = memRequirements.Size,
                MemoryTypeIndex = FindMemoryType(memRequirements.MemoryTypeBits, memoryProperty)
            };

            var imageMemory = device.AllocateMemory(allocInfo);

            device.BindImageMemory(image, imageMemory, 0);

            return (image, imageMemory);
        }

        public void TransitionImageLayout(Vulkan.Image image, Format format, ImageLayout oldLayout, ImageLayout newLayout)
        {
            var cmdBuffer = BeginSingleTimeCommands();

            ImageMemoryBarrier[] barriers = new ImageMemoryBarrier[]
            {
                new ImageMemoryBarrier
                {
                    OldLayout = oldLayout,
                    NewLayout = newLayout,
                    Image = image,
                    SubresourceRange = new ImageSubresourceRange
                    {
                        AspectMask = ImageAspectFlags.Color,
                        BaseMipLevel = 0,
                        LayerCount = 1,
                        BaseArrayLayer = 0,
                        LevelCount = 1
                    }
                },
            };

            PipelineStageFlags sourceStage;
            PipelineStageFlags destinationStage;

            if (oldLayout == ImageLayout.Undefined && newLayout == ImageLayout.TransferDstOptimal)
            {
                barriers[0].SrcAccessMask = 0;
                barriers[0].DstAccessMask = AccessFlags.TransferWrite;

                sourceStage = PipelineStageFlags.TopOfPipe;
                destinationStage = PipelineStageFlags.Transfer;
            }
            else if (oldLayout == ImageLayout.TransferDstOptimal && newLayout == ImageLayout.ShaderReadOnlyOptimal)
            {
                barriers[0].SrcAccessMask = AccessFlags.TransferWrite;
                barriers[0].DstAccessMask = AccessFlags.ShaderRead;

                sourceStage = PipelineStageFlags.Transfer;
                destinationStage = PipelineStageFlags.FragmentShader;
            }
            else
            {
                throw new Exception("unsupport layout transition!");
            }

            cmdBuffer.CmdPipelineBarrier(sourceStage, destinationStage,
                0, null, null, barriers);

            EndSingleTimeCommands(cmdBuffer);
        }

        public void CopyBufferToImage(Vulkan.Buffer buffer, Vulkan.Image image, uint width, uint height)
        {
            var cmdBuffer = BeginSingleTimeCommands();

            var region = new BufferImageCopy
            {
                BufferOffset = 0,
                BufferRowLength = 0,
                BufferImageHeight = 0,
                ImageSubresource = new ImageSubresourceLayers
                {
                    AspectMask = ImageAspectFlags.Color,
                    MipLevel = 0,
                    BaseArrayLayer = 0,
                    LayerCount = 1
                },
                ImageOffset = new Offset3D
                {
                    X = 0,
                    Y = 0,
                    Z = 0
                },
                ImageExtent = new Extent3D
                {
                    Width = width,
                    Height = height,
                    Depth = 1
                }
            };

            cmdBuffer.CmdCopyBufferToImage(buffer, image, ImageLayout.TransferDstOptimal, region);

            EndSingleTimeCommands(cmdBuffer);
        }

        public void CreateTextureImageView()
        {
            textureImageView = CreateImageView(textureImage, Format.B8G8R8A8Unorm);
        }

        public void CreateTextureSampler()
        {
            var samplerInfo = new SamplerCreateInfo
            {
                MagFilter = Filter.Linear,
                MinFilter = Filter.Linear,
                AddressModeU = SamplerAddressMode.Repeat,
                AddressModeV = SamplerAddressMode.Repeat,
                AddressModeW = SamplerAddressMode.Repeat,
                AnisotropyEnable = true,
                MaxAnisotropy = 16,
                BorderColor = BorderColor.IntOpaqueBlack,
                UnnormalizedCoordinates = false,
                CompareEnable = false,
                CompareOp = CompareOp.Always,
                MipmapMode = SamplerMipmapMode.Linear
            };

            textureSampler = device.CreateSampler(samplerInfo);
        }

        public void CreateVertexBuffer()
        {
            var vertSize = vertices.Length * sizeof(float);
            Vulkan.Buffer staginBuffer;
            DeviceMemory stagingBufferMemory;
            (staginBuffer, stagingBufferMemory) = CreateBuffer(vertSize, BufferUsageFlags.TransferSrc, MemoryPropertyFlags.HostVisible | MemoryPropertyFlags.HostCoherent);

            var data = device.MapMemory(stagingBufferMemory, 0, vertSize, 0);
            System.Runtime.InteropServices.Marshal.Copy(vertices, 0, data, vertices.Length);
            device.UnmapMemory(stagingBufferMemory);

            (vertexBuffer, vertexBufferMemory) = CreateBuffer(vertSize, BufferUsageFlags.TransferDst | BufferUsageFlags.VertexBuffer, MemoryPropertyFlags.DeviceLocal);

            CopyBuffer(staginBuffer, vertexBuffer, vertSize);

            device.DestroyBuffer(staginBuffer);
            device.FreeMemory(stagingBufferMemory);
        }

        private uint FindMemoryType(uint typeFilter, MemoryPropertyFlags propertyFlags)
        {
            var memProperties = physicalDevice.GetMemoryProperties();

            uint i = 0;
            foreach (var memType in memProperties.MemoryTypes)
            {
                if ((((typeFilter >> (int)i) & 1) == 1) && ((memType.PropertyFlags & propertyFlags) == propertyFlags))
                    return i;
                i++;
            }

            throw new Exception("failed to find memory type");
        }

        public (Vulkan.Buffer, DeviceMemory) CreateBuffer(DeviceSize size, BufferUsageFlags usageFlags, MemoryPropertyFlags flags)
        {
            Vulkan.Buffer buffer;
            DeviceMemory bufferMemory;

            var bufferInfo = new BufferCreateInfo
            {
                Size = size,
                Usage = usageFlags,
                SharingMode = SharingMode.Exclusive
            };

            buffer = device.CreateBuffer(bufferInfo);

            var memRequirements = device.GetBufferMemoryRequirements(buffer);

            var allocInfo = new MemoryAllocateInfo
            {
                AllocationSize = memRequirements.Size,
                MemoryTypeIndex = FindMemoryType(memRequirements.MemoryTypeBits, MemoryPropertyFlags.HostVisible | MemoryPropertyFlags.HostCoherent)
            };

            bufferMemory = device.AllocateMemory(allocInfo);
            device.BindBufferMemory(buffer, bufferMemory, 0);

            return (buffer, bufferMemory);
        }

        public CommandBuffer BeginSingleTimeCommands()
        {
            var allocInfo = new CommandBufferAllocateInfo
            {
                Level = CommandBufferLevel.Primary,
                CommandPool = commandPool,
                CommandBufferCount = 1
            };

            var cmdBuffer = device.AllocateCommandBuffers(allocInfo)[0];

            cmdBuffer.Begin(new CommandBufferBeginInfo { Flags = CommandBufferUsageFlags.OneTimeSubmit });

            return cmdBuffer;
        }

        public void EndSingleTimeCommands(CommandBuffer commandBuffer)
        {
            commandBuffer.End();

            var submitInfo = new SubmitInfo
            {
                CommandBuffers = new CommandBuffer[] { commandBuffer }
            };

            graphicQueue.Submit(submitInfo);

            graphicQueue.WaitIdle();

            device.FreeCommandBuffer(commandPool, commandBuffer);
        }

        public void CopyBuffer(Vulkan.Buffer srcBuffer, Vulkan.Buffer dstBuffer, DeviceSize size)
        {
            var cmdBuffer = BeginSingleTimeCommands();

            cmdBuffer.CmdCopyBuffer(srcBuffer, dstBuffer, new BufferCopy[] { new BufferCopy { Size = size } });

            EndSingleTimeCommands(cmdBuffer);
        }

        public void CreateIndexBuffer()
        {
            var bufferSize = indices.Length * sizeof(short);

            Vulkan.Buffer stagingBuffer;
            DeviceMemory stagingBufferMemory;
            (stagingBuffer, stagingBufferMemory) = CreateBuffer(bufferSize, BufferUsageFlags.TransferSrc, MemoryPropertyFlags.HostVisible | MemoryPropertyFlags.HostCoherent);

            var data = device.MapMemory(stagingBufferMemory, 0, bufferSize);
            System.Runtime.InteropServices.Marshal.Copy(indices, 0, data, indices.Length);
            device.UnmapMemory(stagingBufferMemory);

            (indexBuffer, indexBufferMemory) = CreateBuffer(bufferSize, BufferUsageFlags.TransferDst | BufferUsageFlags.IndexBuffer, MemoryPropertyFlags.DeviceLocal);

            CopyBuffer(stagingBuffer, indexBuffer, bufferSize);

            device.DestroyBuffer(stagingBuffer);
            device.FreeMemory(stagingBufferMemory);
        }

        public void CreateUniformBuffer()
        {
            var bufferSize = 3 * Marshal.SizeOf<Matrix4x4>();
            (uniformBuffer, uniformBufferMemory) = CreateBuffer(bufferSize, BufferUsageFlags.UniformBuffer, MemoryPropertyFlags.HostVisible | MemoryPropertyFlags.HostCoherent);
        }

        public void CreateDescriptorPool()
        {
            DescriptorPoolSize[] poolSize = new DescriptorPoolSize[]
            {
                new DescriptorPoolSize
                {
                    Type = DescriptorType.UniformBuffer,
                    DescriptorCount = 1
                },

                new DescriptorPoolSize
                {
                    Type = DescriptorType.CombinedImageSampler,
                    DescriptorCount = 1
                }
            };

            var poolInfo = new DescriptorPoolCreateInfo
            {
                PoolSizes = poolSize,
                MaxSets = 1
            };

            descriptorPool = device.CreateDescriptorPool(poolInfo);
        }

        public void CreateDescriptorSet()
        {
            DescriptorSetLayout[] layouts = new DescriptorSetLayout[] { descriptorSetLayout };
            var allocInfo = new DescriptorSetAllocateInfo
            {
                DescriptorPool = descriptorPool,
                SetLayouts = layouts
            };

            descriptorSet = device.AllocateDescriptorSets(allocInfo)[0];

            DescriptorBufferInfo[] bufferInfo = new DescriptorBufferInfo[]
            {
                new DescriptorBufferInfo
                {
                    Buffer = uniformBuffer,
                    Offset = 0,
                    Range = 3 * Marshal.SizeOf<Matrix4x4>()
                }
            };

            DescriptorImageInfo[] imageInfos = new DescriptorImageInfo[]
            {
                new DescriptorImageInfo
                {
                    ImageLayout = ImageLayout.ShaderReadOnlyOptimal,
                    ImageView = textureImageView,
                    Sampler = textureSampler
                }
            };

            WriteDescriptorSet[] descriptorWrites = new WriteDescriptorSet[]
            {
                new WriteDescriptorSet
                {
                    DstSet = descriptorSet,
                    DstBinding = 0,
                    DstArrayElement = 0,
                    DescriptorType = DescriptorType.UniformBuffer,
                    DescriptorCount = 1,
                    BufferInfo = bufferInfo
                },
                
                new WriteDescriptorSet
                {
                    DstSet = descriptorSet,
                    DstBinding = 1,
                    DstArrayElement = 0,
                    DescriptorType = DescriptorType.CombinedImageSampler,
                    DescriptorCount = 1,
                    ImageInfo = imageInfos
                }
            };

            device.UpdateDescriptorSets(descriptorWrites, null);
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

                cmdBuffer.CmdBindIndexBuffer(indexBuffer, 0, IndexType.Uint16);

                cmdBuffer.CmdBindDescriptorSet(PipelineBindPoint.Graphics, pipelineLayout, 0, descriptorSet, null);

                cmdBuffer.CmdDrawIndexed((uint)indices.Length, 1, 0, 0, 0);

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

        public void UpdateUniformBuffer()
        {
            currentTime = DateTime.Now;

            var time = (float)(currentTime - startTime).TotalSeconds;

            var arc = (float)((Math.PI / 180.0f));
            var fov = arc * 45.0f;
            var rotate = arc * 90.0f * time;
            UniformBufferObject ubo = new UniformBufferObject
            {
                model = Matrix4x4.CreateRotationZ(rotate),
                view = Matrix4x4.CreateLookAt(new Vector3(2.0f, 2.0f, 2.0f), new Vector3(0.0f, 0.0f, 0.0f), new Vector3(0.0f, 0.0f, 1.0f)),
                proj = Matrix4x4.CreatePerspectiveFieldOfView(fov, ((float)swapChainextent.Width / swapChainextent.Height), 0.1f, 10.0f)
            };
            ubo.proj.M11 *= -1.0f;
            ubo.proj.M22 *= -1.0f;

            var uboArray = ubo.ToArray();
            var data = device.MapMemory(uniformBufferMemory, 0, 3 * Marshal.SizeOf<Matrix4x4>());
            Marshal.Copy(uboArray, 0, data, uboArray.Length);
            device.UnmapMemory(uniformBufferMemory);
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

            var supportedFeatures = physicalDevice.GetFeatures();

            return indices.IsComplete && extensionSupported && swapChainAdequate && supportedFeatures.SamplerAnisotropy;
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
