using System;
//using System.Collections;
using System.Collections.Generic;
using System.Windows;
using Vulkan;
using Vulkan.Windows;

namespace window_surface
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


    /// <summary>
    /// MainWindow.xaml 的交互逻辑
    /// </summary>
    public partial class MainWindow : Window
    {
        public bool enableValidationLayers = true;


        private Vulkan.Instance instance;
        private Vulkan.Instance.DebugReportCallback debugCallback;
        private Vulkan.PhysicalDevice physicalDevice = null;
        private Vulkan.Device device = null;
        private Vulkan.SurfaceKhr surface = null;

        private Vulkan.Queue graphicQueue;
        private Vulkan.Queue presentQueue;



        private String[] validationLayers = new String[] {
            "VK_LAYER_LUNARG_standard_validation"
        };


        public MainWindow()
        {
            InitializeComponent();

            CreateInstance();
            SetupDebugCallback();
            CreateSurface();
            PickPhysicalDevice();
            CreateLogicalDevice();
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

            List<String> extenstions = new List<String>();
            foreach (var extension in Vulkan.Commands.EnumerateInstanceExtensionProperties())
            {
                extenstions.Add(extension.ExtensionName);
            }
            var createInfo = new Vulkan.InstanceCreateInfo
            {
                ApplicationInfo = appInfo,
                EnabledExtensionNames = extenstions.ToArray(),
                EnabledLayerNames = validationLayers
            };

            instance = new Vulkan.Instance(createInfo);
        }

        public void SetupDebugCallback()
        {
            if (enableValidationLayers)
            {
                debugCallback = new Vulkan.Instance.DebugReportCallback(DebugReportCallback);
                instance.EnableDebug(debugCallback);
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

            float[] queuePrioriteise = { 1.0f };
            DeviceQueueCreateInfo[] queueCreateInfo = {
                new DeviceQueueCreateInfo
                {
                    QueueFamilyIndex = indices.GraphicsFamily,
                    QueuePriorities = queuePrioriteise
                },
            };

            var deviceFeatures = new Vulkan.PhysicalDeviceFeatures();
            var createInfo = new Vulkan.DeviceCreateInfo
            {
                QueueCreateInfos = queueCreateInfo,
                EnabledFeatures = deviceFeatures
            };

            if (enableValidationLayers)
            {
                createInfo.EnabledLayerNames = validationLayers;
            }

            device = physicalDevice.CreateDevice(createInfo);

            graphicQueue = device.GetQueue(indices.GraphicsFamily, 0);
            presentQueue = device.GetQueue(indices.PresentFamily, 0);
        }

        

        private bool IsDeviceSuitable(Vulkan.PhysicalDevice physicalDevice)
        {
            var indices = FindQueueFamilies(physicalDevice);

            return indices.IsComplete;
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

        private void Window_Closed(object sender, EventArgs e)
        {
            instance.DestroySurfaceKHR(surface);
            device.Destroy();

            instance.Destroy();
            //Console.WriteLine("window closed instance destoryed.");
        }


        static Bool32 DebugReportCallback(DebugReportFlagsExt flags, DebugReportObjectTypeExt objectType, ulong objectHandle, IntPtr location, int messageCode, IntPtr layerPrefix, IntPtr message, IntPtr userData)
        {
            string layerString = System.Runtime.InteropServices.Marshal.PtrToStringAnsi(layerPrefix);
            string messageString = System.Runtime.InteropServices.Marshal.PtrToStringAnsi(message);

            System.Console.WriteLine("DebugReport layer: {0} message: {1}", layerString, messageString);

            return false;
        }
    }
}
