using System;
//using System.Collections;
using System.Collections.Generic;
using System.Windows;
using Vulkan;
using Vulkan.Windows;



namespace validation_layers
{
    /// <summary>
    /// MainWindow.xaml 的交互逻辑
    /// </summary>
    public partial class MainWindow : Window
    {
        private Vulkan.Instance instance;
        private Vulkan.Instance.DebugReportCallback debugCallback;


        private String[] validationLayers = new String[] {
            "VK_LAYER_LUNARG_standard_validation"
        };


        public bool enableValidationLayers = true;


        public MainWindow()
        {
            InitializeComponent();

            CreateInstance();
            SetupDebugCallback();
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
            foreach(var extension in Vulkan.Commands.EnumerateInstanceExtensionProperties())
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

        private void Window_Closed(object sender, EventArgs e)
        {
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
