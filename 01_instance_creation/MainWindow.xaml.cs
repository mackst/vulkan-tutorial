using System;
using System.Windows;
using Vulkan;
using Vulkan.Windows;



namespace _01_instance_creation
{
    /// <summary>
    /// MainWindow.xaml 的交互逻辑
    /// </summary>
    public partial class MainWindow : Window
    {
        private Vulkan.Instance instance;


        public MainWindow()
        {
            InitializeComponent();

            CreateInstance();
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

            var createInfo = new Vulkan.InstanceCreateInfo
            {
                ApplicationInfo = appInfo
            };

            instance = new Vulkan.Instance(createInfo);
        }

        private void Window_Closed(object sender, EventArgs e)
        {
            instance.Destroy();
            //Console.WriteLine("window closed instance destoryed.");
        }
    }
}
