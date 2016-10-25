/**************************************************************************************************
 *
 * GLFW V3 API
 *
 *   simplified version of glfw3.h
 *
 **************************************************************************************************/

#define VK_MAX_EXTENSION_NAME_SIZE        256

typedef uint32_t VkFlags;


typedef enum VkResult {
    VK_SUCCESS = 0,
    VK_NOT_READY = 1,
    VK_TIMEOUT = 2,
    VK_EVENT_SET = 3,
    VK_EVENT_RESET = 4,
    VK_INCOMPLETE = 5,
    VK_ERROR_OUT_OF_HOST_MEMORY = -1,
    VK_ERROR_OUT_OF_DEVICE_MEMORY = -2,
    VK_ERROR_INITIALIZATION_FAILED = -3,
    VK_ERROR_DEVICE_LOST = -4,
    VK_ERROR_MEMORY_MAP_FAILED = -5,
    VK_ERROR_LAYER_NOT_PRESENT = -6,
    VK_ERROR_EXTENSION_NOT_PRESENT = -7,
    VK_ERROR_FEATURE_NOT_PRESENT = -8,
    VK_ERROR_INCOMPATIBLE_DRIVER = -9,
    VK_ERROR_TOO_MANY_OBJECTS = -10,
    VK_ERROR_FORMAT_NOT_SUPPORTED = -11,
    VK_ERROR_SURFACE_LOST_KHR = -1000000000,
    VK_ERROR_NATIVE_WINDOW_IN_USE_KHR = -1000000001,
    VK_SUBOPTIMAL_KHR = 1000001003,
    VK_ERROR_OUT_OF_DATE_KHR = -1000001004,
    VK_ERROR_INCOMPATIBLE_DISPLAY_KHR = -1000003001,
    VK_ERROR_VALIDATION_FAILED_EXT = -1000011001,
    VK_ERROR_INVALID_SHADER_NV = -1000012000,
    VK_RESULT_BEGIN_RANGE = VK_ERROR_FORMAT_NOT_SUPPORTED,
    VK_RESULT_END_RANGE = VK_INCOMPLETE,
    VK_RESULT_RANGE_SIZE = 17,
    VK_RESULT_MAX_ENUM = 0x7FFFFFFF
} VkResult;

typedef enum VkSystemAllocationScope {
    VK_SYSTEM_ALLOCATION_SCOPE_COMMAND = 0,
    VK_SYSTEM_ALLOCATION_SCOPE_OBJECT = 1,
    VK_SYSTEM_ALLOCATION_SCOPE_CACHE = 2,
    VK_SYSTEM_ALLOCATION_SCOPE_DEVICE = 3,
    VK_SYSTEM_ALLOCATION_SCOPE_INSTANCE = 4,
    VK_SYSTEM_ALLOCATION_SCOPE_BEGIN_RANGE = VK_SYSTEM_ALLOCATION_SCOPE_COMMAND,
    VK_SYSTEM_ALLOCATION_SCOPE_END_RANGE = VK_SYSTEM_ALLOCATION_SCOPE_INSTANCE,
    VK_SYSTEM_ALLOCATION_SCOPE_RANGE_SIZE = 5,
    VK_SYSTEM_ALLOCATION_SCOPE_MAX_ENUM = 0x7FFFFFFF
} VkSystemAllocationScope;

typedef enum VkInternalAllocationType {
    VK_INTERNAL_ALLOCATION_TYPE_EXECUTABLE = 0,
    VK_INTERNAL_ALLOCATION_TYPE_BEGIN_RANGE = VK_INTERNAL_ALLOCATION_TYPE_EXECUTABLE,
    VK_INTERNAL_ALLOCATION_TYPE_END_RANGE = VK_INTERNAL_ALLOCATION_TYPE_EXECUTABLE,
    VK_INTERNAL_ALLOCATION_TYPE_RANGE_SIZE = 1,
    VK_INTERNAL_ALLOCATION_TYPE_MAX_ENUM = 0x7FFFFFFF
} VkInternalAllocationType;

typedef VkFlags VkStencilFaceFlags;

typedef void* (__stdcall *PFN_vkAllocationFunction)(
    void*                                       pUserData,
    size_t                                      size,
    size_t                                      alignment,
    VkSystemAllocationScope                     allocationScope);

typedef void* (__stdcall *PFN_vkReallocationFunction)(
    void*                                       pUserData,
    void*                                       pOriginal,
    size_t                                      size,
    size_t                                      alignment,
    VkSystemAllocationScope                     allocationScope);

typedef void (__stdcall *PFN_vkFreeFunction)(
    void*                                       pUserData,
    void*                                       pMemory);

typedef void (__stdcall *PFN_vkInternalAllocationNotification)(
    void*                                       pUserData,
    size_t                                      size,
    VkInternalAllocationType                    allocationType,
    VkSystemAllocationScope                     allocationScope);

typedef void (__stdcall *PFN_vkInternalFreeNotification)(
    void*                                       pUserData,
    size_t                                      size,
    VkInternalAllocationType                    allocationType,
    VkSystemAllocationScope                     allocationScope);

typedef struct VkAllocationCallbacks {
    void*                                   pUserData;
    PFN_vkAllocationFunction                pfnAllocation;
    PFN_vkReallocationFunction              pfnReallocation;
    PFN_vkFreeFunction                      pfnFree;
    PFN_vkInternalAllocationNotification    pfnInternalAllocation;
    PFN_vkInternalFreeNotification          pfnInternalFree;
} VkAllocationCallbacks;

typedef struct VkExtensionProperties {
    char        extensionName[VK_MAX_EXTENSION_NAME_SIZE];
    uint32_t    specVersion;
} VkExtensionProperties;

typedef struct VkInstance_T* VkInstance;
typedef struct VkSurfaceKHR_T* VkSurfaceKHR;
typedef struct VkPhysicalDevice_T* VkPhysicalDevice;

typedef struct GLFWmonitor GLFWmonitor;
typedef struct GLFWwindow GLFWwindow;

typedef struct GLFWvidmode
{
    int width;
    int height;
    int redBits;
    int greenBits;
    int blueBits;
    int refreshRate;
} GLFWvidmode;

typedef struct GLFWgammaramp
{
    unsigned short* red;
    unsigned short* green;
    unsigned short* blue;
    unsigned int size;
} GLFWgammaramp;

/**************************************************************************************************/

typedef void (*GLFWglproc)(void);
typedef void (*GLFWvkproc)(void);

typedef void (* GLFWerrorfun)(int,const char*);
typedef void (* GLFWwindowposfun)(GLFWwindow*,int,int);
typedef void (* GLFWwindowsizefun)(GLFWwindow*,int,int);
typedef void (* GLFWwindowclosefun)(GLFWwindow*);
typedef void (* GLFWwindowrefreshfun)(GLFWwindow*);
typedef void (* GLFWwindowfocusfun)(GLFWwindow*,int);
typedef void (* GLFWwindowiconifyfun)(GLFWwindow*,int);
typedef void (* GLFWframebuffersizefun)(GLFWwindow*,int,int);
typedef void (* GLFWmousebuttonfun)(GLFWwindow*,int,int,int);
typedef void (* GLFWcursorposfun)(GLFWwindow*,double,double);
typedef void (* GLFWcursorenterfun)(GLFWwindow*,int);
typedef void (* GLFWscrollfun)(GLFWwindow*,double,double);
typedef void (* GLFWkeyfun)(GLFWwindow*,int,int,int,int);
typedef void (* GLFWcharfun)(GLFWwindow*,unsigned int);
typedef void (* GLFWmonitorfun)(GLFWmonitor*,int);

/**************************************************************************************************/

int glfwInit(void);
void glfwTerminate(void);
void glfwGetVersion(int* major, int* minor, int* rev);
const char* glfwGetVersionString(void);
GLFWerrorfun glfwSetErrorCallback(GLFWerrorfun cbfun);
GLFWmonitor** glfwGetMonitors(int* count);
GLFWmonitor* glfwGetPrimaryMonitor(void);
void glfwGetMonitorPos(GLFWmonitor* monitor, int* xpos, int* ypos);
void glfwGetMonitorPhysicalSize(GLFWmonitor* monitor, int* width, int* height);
const char* glfwGetMonitorName(GLFWmonitor* monitor);
GLFWmonitorfun glfwSetMonitorCallback(GLFWmonitorfun cbfun);
const GLFWvidmode* glfwGetVideoModes(GLFWmonitor* monitor, int* count);
const GLFWvidmode* glfwGetVideoMode(GLFWmonitor* monitor);
void glfwSetGamma(GLFWmonitor* monitor, float gamma);
const GLFWgammaramp* glfwGetGammaRamp(GLFWmonitor* monitor);
void glfwSetGammaRamp(GLFWmonitor* monitor, const GLFWgammaramp* ramp);
void glfwDefaultWindowHints(void);
void glfwWindowHint(int target, int hint);
GLFWwindow* glfwCreateWindow(int width, int height, const char* title, GLFWmonitor* monitor, GLFWwindow* share);
void glfwDestroyWindow(GLFWwindow* window);
int glfwWindowShouldClose(GLFWwindow* window);
void glfwSetWindowShouldClose(GLFWwindow* window, int value);
void glfwSetWindowTitle(GLFWwindow* window, const char* title);
void glfwGetWindowPos(GLFWwindow* window, int* xpos, int* ypos);
void glfwSetWindowPos(GLFWwindow* window, int xpos, int ypos);
void glfwGetWindowSize(GLFWwindow* window, int* width, int* height);
void glfwSetWindowSize(GLFWwindow* window, int width, int height);
void glfwGetFramebufferSize(GLFWwindow* window, int* width, int* height);
void glfwIconifyWindow(GLFWwindow* window);
void glfwRestoreWindow(GLFWwindow* window);
void glfwShowWindow(GLFWwindow* window);
void glfwHideWindow(GLFWwindow* window);
GLFWmonitor* glfwGetWindowMonitor(GLFWwindow* window);
int glfwGetWindowAttrib(GLFWwindow* window, int attrib);
void glfwSetWindowUserPointer(GLFWwindow* window, void* pointer);
void* glfwGetWindowUserPointer(GLFWwindow* window);
GLFWwindowposfun glfwSetWindowPosCallback(GLFWwindow* window, GLFWwindowposfun cbfun);
GLFWwindowsizefun glfwSetWindowSizeCallback(GLFWwindow* window, GLFWwindowsizefun cbfun);
GLFWwindowclosefun glfwSetWindowCloseCallback(GLFWwindow* window, GLFWwindowclosefun cbfun);
GLFWwindowrefreshfun glfwSetWindowRefreshCallback(GLFWwindow* window, GLFWwindowrefreshfun cbfun);
GLFWwindowfocusfun glfwSetWindowFocusCallback(GLFWwindow* window, GLFWwindowfocusfun cbfun);
GLFWwindowiconifyfun glfwSetWindowIconifyCallback(GLFWwindow* window, GLFWwindowiconifyfun cbfun);
GLFWframebuffersizefun glfwSetFramebufferSizeCallback(GLFWwindow* window, GLFWframebuffersizefun cbfun);
void glfwPollEvents(void);
void glfwWaitEvents(void);
int glfwGetInputMode(GLFWwindow* window, int mode);
void glfwSetInputMode(GLFWwindow* window, int mode, int value);
int glfwGetKey(GLFWwindow* window, int key);
int glfwGetMouseButton(GLFWwindow* window, int button);
void glfwGetCursorPos(GLFWwindow* window, double* xpos, double* ypos);
void glfwSetCursorPos(GLFWwindow* window, double xpos, double ypos);
GLFWkeyfun glfwSetKeyCallback(GLFWwindow* window, GLFWkeyfun cbfun);
GLFWcharfun glfwSetCharCallback(GLFWwindow* window, GLFWcharfun cbfun);
GLFWmousebuttonfun glfwSetMouseButtonCallback(GLFWwindow* window, GLFWmousebuttonfun cbfun);
GLFWcursorposfun glfwSetCursorPosCallback(GLFWwindow* window, GLFWcursorposfun cbfun);
GLFWcursorenterfun glfwSetCursorEnterCallback(GLFWwindow* window, GLFWcursorenterfun cbfun);
GLFWscrollfun glfwSetScrollCallback(GLFWwindow* window, GLFWscrollfun cbfun);
int glfwJoystickPresent(int joy);
const float* glfwGetJoystickAxes(int joy, int* count);
const unsigned char* glfwGetJoystickButtons(int joy, int* count);
const char* glfwGetJoystickName(int joy);
void glfwSetClipboardString(GLFWwindow* window, const char* string);
const char* glfwGetClipboardString(GLFWwindow* window);
double glfwGetTime(void);
void glfwSetTime(double time);
void glfwMakeContextCurrent(GLFWwindow* window);
GLFWwindow* glfwGetCurrentContext(void);
void glfwSwapBuffers(GLFWwindow* window);
void glfwSwapInterval(int interval);
int glfwExtensionSupported(const char* extension);
GLFWglproc glfwGetProcAddress(const char* procname);

int glfwVulkanSupported(void);
VkResult glfwCreateWindowSurface(VkInstance instance, GLFWwindow* window, const VkAllocationCallbacks* allocator, VkSurfaceKHR* surface);
GLFWvkproc glfwGetInstanceProcAddress(VkInstance instance, const char* procname);
int glfwGetPhysicalDevicePresentationSupport(VkInstance instance, VkPhysicalDevice device, uint32_t queuefamily);
const char** glfwGetRequiredInstanceExtensions(uint32_t* count);

/**************************************************************************************************
 *
 * End
 *
 **************************************************************************************************/

typedef VkResult (__stdcall *PFN_vkEnumerateInstanceExtensionProperties)(const char* pLayerName, uint32_t* pPropertyCount, VkExtensionProperties* pProperties);
//VkResult __stdcall vkEnumerateInstanceExtensionProperties(
//    const char*                                 pLayerName,
//    uint32_t*                                   pPropertyCount,
//    VkExtensionProperties*                      pProperties);
//typedef VkResult (* vkEnumerateInstanceExtensionProperties)(
//    const char*                                 pLayerName,
//    uint32_t*                                   pPropertyCount,
//    VkExtensionProperties*                      pProperties);
