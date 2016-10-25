####################################################################################################
#
# pyglfw-cffi - A Python Wrapper for GLFW.
# Copyright (C) 2014 Fabrice Salvaire
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
####################################################################################################

####################################################################################################

import six
from six.moves import xrange

####################################################################################################

import os as _os
import re
import inspect

from cffi import FFI as _FFI

####################################################################################################

from ctypes.util import find_library as _find_library
# from util import _find_library

from .Constantes import *

####################################################################################################
__currentDir = _os.path.abspath(inspect.getframeinfo(inspect.currentframe()).filename)
__currentDir = _os.path.dirname(__currentDir)
_ffi = _FFI()
_glfw = None

# Use a function in order to don't spoil the module
def _init():
    glfw_library = None
    glfw_library = _os.path.join(__currentDir, 'glfw3.dll')
    
    # First if there is an environment variable pointing to the library
    if 'GLFW_LIBRARY' in _os.environ:
        library_path = _os.path.realpath(_os.environ['GLFW_LIBRARY'])
        if _os.path.exists(library_path):
            glfw_library = library_path
    
    # Else, try to find it
    if glfw_library is None:
        ordered_library_names = ['glfw', 'glfw3']
        for library_name in ordered_library_names:
            glfw_library = _find_library(library_name)
            if glfw_library is not None:
                break
    
    # Else, we failed and exit
    if glfw_library is None:
        raise OSError('GLFW library not found')
    
    # Parse header
    #api_path = _os.path.join(_os.path.dirname(__file__), 'glfw3-api.h')
    api_path = _os.path.join(__currentDir, 'glfw3-api.h')
    with open(api_path, 'r') as f:
        source = f.read()
    _ffi.cdef(source)
    
    global _glfw
    _glfw = _ffi.dlopen(glfw_library)
    
    if False:
        # Dump the functions
        function_re = re.compile('^.* glfw([A-Za-z]*)\(.*;$')
        camel_case_re = re.compile('(?!^)([A-Z]+)')
        for line in source.split('\n'):
            match = function_re.match(line)
            if match:
                camel_case_function = match.group(1)
                function = camel_case_re.sub( r'_\1', camel_case_function).lower()
                six.print_('{} = _glfw.glfw{}'.format(function, camel_case_function))

_init()

####################################################################################################
#
# Define some helper to wrap reference and structure return.
#

def _reference_wrapper(func, c_type, number_of_items):
    def wrapper(*args):
        references = [_ffi.new(c_type + '*') for i in range(number_of_items)]
        args = list(args) + references
        func(*args)
        return [reference[0] for reference in references]
    return wrapper

def _wrap_2_int(func):
    return _reference_wrapper(func, 'int', 2)

def _wrap_3_int(func):
    return _reference_wrapper(func, 'int', 3)

def _wrap_2_double(func):
    return _reference_wrapper(func, 'double', 2)

def _list_wrapper(func):
    def wrapper(*args):
        count = _ffi.new('int*')
        args = list(args) + [count]
        pointer = func(*args)
        return pointer[0:count[0]]
    return wrapper

####################################################################################################

# Fixme: how to do this automatically ?

# create_window = _glfw.glfwCreateWindow
default_window_hints = _glfw.glfwDefaultWindowHints
destroy_window = _glfw.glfwDestroyWindow
extension_supported = _glfw.glfwExtensionSupported
get_clipboard_string = _glfw.glfwGetClipboardString
get_current_context = _glfw.glfwGetCurrentContext
get_cursor_pos = _wrap_2_double(_glfw.glfwGetCursorPos)
get_framebuffer_size = _wrap_2_int(_glfw.glfwGetFramebufferSize)
get_gamma_ramp = _glfw.glfwGetGammaRamp
get_input_mode = _glfw.glfwGetInputMode
get_joystick_axes = _list_wrapper(_glfw.glfwGetJoystickAxes)
get_joystick_buttons = _list_wrapper(_glfw.glfwGetJoystickButtons)
get_joystick_name = _glfw.glfwGetJoystickName
get_key = _glfw.glfwGetKey
get_monitor_name = _glfw.glfwGetMonitorName
get_monitor_physical_size = _wrap_2_int(_glfw.glfwGetMonitorPhysicalSize)
get_monitor_pos = _wrap_2_int(_glfw.glfwGetMonitorPos)
get_monitors = _list_wrapper(_glfw.glfwGetMonitors)
get_mouse_button = _glfw.glfwGetMouseButton
get_primary_monitor = _glfw.glfwGetPrimaryMonitor
get_proc_address = _glfw.glfwGetProcAddress
get_time = _glfw.glfwGetTime
get_version = _wrap_3_int(_glfw.glfwGetVersion)
get_version_string = _glfw.glfwGetVersionString
get_video_mode = _glfw.glfwGetVideoMode
get_video_modes = _list_wrapper(_glfw.glfwGetVideoModes)
get_window_attrib = _glfw.glfwGetWindowAttrib
get_window_monitor = _glfw.glfwGetWindowMonitor
get_window_pos = _wrap_2_int(_glfw.glfwGetWindowPos)
get_window_size = _wrap_2_int(_glfw.glfwGetWindowSize)
get_window_user_pointer = _glfw.glfwGetWindowUserPointer
hide_window = _glfw.glfwHideWindow
iconify_window = _glfw.glfwIconifyWindow
init = _glfw.glfwInit
joystick_present = _glfw.glfwJoystickPresent
make_context_current = _glfw.glfwMakeContextCurrent
poll_events = _glfw.glfwPollEvents
restore_window = _glfw.glfwRestoreWindow
set_char_callback = _glfw.glfwSetCharCallback
set_clipboard_string = _glfw.glfwSetClipboardString
set_cursor_enter_callback = _glfw.glfwSetCursorEnterCallback
set_cursor_pos_callback = _glfw.glfwSetCursorPosCallback
set_cursor_pos = _glfw.glfwSetCursorPos
# set_error_callback = _glfw.glfwSetErrorCallback
set_framebuffer_size_callback = _glfw.glfwSetFramebufferSizeCallback
set_gamma = _glfw.glfwSetGamma
set_gamma_ramp = _glfw.glfwSetGammaRamp
set_input_mode = _glfw.glfwSetInputMode
set_key_callback = _glfw.glfwSetKeyCallback
set_monitor_callback = _glfw.glfwSetMonitorCallback
set_mouse_button_callback = _glfw.glfwSetMouseButtonCallback
set_scroll_callback = _glfw.glfwSetScrollCallback
set_time = _glfw.glfwSetTime
set_window_close_callback = _glfw.glfwSetWindowCloseCallback
set_window_focus_callback = _glfw.glfwSetWindowFocusCallback
set_window_iconify_callback = _glfw.glfwSetWindowIconifyCallback
set_window_pos_callback = _glfw.glfwSetWindowPosCallback
set_window_pos = _glfw.glfwSetWindowPos
set_window_refresh_callback = _glfw.glfwSetWindowRefreshCallback
set_window_should_close = _glfw.glfwSetWindowShouldClose
set_window_size_callback = _glfw.glfwSetWindowSizeCallback
set_window_size = _glfw.glfwSetWindowSize
set_window_title = _glfw.glfwSetWindowTitle
# set_window_user_pointer = _glfw.glfwSetWindowUserPointer # void*
show_window = _glfw.glfwShowWindow
swap_buffers = _glfw.glfwSwapBuffers
swap_interval = _glfw.glfwSwapInterval
terminate = _glfw.glfwTerminate
wait_events = _glfw.glfwWaitEvents
window_hint = _glfw.glfwWindowHint
window_should_close = _glfw.glfwWindowShouldClose

vulkanSupported = _glfw.glfwVulkanSupported
# createWindowSurface = _glfw.glfwCreateWindowSurface
getInstanceProcAddress = _glfw.glfwGetInstanceProcAddress
# getPhysicalDevicePresentationSupport = _glfw.glfwGetPhysicalDevicePresentationSupport
# getRequiredInstanceExtensions = _glfw.glfwGetRequiredInstanceExtensions

def createWindowSurface(instance, window, x=None):
    instance = _ffi.cast('VkInstance', instance)
    surface = _ffi.new('VkSurfaceKHR *')
    _glfw.glfwCreateWindowSurface(instance, window, _ffi.NULL, surface)
    return surface[0]


def getPhysicalDevicePresentationSupport(instance, device, queuefamily):
    instance = _ffi.cast('VkInstance', instance)
    device = _ffi.cast('VkPhysicalDevice', device)
    return _glfw.glfwGetPhysicalDevicePresentationSupport(instance, device, queuefamily)

def getRequiredInstanceExtensions():
    count = _ffi.new('uint32_t *')
    extensions = _glfw.glfwGetRequiredInstanceExtensions(count)
    return (extensions, count)


####################################################################################################
##  vulkan funcitons
vkCreateInstance = getInstanceProcAddress(_ffi.NULL, _ffi.new('char[]', 'vkCreateInstance'))
vkEnumerateInstanceExtensionProperties = getInstanceProcAddress(_ffi.NULL, _ffi.new('char[]', 'vkEnumerateInstanceExtensionProperties'))
vkEnumerateInstanceLayerProperties = getInstanceProcAddress(_ffi.NULL, _ffi.new('char[]', 'vkEnumerateInstanceLayerProperties'))



####################################################################################################

def create_window(width=640, height=480, title="GLFW Window", monitor=_ffi.NULL, share=_ffi.NULL):
    # if monitor is None:
    #     monitor = _ffi.NULL
    # monitor = ffi.cast('void *', monitor)
    return _glfw.glfwCreateWindow(width, height, title.encode('utf-8'), monitor, share)

####################################################################################################
#
# Callback decorators
#

char_callback = _ffi.callback('void (GLFWwindow*, unsigned int)')
cursor_enter_callback = _ffi.callback('void (GLFWwindow*, int)')
cursor_pos_callback = _ffi.callback('void (GLFWwindow*, double, double)')
error_callback = _ffi.callback('void (int, const char*)')
frame_buffersize_callback = _ffi.callback('void (GLFWwindow*, int, int)')
key_callback = _ffi.callback('void (GLFWwindow*, int, int, int, int)')
monitor_callback = _ffi.callback('void (GLFWmonitor*, int)')
mouse_button_callback = _ffi.callback('void (GLFWwindow*, int, int, int)')
scroll_callback = _ffi.callback('void (GLFWwindow*, double, double)')
window_close_callback = _ffi.callback('void (GLFWwindow*)')
window_focus_callback = _ffi.callback('void (GLFWwindow*, int)')
window_iconify_callback = _ffi.callback('void (GLFWwindow*, int)')
window_pos_callback = _ffi.callback('void (GLFWwindow*, int, int)')
window_refresh_callback = _ffi.callback('void (GLFWwindow*)')
window_size_callback = _ffi.callback('void (GLFWwindow*, int, int)')

####################################################################################################

_error_callback_wrapper = None

def set_error_callback(func):
    @error_callback
    def wrapper(error, description):
        return func(error, _ffi.string(description))
    global _error_callback_wrapper
    _error_callback_wrapper = wrapper
    _glfw.glfwSetErrorCallback(wrapper)

####################################################################################################
#
# End
#
####################################################################################################
