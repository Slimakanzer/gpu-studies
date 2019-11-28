//
// Created by slimakanzer on 25.03.19.
//

#ifndef VULKAN_VULKANTOOLS_H
#define VULKAN_VULKANTOOLS_H

#include "vulkan.h"
#include <stdexcept>
#include <iostream>
#include <vector>
#include <fstream>

#define VK_CHECK_RESULT(res, msg)\
{\
    VkResult result = (res);\
    if (result != VK_SUCCESS)\
        throw std::runtime_error(msg);\
}

namespace shaders
{
    std::vector<char> readShaderFile(const std::string& filename);
    VkShaderModule createShaderModule(const VkDevice device, const std::vector<char>& code);
}

namespace debug
{
    const std::vector<const char*> validationLayers = {
            "VK_LAYER_LUNARG_standard_validation"
    };

    const std::vector<const char*> validationExtensions = {
            VK_EXT_DEBUG_UTILS_EXTENSION_NAME
    };

    void setupDebugMessenger(VkInstance* instance);
    void DestroyDebugUtilsMessengerEXT(VkInstance* instance,  const VkAllocationCallbacks* pAllocator);
}

#endif //VULKAN_VULKANTOOLS_H
