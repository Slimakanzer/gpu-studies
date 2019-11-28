//
// Created by slimakanzer on 15.03.19.
//

#ifndef VULKAN_VULKANDEVICE_H
#define VULKAN_VULKANDEVICE_H
#include "vulkan.h"
#include <vector>
#include <assert.h>
#include <string>
#include <stdexcept>

class VulkanDevice {
public:
    VkDevice  logicalDevice = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties properties;
    VkPhysicalDeviceFeatures availableFeatures;
    VkPhysicalDeviceMemoryProperties memoryProperties;
    std::vector<VkQueueFamilyProperties> availableFamilyProperties;
    std::vector<std::string> availableExtensionNames;
    std::vector<const char*> enableLayerNames;
    std::vector<const char*> enableExtensionNames;
    VkPhysicalDeviceFeatures enableFeatures;
    VkCommandPool commandPool = VK_NULL_HANDLE;

    struct {
        uint32_t graphics;
        uint32_t compute;
        uint32_t transfer;
    } queueFamilyIndices;

    VulkanDevice(VkPhysicalDevice vkPhysicalDevice){
        assert(vkPhysicalDevice);
        physicalDevice = vkPhysicalDevice;
        vkGetPhysicalDeviceProperties(vkPhysicalDevice, &properties);
        vkGetPhysicalDeviceFeatures(vkPhysicalDevice, &availableFeatures);
        vkGetPhysicalDeviceMemoryProperties(vkPhysicalDevice, &memoryProperties);

        uint32_t familyProps;
        vkGetPhysicalDeviceQueueFamilyProperties(vkPhysicalDevice, &familyProps, nullptr);
        assert(familyProps > 0);
        availableFamilyProperties.resize(familyProps);
        vkGetPhysicalDeviceQueueFamilyProperties(vkPhysicalDevice, &familyProps, availableFamilyProperties.data());

        uint32_t extensionCount;
        std::vector<VkExtensionProperties> extensionProperties;
        vkEnumerateDeviceExtensionProperties(vkPhysicalDevice, nullptr, &extensionCount, nullptr);
        extensionProperties.resize(extensionCount);
        vkEnumerateDeviceExtensionProperties(vkPhysicalDevice, nullptr, &extensionCount, extensionProperties.data());
        for (auto extension : extensionProperties){
            availableExtensionNames.push_back(extension.extensionName);
        }
    }

    ~VulkanDevice(){
            if (logicalDevice) vkDestroyDevice(logicalDevice, nullptr);
    }



    void createLogicalDevice(std::vector<const char*>* enableLayerNames, std::vector<const char*>* enableExtensionNames, VkPhysicalDeviceFeatures* enableFeatures, VkQueueFlags requestedQueueTypes = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT){
            VkDeviceCreateInfo deviceCreateInfo = {};
            deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

            if (enableFeatures == nullptr){
                    deviceCreateInfo.pEnabledFeatures = &availableFeatures;
                    this->enableFeatures = availableFeatures;
            }else {
                    deviceCreateInfo.pEnabledFeatures = enableFeatures;
                    this->enableFeatures = *enableFeatures;
            }

            if (enableLayerNames == nullptr){
                    deviceCreateInfo.enabledLayerCount = 0;
            } else {
                    deviceCreateInfo.enabledLayerCount = static_cast<uint32_t>(enableLayerNames->size());
                    deviceCreateInfo.ppEnabledLayerNames = enableLayerNames->data();
                    this->enableLayerNames = *enableLayerNames;
            }

            if (enableExtensionNames == nullptr){
                    deviceCreateInfo.enabledExtensionCount = 0;
            }else {
                    deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(enableExtensionNames->size());
                    deviceCreateInfo.ppEnabledExtensionNames = enableExtensionNames->data();
                    this->enableExtensionNames = *enableExtensionNames;
            }

            std::vector<VkDeviceQueueCreateInfo> queueCreateInfos{};

            const float defaultQueuePriority(0.0f);

            if (requestedQueueTypes & VK_QUEUE_GRAPHICS_BIT)
            {
                    queueFamilyIndices.graphics = getQueueFamilyIndex(VK_QUEUE_GRAPHICS_BIT);
                    VkDeviceQueueCreateInfo queueInfo{};
                    queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
                    queueInfo.queueFamilyIndex = queueFamilyIndices.graphics;
                    queueInfo.queueCount = 1;
                    queueInfo.pQueuePriorities = &defaultQueuePriority;
                    queueCreateInfos.push_back(queueInfo);
            }
            else
            {
                    queueFamilyIndices.graphics = VK_NULL_HANDLE;
            }

            if (requestedQueueTypes & VK_QUEUE_COMPUTE_BIT)
            {
                    queueFamilyIndices.compute = getQueueFamilyIndex(VK_QUEUE_COMPUTE_BIT);
                    if (queueFamilyIndices.compute != queueFamilyIndices.graphics)
                    {
                            VkDeviceQueueCreateInfo queueInfo{};
                            queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
                            queueInfo.queueFamilyIndex = queueFamilyIndices.compute;
                            queueInfo.queueCount = 1;
                            queueInfo.pQueuePriorities = &defaultQueuePriority;
                            queueCreateInfos.push_back(queueInfo);
                    }
            }
            else
            {
                    queueFamilyIndices.compute = queueFamilyIndices.graphics;
            }

            if (requestedQueueTypes & VK_QUEUE_TRANSFER_BIT)
            {
                    queueFamilyIndices.transfer = getQueueFamilyIndex(VK_QUEUE_TRANSFER_BIT);
                    if ((queueFamilyIndices.transfer != queueFamilyIndices.graphics) && (queueFamilyIndices.transfer != queueFamilyIndices.compute))
                    {
                            VkDeviceQueueCreateInfo queueInfo{};
                            queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
                            queueInfo.queueFamilyIndex = queueFamilyIndices.transfer;
                            queueInfo.queueCount = 1;
                            queueInfo.pQueuePriorities = &defaultQueuePriority;
                            queueCreateInfos.push_back(queueInfo);
                    }
            }
            else
            {
                    queueFamilyIndices.transfer = queueFamilyIndices.graphics;
            }

            deviceCreateInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
            deviceCreateInfo.pQueueCreateInfos = queueCreateInfos.data();


            if ( vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &logicalDevice) != VK_SUCCESS ){
                    throw std::runtime_error("Error create logical device");
            }
    }

    void createCommandPool(){
        VkCommandPoolCreateInfo commandPoolCreateInfo = {};
        commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        commandPoolCreateInfo.queueFamilyIndex = queueFamilyIndices.graphics;
        auto result = vkCreateCommandPool(this->logicalDevice, &commandPoolCreateInfo, nullptr, &commandPool);

        if (result != VK_SUCCESS){
            throw new std::runtime_error("Error with create command pool");
        }
    }

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {

        for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && (memoryProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }

        throw std::runtime_error("failed to find suitable memory type!");
    }

private:

    uint32_t getQueueFamilyIndex(VkQueueFlagBits queueFlags)
    {
            if (queueFlags & VK_QUEUE_COMPUTE_BIT)
            {
                    for (uint32_t i = 0; i < static_cast<uint32_t>(availableFamilyProperties.size()); i++)
                    {
                            if ((availableFamilyProperties[i].queueFlags & queueFlags) && ((availableFamilyProperties[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0))
                            {
                                    return i;
                            }
                    }
            }

            if (queueFlags & VK_QUEUE_TRANSFER_BIT)
            {
                    for (uint32_t i = 0; i < static_cast<uint32_t>(availableFamilyProperties.size()); i++)
                    {
                            if ((availableFamilyProperties[i].queueFlags & queueFlags) && ((availableFamilyProperties[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0) && ((availableFamilyProperties[i].queueFlags & VK_QUEUE_COMPUTE_BIT) == 0))
                            {
                                    return i;
                            }
                    }
            }

            for (uint32_t i = 0; i < static_cast<uint32_t>(availableFamilyProperties.size()); i++)
            {
                    if (availableFamilyProperties[i].queueFlags & queueFlags)
                    {
                            return i;
                    }
            }

            throw std::runtime_error("Could not find a matching queue family index");
    }
};


#endif //VULKAN_VULKANDEVICE_H
