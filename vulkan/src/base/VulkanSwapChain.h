//
// Created by slimakanzer on 19.03.19.
//

#ifndef VULKAN_VULKANSWAPCHAIN_H
#define VULKAN_VULKANSWAPCHAIN_H

#include "vulkan.h"
#include "VulkanDevice.h"
#include <vector>
#include <assert.h>
#include <stdexcept>

class VulkanSwapChain {
private:
    VkSurfaceFormatKHR findSupportedFormat(VkSurfaceFormatKHR requestedFormat){
        for (auto format : supportFormats) {
            if (requestedFormat.format == format.format && requestedFormat.colorSpace == format.colorSpace){
                return requestedFormat;
            }
        }
        throw new std::runtime_error("Can't find supported format");
    }

    VkPresentModeKHR findSupportedPresentMode(VkPresentModeKHR requestedMode){
        for (auto presentMode: supportPresentMode){
            if (requestedMode == presentMode){
                return requestedMode;
            }
        }
        throw new std::runtime_error("Can't find supported present mode");
    }

public:
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VulkanDevice* device = nullptr;

    VkSurfaceCapabilitiesKHR supportCapabilities;
    std::vector<VkSurfaceFormatKHR> supportFormats;
    std::vector<VkPresentModeKHR> supportPresentMode;
    VkSurfaceFormatKHR enableFormat;
    VkPresentModeKHR enablePresentMode;
    VkExtent2D currentExtent;
    std::vector<VkImage> images;
    std::vector<VkImageView> imageViews;

    VulkanSwapChain(VkSurfaceKHR surfaceKHR, VulkanDevice* device){
        assert(surfaceKHR);
        assert(device);
        this->surface = surfaceKHR;
        this->device = device;

        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device->physicalDevice, surfaceKHR, &supportCapabilities);

        uint32_t formatsCount = 0;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device->physicalDevice, surfaceKHR, &formatsCount, nullptr);
        supportFormats.resize(formatsCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(device->physicalDevice, surfaceKHR, &formatsCount, supportFormats.data());

        uint32_t presentModeCount = 0;
        vkGetPhysicalDeviceSurfacePresentModesKHR(device->physicalDevice, surfaceKHR, &presentModeCount, nullptr);
        supportPresentMode.resize(presentModeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(device->physicalDevice, surfaceKHR, &presentModeCount, supportPresentMode.data());
    }

    void createSwapChain(uint32_t minImageCount,
            VkSurfaceFormatKHR enableSurfaceFormat,
            VkPresentModeKHR enableSurfacePresentMode,
            VkExtent2D enableSurfaceExtent,
            uint32_t imageArrayLayers = 1,
            VkImageUsageFlags usageFlags = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
            VkCompositeAlphaFlagBitsKHR compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR){
        assert(minImageCount > 0 && minImageCount <= supportCapabilities.maxImageCount);

        this->enableFormat = findSupportedFormat(enableSurfaceFormat);
        this->enablePresentMode = findSupportedPresentMode(enableSurfacePresentMode);

        VkSwapchainCreateInfoKHR swapchainCreateInfoKHR = {};
        swapchainCreateInfoKHR.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        swapchainCreateInfoKHR.surface = surface;
        swapchainCreateInfoKHR.minImageCount = minImageCount;
        swapchainCreateInfoKHR.imageFormat = enableFormat.format;
        swapchainCreateInfoKHR.imageColorSpace = enableFormat.colorSpace;
        swapchainCreateInfoKHR.presentMode = enablePresentMode;

        // TODO сделать валидацию extent (not only currentExtent)
        currentExtent = supportCapabilities.currentExtent;
        swapchainCreateInfoKHR.imageExtent = supportCapabilities.currentExtent;
        swapchainCreateInfoKHR.imageArrayLayers = imageArrayLayers;
        swapchainCreateInfoKHR.imageUsage = usageFlags;
        swapchainCreateInfoKHR.preTransform = supportCapabilities.currentTransform;
        swapchainCreateInfoKHR.compositeAlpha = compositeAlpha;

        uint32_t queueFamilyIndeces[] = { device->queueFamilyIndices.graphics, device->queueFamilyIndices.compute, device->queueFamilyIndices.transfer };

        if (device->queueFamilyIndices.graphics != device->queueFamilyIndices.compute){
            swapchainCreateInfoKHR.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            swapchainCreateInfoKHR.queueFamilyIndexCount = 3;
            swapchainCreateInfoKHR.pQueueFamilyIndices = queueFamilyIndeces;
        }else {
            swapchainCreateInfoKHR.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
            swapchainCreateInfoKHR.queueFamilyIndexCount = 1;
            swapchainCreateInfoKHR.pQueueFamilyIndices = queueFamilyIndeces;
        }
        swapchainCreateInfoKHR.clipped = VK_TRUE;
        swapchainCreateInfoKHR.oldSwapchain = VK_NULL_HANDLE;

        VkBool32 isSupport;

        auto resultSupporting = vkGetPhysicalDeviceSurfaceSupportKHR(device->physicalDevice, device->queueFamilyIndices.graphics, surface, &isSupport);
        if (resultSupporting != VK_SUCCESS){
            throw new std::runtime_error("Can't check supporting device");
        }

        if (isSupport != VK_TRUE){
            throw new std::runtime_error("This device didn't support this surface");
        }

        auto result = vkCreateSwapchainKHR(device->logicalDevice, &swapchainCreateInfoKHR, nullptr, &swapchain);
        if (result != VK_SUCCESS){
            throw new std::runtime_error("Failed create swapchain");
        }

        images.resize(minImageCount);

        auto imagesResult = vkGetSwapchainImagesKHR(device->logicalDevice, swapchain, &minImageCount, images.data());
        if (imagesResult != VK_SUCCESS){
            throw new std::runtime_error("Can't read list of images");
        }

        for (auto image : images){
            VkImageViewCreateInfo imageViewCreateInfo = {};
            imageViewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            imageViewCreateInfo.format = enableFormat.format;
            imageViewCreateInfo.image = image;
            imageViewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            imageViewCreateInfo.components.r = VK_COMPONENT_SWIZZLE_R;
            imageViewCreateInfo.components.g = VK_COMPONENT_SWIZZLE_G;
            imageViewCreateInfo.components.b = VK_COMPONENT_SWIZZLE_B;
            imageViewCreateInfo.components.a = VK_COMPONENT_SWIZZLE_A;
            imageViewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            imageViewCreateInfo.subresourceRange.baseMipLevel = 0;
            imageViewCreateInfo.subresourceRange.levelCount = 1;
            imageViewCreateInfo.subresourceRange.baseArrayLayer = 0;
            imageViewCreateInfo.subresourceRange.layerCount = 1;

            VkImageView imageView = VK_NULL_HANDLE;
            auto createImageView = vkCreateImageView(device->logicalDevice, &imageViewCreateInfo, nullptr, &imageView);

            if (createImageView != VK_SUCCESS){
                __throw_exception_again;
            }
            imageViews.push_back(imageView);
        }
    }
};


#endif //VULKAN_VULKANSWAPCHAIN_H
