#include "vulkan.h"

#include <glm/glm.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/string_cast.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <GLFW/glfw3.h>
#include <array>
#include "base/VulkanDevice.h"
#include "base/VulkanSwapChain.h"
#include "base/VulkanTools.h"

#define VERT_FILE_PATH "./shaders/bin/vert.spv"
#define FRAG_FILE_PATH "./shaders/bin/frag.spv"

const int WIDTH = 800;
const int HEIGHT = 600;

struct Vertex {
    glm::vec2 pos;
    glm::vec3 color;

    static VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription bindingDescription = {};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        return bindingDescription;
    }

    static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions() {
        std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions = {};

        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(Vertex, pos);

        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(Vertex, color);

        return attributeDescriptions;
    }
};

struct MVP {
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
};

const std::vector<Vertex> vertices = {
        {{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
        {{0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
        {{0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
        {{-0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}}
};

const std::vector<uint16_t > indeces = {
        0, 1, 2, 2, 3, 0
};




class App {
    GLFWwindow* window;
    VkInstance instance = VK_NULL_HANDLE;
    VulkanDevice* device = nullptr;
    VulkanSwapChain* swapChain = nullptr;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline graphicsPipeline = VK_NULL_HANDLE;
    VkRenderPass renderPass = VK_NULL_HANDLE;
    std::vector<VkFramebuffer> framebuffers;
    std::vector<VkCommandBuffer> commandBuffers;
    VkSemaphore renderFinished = VK_NULL_HANDLE;
    VkSemaphore imageReady = VK_NULL_HANDLE;
    VkDeviceMemory vertexBufferMemory = VK_NULL_HANDLE;
    VkBuffer vertexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory indexBufferMemory = VK_NULL_HANDLE;
    VkBuffer indexBuffer = VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> descriptorSets;

    std::vector<VkBuffer> uniformBuffers;
    std::vector<VkDeviceMemory> uniformBufferMemories;

    void loop(){
        while (!glfwWindowShouldClose(window)){
            glfwPollEvents();

            uint32_t imageIndex;
            vkAcquireNextImageKHR(device->logicalDevice, swapChain->swapchain, std::numeric_limits<uint64_t>::max(), imageReady, VK_NULL_HANDLE, &imageIndex);

            // -------------------- UPDATE UNIFORM BUFFER MVP -------------------------------//
            MVP mvp = {};
            static auto startTime = glfwGetTime();
            auto currentTime = glfwGetTime();
            auto dT = currentTime - startTime;

            mvp.model = glm::mat4(1);
            mvp.view = glm::lookAt(
                    glm::vec3(1, 1, 1),
                    glm::vec3(0,0,0),
                    glm::vec3(0,1,0));
            mvp.proj = glm::perspective(glm::radians(45.0f), swapChain->currentExtent.width / (float) swapChain->currentExtent.height, 0.1f, 10.0f);

            void* data;
            vkMapMemory(device->logicalDevice, uniformBufferMemories[imageIndex], 0, sizeof(mvp), 0, &data);
            memcpy(data, &mvp, sizeof(mvp));
            vkUnmapMemory(device->logicalDevice, uniformBufferMemories[imageIndex]);
            //--------------------- END OF UPDATE UNIFORM BUFFER ----------------------------//


            VkSubmitInfo submitInfo = {};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

            VkSemaphore waitSemaphores[] = {imageReady};
            VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
            submitInfo.waitSemaphoreCount = 1;
            submitInfo.pWaitSemaphores = waitSemaphores;
            submitInfo.pWaitDstStageMask = waitStages;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &commandBuffers[imageIndex];

            VkSemaphore signalSemaphores[] = {renderFinished};
            submitInfo.signalSemaphoreCount = 1;
            submitInfo.pSignalSemaphores = signalSemaphores;

            VkQueue graphicsQueue;
            vkGetDeviceQueue(device->logicalDevice, device->queueFamilyIndices.graphics, 0, &graphicsQueue);

            VkQueue presentQueue;
            vkGetDeviceQueue(device->logicalDevice, device->queueFamilyIndices.graphics, 0, &presentQueue);

            if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
                throw std::runtime_error("failed to submit draw command buffer!");
            }

            VkPresentInfoKHR presentInfo = {};
            presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

            presentInfo.waitSemaphoreCount = 1;
            presentInfo.pWaitSemaphores = signalSemaphores;

            VkSwapchainKHR swapChains[] = {swapChain->swapchain};
            presentInfo.swapchainCount = 1;
            presentInfo.pSwapchains = swapChains;
            presentInfo.pImageIndices = &imageIndex;
            presentInfo.pResults = nullptr;
            vkQueuePresentKHR(presentQueue, &presentInfo);

            vkQueueWaitIdle(presentQueue);
        }
    }

public:
    void run(){
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        window = glfwCreateWindow(WIDTH, HEIGHT, "GLarochkin", nullptr, nullptr);

        uint32_t glfwExtensionsCount = 0;
        const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionsCount);

        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionsCount);
        extensions.insert(extensions.begin(), debug::validationExtensions.begin(), debug::validationExtensions.end());

        VkInstanceCreateInfo instanceCreateInfo = {};
        instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        instanceCreateInfo.enabledLayerCount = static_cast<uint32_t>(debug::validationLayers.size());
        instanceCreateInfo.ppEnabledLayerNames = debug::validationLayers.data();
        instanceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        instanceCreateInfo.ppEnabledExtensionNames = extensions.data();

        if (vkCreateInstance(&instanceCreateInfo, nullptr, &instance) != VK_SUCCESS){
            throw new std::runtime_error("Failed create instance");
        }

        debug::setupDebugMessenger(&instance);

        uint32_t physicalDeviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, nullptr);
        std::vector<VkPhysicalDevice> physicalDevices(physicalDeviceCount);
        vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, physicalDevices.data());

        for (auto physicalDevice : physicalDevices) {
            device = new VulkanDevice(physicalDevice);
        }

        std::vector<const char*> extensionsSwapchain = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
        device->createLogicalDevice(nullptr, &extensionsSwapchain, nullptr);
        device->createCommandPool();

        VkSurfaceKHR surface;

        if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS){
            throw new std::runtime_error("Failed create surface");
        }

        swapChain = new VulkanSwapChain(surface, device);
        VkSurfaceFormatKHR surfaceFormatKHR = {VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
        VkPresentModeKHR presentModeKHR = VK_PRESENT_MODE_FIFO_KHR;
        VkExtent2D extent2D = { WIDTH, HEIGHT };
        swapChain->createSwapChain(
                3,
                surfaceFormatKHR,
                presentModeKHR,
                extent2D
                );

        /*
         *  Creating render pass
         */

        VkAttachmentDescription colorAttachment = {};
        colorAttachment.format = swapChain->enableFormat.format;
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentReference colorAttachmentRef = {};
        colorAttachmentRef.attachment = 0;
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass = {};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachmentRef;

        VkSubpassDependency dependency = {};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.srcAccessMask = 0;
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        VkRenderPassCreateInfo renderPassInfo = {};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = 1;
        renderPassInfo.pAttachments = &colorAttachment;
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;
        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies = &dependency;

        if (vkCreateRenderPass(device->logicalDevice, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
            throw std::runtime_error("failed to create render pass!");
        }


        /*
         * Creating graphics pipeline
         */

        auto vertCode = shaders::readShaderFile(VERT_FILE_PATH);
        auto fragCode = shaders::readShaderFile(FRAG_FILE_PATH);

        VkShaderModule vertShaderModule = shaders::createShaderModule(device->logicalDevice, vertCode);
        VkShaderModule fragShaderModule = shaders::createShaderModule(device->logicalDevice, fragCode);

        VkPipelineShaderStageCreateInfo vertShaderStageInfo = {};
        vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vertShaderStageInfo.module = vertShaderModule;
        vertShaderStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo fragShaderStageInfo = {};
        fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragShaderStageInfo.module = fragShaderModule;
        fragShaderStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

        VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        auto bindingDescription = Vertex::getBindingDescription();
        auto attributeDescriptions = Vertex::getAttributeDescriptions();

        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();


        VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        VkViewport viewport = {};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = (float) swapChain->currentExtent.width;
        viewport.height = (float) swapChain->currentExtent.height;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;

        VkRect2D scissor = {};
        scissor.offset = {0, 0};
        scissor.extent = swapChain->currentExtent;

        VkPipelineViewportStateCreateInfo viewportState = {};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.pViewports = &viewport;
        viewportState.scissorCount = 1;
        viewportState.pScissors = &scissor;

        VkPipelineRasterizationStateCreateInfo rasterizer = {};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterizer.depthBiasEnable = VK_FALSE;

        VkPipelineMultisampleStateCreateInfo multisampling = {};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_FALSE;

        VkPipelineColorBlendStateCreateInfo colorBlending = {};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.logicOp = VK_LOGIC_OP_COPY;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;
        colorBlending.blendConstants[0] = 0.0f;
        colorBlending.blendConstants[1] = 0.0f;
        colorBlending.blendConstants[2] = 0.0f;
        colorBlending.blendConstants[3] = 0.0f;


        // -------------------- CREATE LAYOUT DESCRIPTORS --------------------//
        VkDescriptorSetLayoutBinding setLayoutBinding = {};
        setLayoutBinding.binding = 0;
        setLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        setLayoutBinding.descriptorCount = 1;

        VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
        descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        descriptorSetLayoutCreateInfo.bindingCount = 1;
        descriptorSetLayoutCreateInfo.pBindings = &setLayoutBinding;

        VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device->logicalDevice, &descriptorSetLayoutCreateInfo, nullptr, &setLayout), "failed to create setLayeout");


        VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &setLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 0;

        if (vkCreatePipelineLayout(device->logicalDevice, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create pipeline layout!");
        }

        // --------------------- END OF CREATING LAYOUT DESCRIPTORS------------------------//

        VkGraphicsPipelineCreateInfo pipelineInfo = {};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = shaderStages;
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.layout = pipelineLayout;
        pipelineInfo.renderPass = renderPass;
        pipelineInfo.subpass = 0;
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

        if (vkCreateGraphicsPipelines(device->logicalDevice, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
            throw std::runtime_error("failed to create graphics pipeline!");
        }

        vkDestroyShaderModule(device->logicalDevice, fragShaderModule, nullptr);
        vkDestroyShaderModule(device->logicalDevice, vertShaderModule, nullptr);

        for (auto imageView : swapChain->imageViews){
            VkImageView attachments[] = {
                    imageView
            };


            VkFramebufferCreateInfo framebufferInfo = {};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = 1;
            framebufferInfo.pAttachments = attachments;
            framebufferInfo.width = swapChain->currentExtent.width;
            framebufferInfo.height = swapChain->currentExtent.height;
            framebufferInfo.layers = 1;

            VkFramebuffer framebuffer = VK_NULL_HANDLE;
            if (vkCreateFramebuffer(device->logicalDevice, &framebufferInfo, nullptr, &framebuffer) != VK_SUCCESS) {
                throw std::runtime_error("failed to create framebuffer!");
            }

            framebuffers.push_back(framebuffer);
        }

        commandBuffers.resize(framebuffers.size());

        VkCommandBufferAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = device->commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = (uint32_t) commandBuffers.size();

        if (vkAllocateCommandBuffers(device->logicalDevice, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate command buffers!");
        }

        // ------------------------------ CREATE VERTEX BUFFER ----------------------------------//
        VkBufferCreateInfo bufferCreateInfo = {};
        bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferCreateInfo.size = sizeof(vertices[0]) * vertices.size();
        bufferCreateInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
        bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VK_CHECK_RESULT(vkCreateBuffer(device->logicalDevice, &bufferCreateInfo, nullptr, &vertexBuffer), "failed to create vertex buffer");

        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device->logicalDevice, vertexBuffer, &memRequirements);

        VkMemoryAllocateInfo allocMemInfo = {};
        allocMemInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocMemInfo.allocationSize = memRequirements.size;
        allocMemInfo.memoryTypeIndex = device->findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        VK_CHECK_RESULT(vkAllocateMemory(device->logicalDevice, &allocMemInfo, nullptr, &vertexBufferMemory), "failed to allocate vertex buffer memory");

        vkBindBufferMemory(device->logicalDevice, vertexBuffer, vertexBufferMemory, 0);
        void* data;
        vkMapMemory(device->logicalDevice, vertexBufferMemory, 0, bufferCreateInfo.size, 0, &data);
        memcpy(data, vertices.data(), (size_t) bufferCreateInfo.size);
        vkUnmapMemory(device->logicalDevice, vertexBufferMemory);

        // ---------------------------- END OF CREATEING VERTEX BUFFER -----------------------//


        // -------------------------- CREATE INDEX BUFFER ------------------------------------//
        VkBufferCreateInfo indexBufferCreateInfo = {};
        indexBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        indexBufferCreateInfo.size = sizeof(uint16_t) * indeces.size();
        indexBufferCreateInfo.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
        indexBufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VK_CHECK_RESULT(vkCreateBuffer(device->logicalDevice, &indexBufferCreateInfo, nullptr, &indexBuffer), "failded to create index buffer");

        VkMemoryRequirements memoryRequirements;
        vkGetBufferMemoryRequirements(device->logicalDevice, indexBuffer, &memoryRequirements);

        VkMemoryAllocateInfo allocIndexMemInf = {};
        allocIndexMemInf.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocIndexMemInf.allocationSize = memoryRequirements.size;
        allocIndexMemInf.memoryTypeIndex = device->findMemoryType(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        VK_CHECK_RESULT(vkAllocateMemory(device->logicalDevice, &allocIndexMemInf, nullptr, &indexBufferMemory), "failed to allocate index buffer memory");

        vkBindBufferMemory(device->logicalDevice, indexBuffer, indexBufferMemory, 0);

        void* dataIndex;
        vkMapMemory(device->logicalDevice, indexBufferMemory, 0, indexBufferCreateInfo.size, 0, &dataIndex);
        memcpy(dataIndex, indeces.data(), (size_t) indexBufferCreateInfo.size);
        vkUnmapMemory(device->logicalDevice, indexBufferMemory);

        // -------------------------- END CREATE INDEX BUFFER ---------------------------------//

        // -------------------------- CREATE UNIFORM BUFFER -----------------------------------//
        uniformBuffers.resize(swapChain->images.size());
        uniformBufferMemories.resize(swapChain->images.size());

        for (int j = 0; j < uniformBuffers.size(); ++j) {
            VkBufferCreateInfo uniformBufferCreateInfo = {};
            uniformBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            uniformBufferCreateInfo.size = sizeof(MVP);
            uniformBufferCreateInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
            uniformBufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

            VK_CHECK_RESULT(vkCreateBuffer(device->logicalDevice, &uniformBufferCreateInfo, nullptr, &uniformBuffers[j]), "failed to create uniform buffer");

            VkMemoryRequirements uniformMemoryRequirements;
            vkGetBufferMemoryRequirements(device->logicalDevice, uniformBuffers[j], &uniformMemoryRequirements);

            VkMemoryAllocateInfo allocUniformInfo = {};
            allocUniformInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            allocUniformInfo.allocationSize = uniformMemoryRequirements.size;
            allocUniformInfo.memoryTypeIndex = device->findMemoryType(uniformMemoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

            VK_CHECK_RESULT(vkAllocateMemory(device->logicalDevice, &allocUniformInfo, nullptr, &uniformBufferMemories[j]), "failed to allocate uniform memory");

            vkBindBufferMemory(device->logicalDevice, uniformBuffers[j], uniformBufferMemories[j], 0);

        }
        // -------------------------- END CREATE UNIFORM BUFFER -------------------------------//

        // -------------------------- BIND UNIFORM BUFFER -------------------------------------//

        VkDescriptorPoolSize descriptorPoolSize = {};
        descriptorPoolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorPoolSize.descriptorCount = static_cast<uint32_t>(uniformBuffers.size());

        VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
        descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        descriptorPoolCreateInfo.poolSizeCount = 1;
        descriptorPoolCreateInfo.pPoolSizes = &descriptorPoolSize;
        descriptorPoolCreateInfo.maxSets = static_cast<uint32_t>(uniformBuffers.size());

        VK_CHECK_RESULT(vkCreateDescriptorPool(device->logicalDevice, &descriptorPoolCreateInfo, nullptr, &descriptorPool), "failed to create descriptor pool");

        std::vector<VkDescriptorSetLayout> layouts(uniformBuffers.size(), setLayout);

        VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = {};
        descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        descriptorSetAllocateInfo.descriptorPool = descriptorPool;
        descriptorSetAllocateInfo.descriptorSetCount = static_cast<uint32_t>(layouts.size());
        descriptorSetAllocateInfo.pSetLayouts = layouts.data();

        descriptorSets.resize(uniformBuffers.size());
        VK_CHECK_RESULT(vkAllocateDescriptorSets(device->logicalDevice, &descriptorSetAllocateInfo, descriptorSets.data()), "failed to allocate descriptor set");

        // -------------------------- END BIND UNIFORM BUFFER ---------------------------------//

        // WRITE CMD TO BUFFER
        for (size_t i = 0; i < commandBuffers.size(); i++) {
            VkCommandBufferBeginInfo beginInfo = {};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
            beginInfo.pInheritanceInfo = nullptr; // Optional

            if (vkBeginCommandBuffer(commandBuffers[i], &beginInfo) != VK_SUCCESS) {
                throw std::runtime_error("failed to begin recording command buffer!");
            }

            VkRenderPassBeginInfo renderPassInfo = {};
            renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            renderPassInfo.renderPass = renderPass;
            renderPassInfo.framebuffer = framebuffers[i];
            renderPassInfo.renderArea.offset = {0, 0};
            renderPassInfo.renderArea.extent = swapChain->currentExtent;

            VkClearValue clearColor = {0.0f, 0.0f, 0.0f, 1.0f};
            renderPassInfo.clearValueCount = 1;
            renderPassInfo.pClearValues = &clearColor;

            vkCmdBeginRenderPass(commandBuffers[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

            vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

            VkBuffer vertexBuffers[] = {vertexBuffer};
            VkDeviceSize offsets[] = {0};
            vkCmdBindVertexBuffers(commandBuffers[i], 0, 1, vertexBuffers, offsets);

            vkCmdBindIndexBuffer(commandBuffers[i], indexBuffer, 0, VK_INDEX_TYPE_UINT16);

            vkCmdDrawIndexed(commandBuffers[i], static_cast<uint32_t>(indeces.size()), 1, 0, 0, 0);

            vkCmdEndRenderPass(commandBuffers[i]);

            if (vkEndCommandBuffer(commandBuffers[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to record command buffer!");
            }
        }

        VkSemaphoreCreateInfo renderFinishedSemaphoreInfo = {};
        renderFinishedSemaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        VK_CHECK_RESULT(vkCreateSemaphore(device->logicalDevice, &renderFinishedSemaphoreInfo, nullptr, &renderFinished), "error create render semaphore");

        VkSemaphoreCreateInfo imageReadySemaphoreInfo = {};
        imageReadySemaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        VK_CHECK_RESULT(vkCreateSemaphore(device->logicalDevice, &imageReadySemaphoreInfo, nullptr, &imageReady), "error create image semaphore");

        loop();

    }
};

int main()
{
    App app;

    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}