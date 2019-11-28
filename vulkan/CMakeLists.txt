cmake_minimum_required(VERSION 3.12.1)
project(Vulkan)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(external/glfw)
include_directories(
        ${CMAKE_SOURCE_DIR}/external/glfw/include/GLFW
        ${CMAKE_SOURCE_DIR}/external/glm/glm
        ${CMAKE_SOURCE_DIR}/external/stb
        ${CMAKE_SOURCE_DIR}/external/vulkan/1.1.101.0/x86_64/include/vulkan
)

link_directories(
        ${CMAKE_SOURCE_DIR}/external/vulkan/1.1.101.0/x86_64/lib
)

set(VULKAN_LIBRARIES
        glfw
        libvulkan.so
        )

message(STATUS "Vulkan libraries: ${VULKAN_LIBRARIES}")

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/shaders
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/
        )

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/textures
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/
        )

add_executable(Vulkan
        src/main.cpp
        src/base/VulkanDevice.cpp
        src/base/VulkanSwapChain.cpp
        src/base/VulkanTools.cpp
        )

add_executable(Square
        src/square.cpp
        )

target_link_libraries(Vulkan
    ${VULKAN_LIBRARIES}
)
target_link_libraries(Square
        ${VULKAN_LIBRARIES}
        )
