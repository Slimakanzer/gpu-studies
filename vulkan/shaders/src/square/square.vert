#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec3 lightPosition;
} ubo;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec2 inTexCoord;

layout(location = 0) out vec2 fragTexCoord;
layout(location = 1) out vec3 fragViewVec;
layout(location = 2) out vec3 fragLightVec;

void main() {
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 1.0);

    vec4 worldPosition = ubo.model * vec4(inPosition, 1.0);
    fragTexCoord = inTexCoord;
    fragViewVec = (ubo.view * worldPosition).xyz;
    fragLightVec = ubo.lightPosition - vec3(worldPosition);
}