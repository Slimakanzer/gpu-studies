#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 1) uniform sampler2D texSampler;
layout(binding = 2) uniform sampler2D texNormalSampler;

layout(location = 0) in vec2 fragTexCoord;
layout(location = 1) in vec3 fragViewVec;
layout(location = 2) in vec3 fragLightVec;

layout(location = 0) out vec4 outColor;

void main() {
    vec3 N = normalize(texture(texNormalSampler, fragTexCoord).xyz);
    vec3 L = normalize(fragLightVec);
    vec3 V = normalize(fragViewVec);
    vec3 R = reflect(L, N);

    vec3 texColor = texture(texSampler, fragTexCoord).xyz;

    vec3 ambient = texColor * 0.1;
    vec3 diffuse = max(dot(N, L), 0.0) * texColor;
    vec3 specular = pow(max(dot(R, V), 0.0), 16.0) * vec3(1.35);

    outColor = vec4(ambient + diffuse + specular, 1.0);
}