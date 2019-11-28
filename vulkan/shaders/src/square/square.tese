#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(quads, fractional_odd_spacing) in;

layout(location = 0) in vec2 texCoord_TE_in[];

layout(location = 0) out vec2 fragTexCoord;
layout(location = 1) out vec3 fragViewVec;
layout(location = 2) out vec3 fragLightVec;

void main() {

}