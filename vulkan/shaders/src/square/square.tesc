#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(vertices = 4) out;

layout(location = 0) in vec2 texCoord_TC_in[];

layout(location = 0) out vec2 texCoord_TE_in[4];

void main() {
    texCoord_TE_in[gl_InvocationID] = texCoord_TC_in[gl_InvocationID];

    if (gl_InvocationID == 0) {
        gl_TessLevelInner[0] = 64.0f;
        gl_TessLevelInner[1] = 64.0f;

        gl_TessLevelOuter[0] = 64.0f;
        gl_TessLevelOuter[1] = 64.0f;
        gl_TessLevelOuter[2] = 64.0f;
        gl_TessLevelOuter[3] = 64.0f;
    }
    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
}