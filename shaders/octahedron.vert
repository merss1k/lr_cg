#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_color;

layout(location = 0) out vec3 out_color;

layout(set = 0, binding = 0) uniform Scene {
    mat4 view;
    mat4 proj;
} scene;

layout(set = 1, binding = 0) uniform Model {
    mat4 model;
    vec3 color;
    float _padding;
} model_data;

void main() {
    vec4 world_pos = model_data.model * vec4(in_position, 1.0);
    vec4 view_pos = scene.view * world_pos;
    gl_Position = scene.proj * view_pos;
    out_color = in_color;
}