#version 450

layout(location = 0) in vec3 in_position;

layout(std140, set = 0, binding = 0) uniform SceneUniforms {
    mat4 view_proj;
    vec4 camera_position;
    vec4 directional_light_direction;
    vec4 directional_light_color;
    vec4 ambient_light;
    uint point_light_count;
    uint spot_light_count;
} scene;

layout(push_constant) uniform PushConsts {
    mat4 model;
    vec4 color;
} push;

layout(location = 0) out vec4 out_color;

void main() {
    out_color = push.color;
    gl_Position = scene.view_proj * push.model * vec4(in_position, 1.0);
}