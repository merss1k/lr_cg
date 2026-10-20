#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec3 in_color;

layout(location = 0) out vec3 out_world_pos;
layout(location = 1) out vec3 out_normal;
layout(location = 2) out vec3 out_color;

layout(std140, set = 0, binding = 0) uniform SceneUniforms {
    mat4 view_proj;
    vec4 camera_position;
    vec4 directional_light_direction;
    vec4 directional_light_color;
    vec4 ambient_light;
    uint point_light_count;
    uint spot_light_count;
} scene;

layout(std140, set = 1, binding = 0) uniform ModelUniform {
    mat4 model;
    vec4 albedo;
    vec4 specular; // rgb = specular, w = alpha (shininess)
} model_data;

void main() {
    vec4 world_pos = model_data.model * vec4(in_position, 1.0);
    out_world_pos = world_pos.xyz;

    mat3 normal_matrix = transpose(inverse(mat3(model_data.model)));
    out_normal = normalize(normal_matrix * in_normal);

    out_color = in_color;
    gl_Position = scene.view_proj * world_pos;
}