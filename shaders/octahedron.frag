#version 450

layout(location = 0) in vec3 in_world_pos;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec3 in_color;

layout(location = 0) out vec4 out_frag_color;

layout(std140, set = 0, binding = 0) uniform SceneUniforms {
    mat4 view_proj;
    vec4 camera_position;
    vec4 directional_light_direction;
    vec4 directional_light_color;
    vec4 ambient_light;
    uint point_light_count;
    uint spot_light_count;
} scene;

struct PointLight {
    vec4 position; // xyz = pos, w = intensity
    vec4 color;    // rgb = color
};

struct SpotLight {
    vec4 position;  // xyz = pos, w = inner_cutoff_cos
    vec4 direction; // xyz = dir, w = outer_cutoff_cos
    vec4 color;     // rgb = color, w = intensity
};

layout(std430, set = 0, binding = 1) readonly buffer PointLights {
    PointLight point_lights[];
};

layout(std430, set = 0, binding = 2) readonly buffer SpotLights {
    SpotLight spot_lights[];
};

layout(std140, set = 1, binding = 0) uniform ModelUniform {
    mat4 model;
    vec4 albedo;
    vec4 specular; // rgb = specular, w = alpha (shininess)
} model_data;

vec3 calcBlinnPhong(vec3 light_dir, vec3 light_color, vec3 normal, vec3 view_dir, vec3 diff_color, vec3 spec_color, float shininess) {
    float diff = max(dot(normal, light_dir), 0.0);
    vec3 diffuse = diff * diff_color;

    vec3 halfway_dir = normalize(light_dir + view_dir);
    float spec = 0.0;
    if (diff > 0.0) {
        spec = pow(max(dot(normal, halfway_dir), 0.0), shininess);
    }
    vec3 specular_part = spec * spec_color;

    return (diffuse + specular_part) * light_color;
}

void main() {
    vec3 norm = normalize(in_normal);
    vec3 view_dir = normalize(scene.camera_position.xyz - in_world_pos);

    vec3 base_albedo = model_data.albedo.rgb * in_color;
    vec3 spec_color = model_data.specular.rgb;
    float shininess = max(model_data.specular.w, 1.0);

    // 1. Ambient
    vec3 total_light = scene.ambient_light.rgb * base_albedo;

    // 2. Directional light
    vec3 dir_to_light = normalize(-scene.directional_light_direction.xyz);
    total_light += calcBlinnPhong(dir_to_light, scene.directional_light_color.rgb, norm, view_dir, base_albedo, spec_color, shininess);

    // 3. Point lights (Закон обратных квадратов)
    for (uint i = 0; i < scene.point_light_count; ++i) {
        vec3 light_vec = point_lights[i].position.xyz - in_world_pos;
        float dist = length(light_vec);
        if (dist > 0.0001) {
            vec3 light_dir = light_vec / dist;
            float attenuation = point_lights[i].position.w / (dist * dist + 0.01);
            vec3 light_intensity = point_lights[i].color.rgb * attenuation;
            total_light += calcBlinnPhong(light_dir, light_intensity, norm, view_dir, base_albedo, spec_color, shininess);
        }
    }

    // 4. Spot lights (Мягкие края + закон обратных квадратов)
    for (uint i = 0; i < scene.spot_light_count; ++i) {
        vec3 light_vec = spot_lights[i].position.xyz - in_world_pos;
        float dist = length(light_vec);
        if (dist > 0.0001) {
            vec3 light_dir = light_vec / dist;
            vec3 spot_dir = normalize(spot_lights[i].direction.xyz);

            float theta = dot(-light_dir, spot_dir);
            float inner_cos = spot_lights[i].position.w;
            float outer_cos = spot_lights[i].direction.w;

            float spot_factor = clamp((theta - outer_cos) / max(inner_cos - outer_cos, 0.0001), 0.0, 1.0);
            float attenuation = spot_lights[i].color.w / (dist * dist + 0.01);
            vec3 light_intensity = spot_lights[i].color.rgb * (attenuation * spot_factor);
            total_light += calcBlinnPhong(light_dir, light_intensity, norm, view_dir, base_albedo, spec_color, shininess);
        }
    }

    out_frag_color = vec4(total_light, 1.0);
}