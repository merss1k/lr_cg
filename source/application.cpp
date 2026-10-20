#define GLM_FORCE_DEPTH_ZERO_TO_ONE // чтобы ортографическая проекция не отсекала Z<=0

#include "application.hpp"

#include <cstring>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
#include <algorithm>

#include <imgui.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "graphics_internal.hpp"

namespace {

    // позиция, нормаль и цвет для вершины октаэдра
    struct Vertex {
        glm::vec3 position;
        glm::vec3 normal;
        glm::vec3 color;
    };

    // позиция для маркеров ламп и линий конусов
    struct SimpleVertex {
        glm::vec3 position;
    };

    // push-константы для unlit-пайплайна: model-матрица и цвет
    struct PushConstantsUnlit {
        glm::mat4 model;
        glm::vec4 color;
    };

    // чтение .spv-файла и создание VkShaderModule
    VkShaderModule loadShaderModule(const char* path) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file) {
            std::cerr << "Failed to open shader file: " << path << '\n';
            return VK_NULL_HANDLE;
        }

        const size_t size = static_cast<size_t>(file.tellg());
        std::vector<char> buffer(size); // выделение буфера под содержимое файла
        file.seekg(0);
        file.read(buffer.data(), size);
        file.close();

        const VkShaderModuleCreateInfo info = {
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = size,
            .pCode = reinterpret_cast<const uint32_t*>(buffer.data()),
        };

        VkShaderModule result = VK_NULL_HANDLE;
        if (vkCreateShaderModule(graphics::internal::context.device,
            &info, nullptr, &result) != VK_SUCCESS) {
            std::cerr << "Failed to create shader module: " << path << '\n';
            return VK_NULL_HANDLE;
        }
        return result;
    }

} // namespace

namespace application {

    namespace {

        constexpr uint32_t MAX_LIGHTS = 64; // максимум ламп каждого типа

        // данные UBO сцены (set = 0, binding = 0)
        struct SceneUniforms {
            glm::mat4 view_proj;
            glm::vec4 camera_position;
            glm::vec4 directional_light_direction;
            glm::vec4 directional_light_color;
            glm::vec4 ambient_light;
            uint32_t point_light_count;
            uint32_t spot_light_count;
            float _pad[2]; // выравнивание до 16 байт
        };

        // SSBO структуры источников
        struct GPUPointLight {
            glm::vec4 position;
            glm::vec4 color;
        };

        struct GPUSpotLight {
            glm::vec4 position;
            glm::vec4 direction;
            glm::vec4 color;
        };

        // данные UBO модели октаэдра (set = 1, binding = 0)
        struct ModelUniform {
            glm::mat4 model;
            glm::vec4 albedo; // rgb = Kd
            glm::vec4 specular; // rgb = Ks, w = shininess
        };

        // внутренние структуры для управления в UI
        struct PointLightData {
            glm::vec3 position{ 1.8f, 1.5f, 1.0f };
            glm::vec3 color{ 1.0f, 0.85f, 0.6f };
            float intensity = 5.0f;
        };

        struct SpotLightData {
            glm::vec3 position{ -1.8f, 2.2f, 1.2f };
            glm::vec3 direction{ 0.8f, -1.0f, -0.6f };
            glm::vec3 color{ 0.3f, 0.7f, 1.0f };
            float intensity = 12.0f;
            float inner_angle_deg = 15.0f;
            float outer_angle_deg = 25.0f;
        };

        // списки ламп, редактируемые в UI
        std::vector<PointLightData> point_lights;
        std::vector<SpotLightData> spot_lights;

        // буферы геометрии октаэдра
        std::vector<Vertex> octahedron_vertices;
        std::vector<uint32_t> octahedron_indices;

        // вершинный и индексный буферы октаэдра
        VkBuffer vk_vertex_buffer = VK_NULL_HANDLE;
        VmaAllocation vk_vertex_buffer_allocation = VK_NULL_HANDLE;

        VkBuffer vk_index_buffer = VK_NULL_HANDLE;
        VmaAllocation vk_index_buffer_allocation = VK_NULL_HANDLE;

        // кубик-маркер для ламп и линии-конусы для прожекторов
        VkBuffer vk_marker_cube_buffer = VK_NULL_HANDLE;
        VmaAllocation vk_marker_cube_alloc = VK_NULL_HANDLE;

        VkBuffer vk_cone_vertex_buffer = VK_NULL_HANDLE;
        VmaAllocation vk_cone_vertex_buffer_allocation = VK_NULL_HANDLE;
        uint32_t cone_vertex_count = 0;

        // UBO сцены и модели, SSBO массивов источников
        VkBuffer vk_scene_uniform_buffer = VK_NULL_HANDLE;
        VmaAllocation vk_scene_uniform_buffer_allocation = VK_NULL_HANDLE;
        SceneUniforms* vk_scene_uniform_buffer_mapped = nullptr;

        VkBuffer vk_point_ssbo = VK_NULL_HANDLE;
        VmaAllocation vk_point_ssbo_allocation = VK_NULL_HANDLE;
        GPUPointLight* vk_point_ssbo_mapped = nullptr;

        VkBuffer vk_spot_ssbo = VK_NULL_HANDLE;
        VmaAllocation vk_spot_ssbo_allocation = VK_NULL_HANDLE;
        GPUSpotLight* vk_spot_ssbo_mapped = nullptr;

        VkBuffer vk_model_uniform_buffer = VK_NULL_HANDLE;
        VmaAllocation vk_model_uniform_buffer_allocation = VK_NULL_HANDLE;
        ModelUniform* vk_model_uniform_buffer_mapped = nullptr;

        // set layout'ы, пул и наборы дескрипторов
        VkDescriptorSetLayout vk_scene_descriptor_set_layout = VK_NULL_HANDLE;
        VkDescriptorSetLayout vk_model_descriptor_set_layout = VK_NULL_HANDLE;
        VkDescriptorPool vk_descriptor_pool = VK_NULL_HANDLE;

        VkDescriptorSet vk_scene_descriptor_set = VK_NULL_HANDLE;
        VkDescriptorSet vk_model_descriptor_set = VK_NULL_HANDLE;

        // освещённый октаэдр и unlit-маркеры (fill + lines)
        VkPipelineLayout vk_pipeline_layout = VK_NULL_HANDLE;
        VkPipeline vk_pipeline = VK_NULL_HANDLE;

        VkPipelineLayout vk_unlit_pipeline_layout = VK_NULL_HANDLE;
        VkPipeline vk_unlit_pipeline = VK_NULL_HANDLE;
        VkPipeline vk_unlit_lines_pipeline = VK_NULL_HANDLE;

        // орбитальная камера вокруг начала координат
        struct Camera {
            glm::vec3 position{ 2.5f, 2.0f, 4.0f };
            float yaw = -120.0f;
            float pitch = -20.0f;
            float fov = 50.0f;
        } camera;

        // параметры освещения сцены
        glm::vec3 dir_light_direction = glm::normalize(glm::vec3(-0.5f, -1.0f, -0.5f));
        glm::vec3 dir_light_color = glm::vec3(1.0f, 0.95f, 0.9f);
        float dir_light_intensity = 0.8f;
        glm::vec3 ambient_light_color = glm::vec3(0.08f, 0.08f, 0.10f);
        bool show_light_visualizers = true;

        // процедурная генерация граней октаэдра с нормалями и цветами
        void buildOctahedron() {
            // 6 вершин октаэдра: по одной на каждую полуось
            const glm::vec3 pos[6] = {
                { 1.0f,  0.0f,  0.0f },
                { -1.0f,  0.0f,  0.0f },
                { 0.0f,  1.0f,  0.0f },
                { 0.0f, -1.0f,  0.0f },
                { 0.0f,  0.0f,  1.0f },
                { 0.0f,  0.0f, -1.0f } 
            };

            // индексы 8 треугольных граней
            const uint32_t faces[8][3] = {
                { 2, 4, 0 },{ 2, 1, 4 },{ 2, 5, 1 },{ 2, 0, 5 },
                { 3, 0, 4 },{ 3, 4, 1 },{ 3, 1, 5 },{ 3, 5, 0 }
            };

            // уникальный цвет для каждой грани
            const glm::vec3 face_colors[8] = {
                { 1.0f, 0.3f, 0.3f },{ 0.3f, 1.0f, 0.3f },
                { 0.3f, 0.5f, 1.0f },{ 1.0f, 1.0f, 0.3f },
                { 1.0f, 0.3f, 1.0f },{ 0.2f, 0.9f, 0.9f },
                { 1.0f, 0.6f, 0.2f },{ 0.8f, 0.8f, 0.8f }
            };

            octahedron_vertices.clear();
            octahedron_indices.clear();

            // для каждой грани три вершины с одной нормалью 
            for (int f = 0; f < 8; ++f) {
                glm::vec3 p0 = pos[faces[f][0]];
                glm::vec3 p1 = pos[faces[f][1]];
                glm::vec3 p2 = pos[faces[f][2]];

                // векторное произведение даёт нормаль грани
                glm::vec3 normal = glm::normalize(glm::cross(p1 - p0, p2 - p0));

                uint32_t base = static_cast<uint32_t>(octahedron_vertices.size());
                octahedron_vertices.push_back({ p0, normal, face_colors[f] });
                octahedron_vertices.push_back({ p1, normal, face_colors[f] });
                octahedron_vertices.push_back({ p2, normal, face_colors[f] });

                octahedron_indices.push_back(base + 0);
                octahedron_indices.push_back(base + 1);
                octahedron_indices.push_back(base + 2);
            }
        }

        // создание геометрии, UBO и SSBO
        bool createBuffers() {
            auto& ctx = graphics::internal::context;
            buildOctahedron();

            const VmaAllocationCreateInfo alloc_info = {
                .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
                .usage = VMA_MEMORY_USAGE_AUTO,
            };

            // вершинный буфер октаэдра
            {
                VkBufferCreateInfo buf_info{
                    .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                    .size = octahedron_vertices.size() * sizeof(Vertex),
                    .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                    .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                };
                VmaAllocationInfo ai{};
                if (vmaCreateBuffer(ctx.allocator, &buf_info, &alloc_info, &vk_vertex_buffer, &vk_vertex_buffer_allocation, &ai) != VK_SUCCESS) return false;
                std::memcpy(ai.pMappedData, octahedron_vertices.data(), buf_info.size);
            }

            // индексный буфер октаэдра
            {
                VkBufferCreateInfo buf_info{
                    .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                    .size = octahedron_indices.size() * sizeof(uint32_t),
                    .usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                    .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                };
                VmaAllocationInfo ai{};
                if (vmaCreateBuffer(ctx.allocator, &buf_info, &alloc_info, &vk_index_buffer, &vk_index_buffer_allocation, &ai) != VK_SUCCESS) return false;
                std::memcpy(ai.pMappedData, octahedron_indices.data(), buf_info.size);
            }

            // кубик-маркер для ламп (12 треугольников и 36 вершин)
            {
                const float s = 0.08f;
                const SimpleVertex cube_v[] = {
                    { { -s,-s, s } },{ { s,-s, s } },{ { s, s, s } },{ { -s,-s, s } },{ { s, s, s } },{ { -s, s, s } },
                    { { -s,-s,-s } },{ { -s, s,-s } },{ { s, s,-s } },{ { -s,-s,-s } },{ { s, s,-s } },{ { s,-s,-s } },
                    { { -s,-s,-s } },{ { -s,-s, s } },{ { -s, s, s } },{ { -s,-s,-s } },{ { -s, s, s } },{ { -s, s,-s } },
                    { { s,-s,-s } },{ { s, s,-s } },{ { s, s, s } },{ { s,-s,-s } },{ { s, s, s } },{ { s,-s, s } },
                    { { -s, s,-s } },{ { -s, s, s } },{ { s, s, s } },{ { -s, s,-s } },{ { s, s, s } },{ { s, s,-s } },
                    { { -s,-s,-s } },{ { s,-s,-s } },{ { s,-s, s } },{ { -s,-s,-s } },{ { s,-s, s } },{ { -s,-s, s } }
                };
                VkBufferCreateInfo buf_info{
                    .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                    .size = sizeof(cube_v),
                    .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                    .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                };
                VmaAllocationInfo ai{};
                if (vmaCreateBuffer(ctx.allocator, &buf_info, &alloc_info, &vk_marker_cube_buffer, &vk_marker_cube_alloc, &ai) != VK_SUCCESS) return false;
                std::memcpy(ai.pMappedData, cube_v, sizeof(cube_v));
            }

            // линии конуса прожектора, окружность основания, образующие
            {
                std::vector<SimpleVertex> cone_lines;
                constexpr int segments = 24;
                for (int i = 0; i < segments; ++i) {
                    float a0 = (float(i) / segments) * 2.0f * 3.14159265f;
                    float a1 = (float(i + 1) / segments) * 2.0f * 3.14159265f;
                    // окружность основания
                    cone_lines.push_back({ { std::cos(a0), std::sin(a0), 1.0f } });
                    cone_lines.push_back({ { std::cos(a1), std::sin(a1), 1.0f } });
                    // образующие от вершины конуса к основанию
                    if (i % 6 == 0) {
                        cone_lines.push_back({ { 0.0f, 0.0f, 0.0f } });
                        cone_lines.push_back({ { std::cos(a0), std::sin(a0), 1.0f } });
                    }
                }
                cone_vertex_count = static_cast<uint32_t>(cone_lines.size());
                VkBufferCreateInfo buf_info{
                    .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                    .size = cone_lines.size() * sizeof(SimpleVertex),
                    .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                    .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                };
                VmaAllocationInfo ai{};
                if (vmaCreateBuffer(ctx.allocator, &buf_info, &alloc_info, &vk_cone_vertex_buffer, &vk_cone_vertex_buffer_allocation, &ai) != VK_SUCCESS) return false;
                std::memcpy(ai.pMappedData, cone_lines.data(), buf_info.size);
            }

            // UBO сцены
            {
                VkBufferCreateInfo buf_info{
                    .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                    .size = sizeof(SceneUniforms),
                    .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                    .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                };
                VmaAllocationInfo ai{};
                if (vmaCreateBuffer(ctx.allocator, &buf_info, &alloc_info, &vk_scene_uniform_buffer, &vk_scene_uniform_buffer_allocation, &ai) != VK_SUCCESS) return false;
                vk_scene_uniform_buffer_mapped = static_cast<SceneUniforms*>(ai.pMappedData);
            }

            // SSBO точечных источников (фиксированный массив MAX_LIGHTS)
            {
                VkBufferCreateInfo buf_info{
                    .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                    .size = sizeof(GPUPointLight) * MAX_LIGHTS,
                    .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                };
                VmaAllocationInfo ai{};
                if (vmaCreateBuffer(ctx.allocator, &buf_info, &alloc_info, &vk_point_ssbo, &vk_point_ssbo_allocation, &ai) != VK_SUCCESS) return false;
                vk_point_ssbo_mapped = static_cast<GPUPointLight*>(ai.pMappedData);
            }

            // SSBO прожекторных источников
            {
                VkBufferCreateInfo buf_info{
                    .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                    .size = sizeof(GPUSpotLight) * MAX_LIGHTS,
                    .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                };
                VmaAllocationInfo ai{};
                if (vmaCreateBuffer(ctx.allocator, &buf_info, &alloc_info, &vk_spot_ssbo, &vk_spot_ssbo_allocation, &ai) != VK_SUCCESS) return false;
                vk_spot_ssbo_mapped = static_cast<GPUSpotLight*>(ai.pMappedData);
            }

            // UBO модели
            {
                VkBufferCreateInfo buf_info{
                    .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                    .size = sizeof(ModelUniform),
                    .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                    .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                };
                VmaAllocationInfo ai{};
                if (vmaCreateBuffer(ctx.allocator, &buf_info, &alloc_info, &vk_model_uniform_buffer, &vk_model_uniform_buffer_allocation, &ai) != VK_SUCCESS) return false;
                vk_model_uniform_buffer_mapped = static_cast<ModelUniform*>(ai.pMappedData);
            }

            return true;
        }

        bool createDescriptors() {
            auto& device = graphics::internal::context.device;

            // scene UBO + point SSBO + spot SSBO
            const VkDescriptorSetLayoutBinding scene_bindings[] = {
                { .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT },
                { .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT },
                { .binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT },
            };
            const VkDescriptorSetLayoutCreateInfo scene_layout_info{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                .bindingCount = 3,
                .pBindings = scene_bindings,
            };
            if (vkCreateDescriptorSetLayout(device, &scene_layout_info, nullptr, &vk_scene_descriptor_set_layout) != VK_SUCCESS) return false;

            // model UBO
            const VkDescriptorSetLayoutBinding model_binding{
                .binding = 0,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            };
            const VkDescriptorSetLayoutCreateInfo model_layout_info{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                .bindingCount = 1,
                .pBindings = &model_binding,
            };
            if (vkCreateDescriptorSetLayout(device, &model_layout_info, nullptr, &vk_model_descriptor_set_layout) != VK_SUCCESS) return false;

            // pipeline layout для модели
            const VkDescriptorSetLayout set_layouts[] = { vk_scene_descriptor_set_layout, vk_model_descriptor_set_layout };
            const VkPipelineLayoutCreateInfo pipeline_layout_info{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                .setLayoutCount = 2,
                .pSetLayouts = set_layouts,
            };
            if (vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &vk_pipeline_layout) != VK_SUCCESS) return false;

            // pipeline layout для unlit маркеров
            const VkPushConstantRange push_range{
                .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
                .offset = 0,
                .size = sizeof(PushConstantsUnlit),
            };
            const VkPipelineLayoutCreateInfo unlit_layout_info{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                .setLayoutCount = 1,
                .pSetLayouts = &vk_scene_descriptor_set_layout,
                .pushConstantRangeCount = 1,
                .pPushConstantRanges = &push_range,
            };
            if (vkCreatePipelineLayout(device, &unlit_layout_info, nullptr, &vk_unlit_pipeline_layout) != VK_SUCCESS) return false;

            // пул дескрипторов: 2 UBO и 2 SSBO
            const VkDescriptorPoolSize pool_sizes[] = {
                { .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 2 },
                { .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 2 },
            };
            const VkDescriptorPoolCreateInfo pool_info{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                .maxSets = 2,
                .poolSizeCount = 2,
                .pPoolSizes = pool_sizes,
            };
            if (vkCreateDescriptorPool(device, &pool_info, nullptr, &vk_descriptor_pool) != VK_SUCCESS) return false;

            // выделение scene set
            const VkDescriptorSetAllocateInfo scene_alloc{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                .descriptorPool = vk_descriptor_pool,
                .descriptorSetCount = 1,
                .pSetLayouts = &vk_scene_descriptor_set_layout,
            };
            if (vkAllocateDescriptorSets(device, &scene_alloc, &vk_scene_descriptor_set) != VK_SUCCESS) return false;

            // выделение model set
            const VkDescriptorSetAllocateInfo model_alloc{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                .descriptorPool = vk_descriptor_pool,
                .descriptorSetCount = 1,
                .pSetLayouts = &vk_model_descriptor_set_layout,
            };
            if (vkAllocateDescriptorSets(device, &model_alloc, &vk_model_descriptor_set) != VK_SUCCESS) return false;

            // привязка scene set к UBO сцены и SSBO источников
            VkDescriptorBufferInfo scene_ubo_info{ .buffer = vk_scene_uniform_buffer, .offset = 0, .range = sizeof(SceneUniforms) };
            VkDescriptorBufferInfo point_ssbo_info{ .buffer = vk_point_ssbo, .offset = 0, .range = VK_WHOLE_SIZE };
            VkDescriptorBufferInfo spot_ssbo_info{ .buffer = vk_spot_ssbo, .offset = 0, .range = VK_WHOLE_SIZE };

            const VkWriteDescriptorSet scene_writes[] = {
                { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = vk_scene_descriptor_set, .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .pBufferInfo = &scene_ubo_info },
                { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = vk_scene_descriptor_set, .dstBinding = 1, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &point_ssbo_info },
                { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = vk_scene_descriptor_set, .dstBinding = 2, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &spot_ssbo_info },
            };
            vkUpdateDescriptorSets(device, 3, scene_writes, 0, nullptr);

            // привязка model set к UBO модели
            VkDescriptorBufferInfo model_ubo_info{ .buffer = vk_model_uniform_buffer, .offset = 0, .range = sizeof(ModelUniform) };
            const VkWriteDescriptorSet model_write{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = vk_model_descriptor_set,
                .dstBinding = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .pBufferInfo = &model_ubo_info,
            };
            vkUpdateDescriptorSets(device, 1, &model_write, 0, nullptr);

            return true;
        }

        // создаёт три pipeline
        bool createPipelines() {
            auto& device = graphics::internal::context.device;

            VkShaderModule vert_m = loadShaderModule("shaders/octahedron.vert.spv");
            VkShaderModule frag_m = loadShaderModule("shaders/octahedron.frag.spv");
            if (!vert_m || !frag_m) return false;

            const VkPipelineShaderStageCreateInfo stages[] = {
                { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vert_m, .pName = "main" },
                { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = frag_m, .pName = "main" },
            };

            const VkVertexInputBindingDescription vertex_binding{
                .binding = 0,
                .stride = sizeof(Vertex),
                .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
            };

            const VkVertexInputAttributeDescription vertex_attributes[] = {
                { .location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, position) },
                { .location = 1, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, normal) },
                { .location = 2, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, color) },
            };

            const VkPipelineVertexInputStateCreateInfo vertex_input{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                .vertexBindingDescriptionCount = 1,
                .pVertexBindingDescriptions = &vertex_binding,
                .vertexAttributeDescriptionCount = 3,
                .pVertexAttributeDescriptions = vertex_attributes,
            };

            const VkPipelineInputAssemblyStateCreateInfo input_assembly{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            };

            // viewport и scissor задаются динамически
            const VkPipelineViewportStateCreateInfo viewport_state{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                .viewportCount = 1, .scissorCount = 1,
            };

            const VkPipelineRasterizationStateCreateInfo rasterization{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                .polygonMode = VK_POLYGON_MODE_FILL,
                .cullMode = VK_CULL_MODE_NONE,
                .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
                .lineWidth = 1.0f,
            };

            const VkPipelineMultisampleStateCreateInfo multisample{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
            };

            // depth-тест
            const VkPipelineDepthStencilStateCreateInfo depth_stencil{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
                .depthTestEnable = VK_TRUE,
                .depthWriteEnable = VK_TRUE,
                .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
            };

            // без смешивания, все 4 канала
            const VkPipelineColorBlendAttachmentState color_attachment{
                .blendEnable = VK_FALSE,
                .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
            };

            const VkPipelineColorBlendStateCreateInfo color_blend{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                .attachmentCount = 1,
                .pAttachments = &color_attachment,
            };

            const VkDynamicState dynamic_states[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
            const VkPipelineDynamicStateCreateInfo dynamic_state{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                .dynamicStateCount = 2,
                .pDynamicStates = dynamic_states,
            };

            // сборка основного пайплайна
            const VkGraphicsPipelineCreateInfo pipeline_info{
                .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                .stageCount = 2,
                .pStages = stages,
                .pVertexInputState = &vertex_input,
                .pInputAssemblyState = &input_assembly,
                .pViewportState = &viewport_state,
                .pRasterizationState = &rasterization,
                .pMultisampleState = &multisample,
                .pDepthStencilState = &depth_stencil,
                .pColorBlendState = &color_blend,
                .pDynamicState = &dynamic_state,
                .layout = vk_pipeline_layout,
                .renderPass = graphics::internal::context.render_pass,
            };

            if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &vk_pipeline) != VK_SUCCESS) return false;

            vkDestroyShaderModule(device, vert_m, nullptr);
            vkDestroyShaderModule(device, frag_m, nullptr);

            // unlit-пайплайны для маркеров и линий конусов
            VkShaderModule unlit_vert_m = loadShaderModule("shaders/unlit.vert.spv");
            VkShaderModule unlit_frag_m = loadShaderModule("shaders/unlit.frag.spv");
            if (!unlit_vert_m || !unlit_frag_m) return false;

            const VkPipelineShaderStageCreateInfo unlit_stages[] = {
                { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = unlit_vert_m, .pName = "main" },
                { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = unlit_frag_m, .pName = "main" },
            };

            const VkVertexInputBindingDescription simple_binding{
                .binding = 0,
                .stride = sizeof(SimpleVertex),
                .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
            };
            const VkVertexInputAttributeDescription simple_attr{
                .location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = 0,
            };
            const VkPipelineVertexInputStateCreateInfo simple_input{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                .vertexBindingDescriptionCount = 1,
                .pVertexBindingDescriptions = &simple_binding,
                .vertexAttributeDescriptionCount = 1,
                .pVertexAttributeDescriptions = &simple_attr,
            };

            // копия основного пайплайна с unlit-шейдерами и своим layout
            VkGraphicsPipelineCreateInfo unlit_info = pipeline_info;
            unlit_info.pStages = unlit_stages;
            unlit_info.pVertexInputState = &simple_input;
            unlit_info.layout = vk_unlit_pipeline_layout;

            if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &unlit_info, nullptr, &vk_unlit_pipeline) != VK_SUCCESS) return false;

            VkPipelineInputAssemblyStateCreateInfo lines_assembly{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                .topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST,
            };
            unlit_info.pInputAssemblyState = &lines_assembly;

            if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &unlit_info, nullptr, &vk_unlit_lines_pipeline) != VK_SUCCESS) return false;

            vkDestroyShaderModule(device, unlit_vert_m, nullptr);
            vkDestroyShaderModule(device, unlit_frag_m, nullptr);

            return true;
        }

    } // namespace

    // создаёт начальные источники света и все объекты
    bool initialize() {
        // стартовые лампы
        point_lights.push_back({ { 1.5f, 1.2f, 1.2f },{ 1.0f, 0.85f, 0.6f }, 4.0f });
        spot_lights.push_back({ { -1.8f, 2.2f, 1.5f }, glm::normalize(glm::vec3(0.9f, -1.0f, -0.7f)),{ 0.3f, 0.7f, 1.0f }, 15.0f, 15.0f, 25.0f });

        // пайплайны зависят от дескрипторов, а дескрипторы от буферов
        if (!createBuffers()) return false;
        if (!createDescriptors()) return false;
        if (!createPipelines()) return false;
        return true;
    }

    // уничтожает все объекты в порядке, обратном созданию
    void shutdown() {
        auto& context = graphics::internal::context;
        vkDeviceWaitIdle(context.device);

        if (vk_unlit_lines_pipeline != VK_NULL_HANDLE) vkDestroyPipeline(context.device, vk_unlit_lines_pipeline, nullptr);
        if (vk_unlit_pipeline != VK_NULL_HANDLE) vkDestroyPipeline(context.device, vk_unlit_pipeline, nullptr);
        if (vk_pipeline != VK_NULL_HANDLE) vkDestroyPipeline(context.device, vk_pipeline, nullptr);

        if (vk_unlit_pipeline_layout != VK_NULL_HANDLE) vkDestroyPipelineLayout(context.device, vk_unlit_pipeline_layout, nullptr);
        if (vk_pipeline_layout != VK_NULL_HANDLE) vkDestroyPipelineLayout(context.device, vk_pipeline_layout, nullptr);

        if (vk_descriptor_pool != VK_NULL_HANDLE) vkDestroyDescriptorPool(context.device, vk_descriptor_pool, nullptr);
        if (vk_scene_descriptor_set_layout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(context.device, vk_scene_descriptor_set_layout, nullptr);
        if (vk_model_descriptor_set_layout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(context.device, vk_model_descriptor_set_layout, nullptr);

        if (vk_scene_uniform_buffer != VK_NULL_HANDLE) vmaDestroyBuffer(context.allocator, vk_scene_uniform_buffer, vk_scene_uniform_buffer_allocation);
        if (vk_point_ssbo != VK_NULL_HANDLE) vmaDestroyBuffer(context.allocator, vk_point_ssbo, vk_point_ssbo_allocation);
        if (vk_spot_ssbo != VK_NULL_HANDLE) vmaDestroyBuffer(context.allocator, vk_spot_ssbo, vk_spot_ssbo_allocation);
        if (vk_model_uniform_buffer != VK_NULL_HANDLE) vmaDestroyBuffer(context.allocator, vk_model_uniform_buffer, vk_model_uniform_buffer_allocation);

        if (vk_vertex_buffer != VK_NULL_HANDLE) vmaDestroyBuffer(context.allocator, vk_vertex_buffer, vk_vertex_buffer_allocation);
        if (vk_index_buffer != VK_NULL_HANDLE) vmaDestroyBuffer(context.allocator, vk_index_buffer, vk_index_buffer_allocation);
        if (vk_marker_cube_buffer != VK_NULL_HANDLE) vmaDestroyBuffer(context.allocator, vk_marker_cube_buffer, vk_marker_cube_alloc);
        if (vk_cone_vertex_buffer != VK_NULL_HANDLE) vmaDestroyBuffer(context.allocator, vk_cone_vertex_buffer, vk_cone_vertex_buffer_allocation);
    }

    // обновление данных сцены, камеры, ламп и интерфейса каждый кадр
    void update(double time) {
        const auto& extent = graphics::internal::context.swapchain_extent;
        const float aspect = float(extent.width) / float(extent.height);

        
        static int projection_mode = 0; // 0 = перспективная, 1 = ортографическая
        static glm::vec3 manual_position = glm::vec3(0.0f); // ручное смещение объекта
        static glm::vec3 manual_rotation = glm::vec3(0.0f); // ручной поворот объекта в градусах
        static glm::vec3 scale = glm::vec3(1.0f); // масштаб объекта

        // материал октаэдра
        static glm::vec3 albedo_color = glm::vec3(1.0f); // Kd
        static glm::vec3 specular_color = glm::vec3(1.0f); // Ks
        static float shininess = 64.0f; // альфа

        // состояние анимации
        static bool is_playing = false; // идет ли анимация
        static float anim_speed = 1.0f; // скорость анимации
        static float anim_radius = 1.5f; // радиус траектории
        static float anim_height = 0.3f; // высота траектории по Y
        static float anim_angle = 0.0f; // текущий угол на траектории

        // deltaTime — сколько секунд прошло с прошлого кадра
        // нужно, чтобы скорость анимации не зависела от FPS
        static double last_time = time;
        const double delta = time - last_time;
        last_time = time;

        // обновление угла анимации
        if (is_playing) {
            anim_angle += float(delta) * anim_speed;
            if (anim_angle > 2.0f * 3.14159265f) anim_angle -= 2.0f * 3.14159265f;
        }

        // UI (ImGUI)
        ImGui::Begin("Octahedron: Camera & Lighting");

        // камера и проекция
        if (ImGui::CollapsingHeader("Projection & Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::RadioButton("Perspective", &projection_mode, 0); ImGui::SameLine();
            ImGui::RadioButton("Orthographic", &projection_mode, 1);
            ImGui::SliderFloat3("Camera Pos", &camera.position.x, -10.0f, 10.0f);
            ImGui::SliderFloat("Yaw", &camera.yaw, -180.0f, 180.0f);
            ImGui::SliderFloat("Pitch", &camera.pitch, -89.0f, 89.0f);
            ImGui::SliderFloat("FOV", &camera.fov, 15.0f, 110.0f);
        }

        // материал октаэдра
        if (ImGui::CollapsingHeader("Octahedron Material (Blinn-Phong)", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::ColorEdit3("Albedo (Kd)", &albedo_color.x);
            ImGui::ColorEdit3("Specular (Ks)", &specular_color.x);
            ImGui::SliderFloat("Shininess (Alpha)", &shininess, 1.0f, 256.0f);
        }

        // трансформации и анимация октаэдра
        if (ImGui::CollapsingHeader("Octahedron Transform & Animation")) {
            ImGui::SliderFloat3("Position", &manual_position.x, -3.0f, 3.0f);
            ImGui::SliderFloat3("Rotation", &manual_rotation.x, -180.0f, 180.0f);
            ImGui::SliderFloat3("Scale", &scale.x, 0.1f, 3.0f);
            ImGui::Separator();
            ImGui::Checkbox("Animation", &is_playing);
            ImGui::SliderFloat("Speed", &anim_speed, -3.0f, 3.0f);
            ImGui::SliderFloat("Radius", &anim_radius, 0.0f, 3.0f);
            ImGui::SliderFloat("Height", &anim_height, -1.0f, 1.0f);
        }

        // направленный и ambient-свет
        if (ImGui::CollapsingHeader("Directional & Ambient Light", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::ColorEdit3("Ambient Light", &ambient_light_color.x);
            ImGui::SliderFloat3("Sun Direction", &dir_light_direction.x, -1.0f, 1.0f);
            if (glm::length(dir_light_direction) > 0.001f) dir_light_direction = glm::normalize(dir_light_direction);
            ImGui::ColorEdit3("Sun Color", &dir_light_color.x);
            ImGui::SliderFloat("Sun Intensity", &dir_light_intensity, 0.0f, 5.0f);
            ImGui::Checkbox("Show Light Gizmos & Cones", &show_light_visualizers);
        }

        // список точечных источников с добавлением и удалением
        if (ImGui::CollapsingHeader("Point Lights (SSBO Array)")) {
            if (ImGui::Button("+ Add Point Light") && point_lights.size() < MAX_LIGHTS) {
                point_lights.push_back({ { 0.0f, 2.0f, 0.0f },{ 1.0f, 1.0f, 1.0f }, 4.0f });
            }
            ImGui::SameLine();
            ImGui::Text("(Count: %zu)", point_lights.size());

            for (size_t i = 0; i < point_lights.size(); ++i) {
                ImGui::PushID(static_cast<int>(i));
                std::string title = "Point Light #" + std::to_string(i + 1);
                if (ImGui::TreeNode(title.c_str())) {
                    ImGui::SliderFloat3("Position", &point_lights[i].position.x, -6.0f, 6.0f);
                    ImGui::ColorEdit3("Color", &point_lights[i].color.x);
                    ImGui::SliderFloat("Intensity", &point_lights[i].intensity, 0.1f, 20.0f);
                    if (ImGui::Button("Delete")) {
                        point_lights.erase(point_lights.begin() + i);
                        ImGui::TreePop();
                        ImGui::PopID();
                        break; // прерываем цикл, т.к. вектор изменился
                    }
                    ImGui::TreePop();
                }
                ImGui::PopID();
            }
        }

        // список прожекторных источников
        if (ImGui::CollapsingHeader("Spot Lights (SSBO Array)")) {
            if (ImGui::Button("+ Add Spot Light") && spot_lights.size() < MAX_LIGHTS) {
                spot_lights.push_back({ { 0.0f, 3.0f, 1.0f },{ 0.0f, -1.0f, -0.3f },{ 1.0f, 1.0f, 1.0f }, 10.0f, 15.0f, 25.0f });
            }
            ImGui::SameLine();
            ImGui::Text("(Count: %zu)", spot_lights.size());

            for (size_t i = 0; i < spot_lights.size(); ++i) {
                ImGui::PushID(static_cast<int>(100 + i));
                std::string title = "Spot Light #" + std::to_string(i + 1);
                if (ImGui::TreeNode(title.c_str())) {
                    ImGui::SliderFloat3("Position", &spot_lights[i].position.x, -6.0f, 6.0f);
                    ImGui::SliderFloat3("Direction", &spot_lights[i].direction.x, -1.0f, 1.0f);
                    if (glm::length(spot_lights[i].direction) > 0.001f) spot_lights[i].direction = glm::normalize(spot_lights[i].direction);
                    ImGui::ColorEdit3("Color", &spot_lights[i].color.x);
                    ImGui::SliderFloat("Intensity", &spot_lights[i].intensity, 0.1f, 30.0f);
                    ImGui::SliderFloat("Inner Angle", &spot_lights[i].inner_angle_deg, 1.0f, 60.0f);
                    ImGui::SliderFloat("Outer Angle", &spot_lights[i].outer_angle_deg, spot_lights[i].inner_angle_deg + 0.1f, 80.0f);
                    if (ImGui::Button("Delete")) {
                        spot_lights.erase(spot_lights.begin() + i);
                        ImGui::TreePop();
                        ImGui::PopID();
                        break; // прерываем цикл, т.к. вектор изменился
                    }
                    ImGui::TreePop();
                }
                ImGui::PopID();
            }
        }

        ImGui::End();

        // расчёт камеры по yaw/pitch
        glm::vec3 front;
        front.x = cos(glm::radians(camera.yaw)) * cos(glm::radians(camera.pitch));
        front.y = sin(glm::radians(camera.pitch));
        front.z = sin(glm::radians(camera.yaw)) * cos(glm::radians(camera.pitch));
        front = glm::normalize(front);

        glm::mat4 view = glm::lookAt(camera.position, camera.position + front, glm::vec3(0.0f, 1.0f, 0.0f));

        // proj-матрица в зависимости от режима
        glm::mat4 proj;
        if (projection_mode == 0) {
            proj = glm::perspective(glm::radians(camera.fov), aspect, 0.1f, 100.0f);
        }
        else {
            float ortho_size = 3.5f;
            proj = glm::ortho(-ortho_size * aspect, ortho_size * aspect, -ortho_size, ortho_size, 0.1f, 100.0f);
        }
        proj[1][1] *= -1.0f;

        // сдвиг, поворот, масштаб, траектория (model-матрица)
        glm::mat4 model = glm::mat4(1.0f);
        glm::vec3 final_position = manual_position;
        final_position.x += anim_radius * std::cos(anim_angle);
        final_position.z += anim_radius * std::sin(anim_angle * 2.0f);
        final_position.y += anim_height;

        model = glm::translate(model, final_position);
        model = glm::rotate(model, glm::radians(manual_rotation.x), glm::vec3(1, 0, 0));
        model = glm::rotate(model, glm::radians(manual_rotation.y) + anim_angle * 2.0f, glm::vec3(0, 1, 0));
        model = glm::rotate(model, glm::radians(manual_rotation.z), glm::vec3(0, 0, 1));
        model = glm::scale(model, scale);

        // камера, направленный свет, счётчики (scene UBO)
        vk_scene_uniform_buffer_mapped->view_proj = proj * view;
        vk_scene_uniform_buffer_mapped->camera_position = glm::vec4(camera.position, 1.0f);
        vk_scene_uniform_buffer_mapped->directional_light_direction = glm::vec4(dir_light_direction, 0.0f);
        vk_scene_uniform_buffer_mapped->directional_light_color = glm::vec4(dir_light_color * dir_light_intensity, 1.0f);
        vk_scene_uniform_buffer_mapped->ambient_light = glm::vec4(ambient_light_color, 1.0f);
        vk_scene_uniform_buffer_mapped->point_light_count = static_cast<uint32_t>(point_lights.size());
        vk_scene_uniform_buffer_mapped->spot_light_count = static_cast<uint32_t>(spot_lights.size());

        // модель и материал (model UBO)
        vk_model_uniform_buffer_mapped->model = model;
        vk_model_uniform_buffer_mapped->albedo = glm::vec4(albedo_color, 1.0f);
        vk_model_uniform_buffer_mapped->specular = glm::vec4(specular_color, shininess);

        // point SSBO (только активные лампы)
        for (size_t i = 0; i < point_lights.size(); ++i) {
            vk_point_ssbo_mapped[i].position = glm::vec4(point_lights[i].position, point_lights[i].intensity);
            vk_point_ssbo_mapped[i].color = glm::vec4(point_lights[i].color, 1.0f);
        }

        // углы конвертируются в косинусы
        for (size_t i = 0; i < spot_lights.size(); ++i) {
            float inner_cos = std::cos(glm::radians(spot_lights[i].inner_angle_deg));
            float outer_cos = std::cos(glm::radians(spot_lights[i].outer_angle_deg));
            vk_spot_ssbo_mapped[i].position = glm::vec4(spot_lights[i].position, inner_cos);
            vk_spot_ssbo_mapped[i].direction = glm::vec4(spot_lights[i].direction, outer_cos);
            vk_spot_ssbo_mapped[i].color = glm::vec4(spot_lights[i].color, spot_lights[i].intensity);
        }
    }

    // запись команд рендера в командный буфер текущего кадра
    void render(const graphics::internal::FrameData& fd) {
        // начало записи команд
        vkResetCommandBuffer(fd.command_buffer, 0); // очищение буфера от предыдущего кадра

        const VkCommandBufferBeginInfo begin_info{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, // буфер используется один раз
        };
        vkBeginCommandBuffer(fd.command_buffer, &begin_info);

        // значения очистки
        const VkClearValue clear_values[] = {
            { .color = { .float32 = { 0.05f, 0.05f, 0.08f, 1.0f } } },
            { .depthStencil = { 1.0f, 0 } },
        };

        // начало render pass
        const VkRenderPassBeginInfo render_pass_begin{
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .renderPass = graphics::internal::context.render_pass,
            .framebuffer = fd.framebuffer,
            .renderArea = { .extent = graphics::internal::context.swapchain_extent },
            .clearValueCount = 2,
            .pClearValues = clear_values,
        };

        vkCmdBeginRenderPass(fd.command_buffer, &render_pass_begin, VK_SUBPASS_CONTENTS_INLINE);

        // динамические viewport и scissor
        const VkViewport viewport{
            .x = 0.0f, .y = 0.0f,
            .width = float(graphics::internal::context.swapchain_extent.width),
            .height = float(graphics::internal::context.swapchain_extent.height),
            .minDepth = 0.0f, .maxDepth = 1.0f,
        };
        const VkRect2D scissor{ .extent = graphics::internal::context.swapchain_extent };
        vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);
        vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);

        // октаэдр с освещением (Blinn-Phong)
        vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_pipeline);

        const VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &vk_vertex_buffer, &offset);
        vkCmdBindIndexBuffer(fd.command_buffer, vk_index_buffer, 0, VK_INDEX_TYPE_UINT32);

        // scene set (set=0)
        vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
            vk_pipeline_layout, 0, 1, &vk_scene_descriptor_set, 0, nullptr);
        // model set (set=1)
        vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
            vk_pipeline_layout, 1, 1, &vk_model_descriptor_set, 0, nullptr);

        vkCmdDrawIndexed(fd.command_buffer, static_cast<uint32_t>(octahedron_indices.size()), 1, 0, 0, 0);

        // визуализация источников
        if (show_light_visualizers) {
            // сплошные кубики маркеров цвета ламп
            vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_unlit_pipeline);
            vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                vk_unlit_pipeline_layout, 0, 1, &vk_scene_descriptor_set, 0, nullptr);
            vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &vk_marker_cube_buffer, &offset);

            // точечные источники
            for (const auto& pl : point_lights) {
                glm::mat4 m = glm::translate(glm::mat4(1.0f), pl.position);
                PushConstantsUnlit push{ .model = m, .color = glm::vec4(pl.color, 1.0f) };
                vkCmdPushConstants(fd.command_buffer, vk_unlit_pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(push), &push);
                vkCmdDraw(fd.command_buffer, 36, 1, 0, 0);
            }

            // прожекторные источники
            for (const auto& sl : spot_lights) {
                glm::mat4 m = glm::translate(glm::mat4(1.0f), sl.position);
                PushConstantsUnlit push{ .model = m, .color = glm::vec4(sl.color, 1.0f) };
                vkCmdPushConstants(fd.command_buffer, vk_unlit_pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(push), &push);
                vkCmdDraw(fd.command_buffer, 36, 1, 0, 0);
            }

            // линии конусов прожекторов
            vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_unlit_lines_pipeline);
            vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &vk_cone_vertex_buffer, &offset);

            for (const auto& sl : spot_lights) {
                // поворот конуса из +Z в направление прожектора
                glm::vec3 dir = glm::normalize(sl.direction);
                glm::vec3 default_dir(0.0f, 0.0f, 1.0f);
                glm::mat4 rot(1.0f);

                if (glm::length(dir - default_dir) > 0.0001f) {
                    if (glm::length(dir + default_dir) < 0.0001f) {
                        rot = glm::rotate(glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(1, 0, 0));
                    }
                    else {
                        glm::vec3 axis = glm::cross(default_dir, dir);
                        float angle = std::acos(std::clamp(glm::dot(default_dir, dir), -1.0f, 1.0f));
                        rot = glm::rotate(glm::mat4(1.0f), angle, glm::normalize(axis));
                    }
                }

                // длина конуса и радиус основания из outer-угла
                float length = 2.0f;
                float radius = length * std::tan(glm::radians(sl.outer_angle_deg));

                // translate * rotate * scale
                glm::mat4 m = glm::translate(glm::mat4(1.0f), sl.position) * rot * glm::scale(glm::mat4(1.0f), glm::vec3(radius, radius, length));
                PushConstantsUnlit push{ .model = m, .color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f) };
                vkCmdPushConstants(fd.command_buffer, vk_unlit_pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(push), &push);
                vkCmdDraw(fd.command_buffer, cone_vertex_count, 1, 0, 0);
            }
        }

        vkCmdEndRenderPass(fd.command_buffer);
        vkEndCommandBuffer(fd.command_buffer);
    }

} // namespace application