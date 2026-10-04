#define GLM_FORCE_DEPTH_ZERO_TO_ONE // чтобы ортографическая проекция не отсекала Z<=0

#include "application.hpp"

#include <cstring>
#include <iostream>

#include <fstream>
#include <vector>

#include <cmath>

#include <imgui.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "graphics_internal.hpp"

namespace {

    // позиция (X, Y, Z) и цвет (R, G, B) для вершины
    struct Vertex {
        float position[3];
        float color[3];
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

        // чтение файла
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

        // 6 вершин октаэдра: по одной на каждую полуось координат
        const Vertex octahedron_vertices[] = {
            { { 1.0f,  0.0f,  0.0f },{ 1.0f, 0.5f, 0.5f } }, // +X
            { { -1.0f,  0.0f,  0.0f },{ 0.0f, 0.5f, 0.5f } }, // -X
            { { 0.0f,  1.0f,  0.0f },{ 0.5f, 1.0f, 0.5f } }, // +Y
            { { 0.0f, -1.0f,  0.0f },{ 0.5f, 0.0f, 0.5f } }, // -Y
            { { 0.0f,  0.0f,  1.0f },{ 0.5f, 0.5f, 1.0f } }, // +Z
            { { 0.0f,  0.0f, -1.0f },{ 0.5f, 0.5f, 0.0f } }, // -Z
        };

        // индексы треугольных граней октаэдра
        const uint32_t octahedron_indices[] = {
            2, 4, 0,
            2, 1, 4,
            2, 5, 1,
            2, 0, 5,
            3, 0, 4,
            3, 4, 1,
            3, 1, 5,
            3, 5, 0,
        };

        // вершинный буфер на GPU
        VkBuffer vk_vertex_buffer = VK_NULL_HANDLE;
        VmaAllocation vk_vertex_buffer_allocation = VK_NULL_HANDLE;

        bool createVertexBuffer() {
            const VkBufferCreateInfo buffer_info = {
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .size = sizeof(octahedron_vertices),
                .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            };

            const VmaAllocationCreateInfo alloc_info = {
                .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT
                       | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
                .usage = VMA_MEMORY_USAGE_AUTO,
            };

            VmaAllocationInfo allocation_info{};

            if (vmaCreateBuffer(graphics::internal::context.allocator,
                &buffer_info, &alloc_info,
                &vk_vertex_buffer, &vk_vertex_buffer_allocation,
                &allocation_info) != VK_SUCCESS) {
                std::cerr << "Failed to create vertex buffer\n";
                return false;
            }

            std::memcpy(allocation_info.pMappedData,
                octahedron_vertices,
                sizeof(octahedron_vertices));

            return true;
        }

        // то же самое для индексного буфера
        VkBuffer vk_index_buffer = VK_NULL_HANDLE;
        VmaAllocation vk_index_buffer_allocation = VK_NULL_HANDLE;

        bool createIndexBuffer() {
            const VkBufferCreateInfo buffer_info = {
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .size = sizeof(octahedron_indices),
                .usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            };

            const VmaAllocationCreateInfo alloc_info = {
                .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT
                       | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
                .usage = VMA_MEMORY_USAGE_AUTO,
            };

            VmaAllocationInfo allocation_info{};

            if (vmaCreateBuffer(graphics::internal::context.allocator,
                &buffer_info, &alloc_info,
                &vk_index_buffer, &vk_index_buffer_allocation,
                &allocation_info) != VK_SUCCESS) {
                std::cerr << "Failed to create index buffer\n";
                return false;
            }

            std::memcpy(allocation_info.pMappedData,
                octahedron_indices,
                sizeof(octahedron_indices));

            return true;
        }

        // данные всей сцены: view и proj
        struct SceneUniforms {
            glm::mat4 view;
            glm::mat4 proj;
        };

        VkBuffer vk_scene_uniform_buffer = VK_NULL_HANDLE;
        VmaAllocation vk_scene_uniform_buffer_allocation = VK_NULL_HANDLE;
        SceneUniforms* vk_scene_uniform_buffer_mapped = nullptr;

        // данные одного объекта: model-матрица и цвет
        struct ModelUniform {
            glm::mat4 model;
            glm::vec3 color;
            float _padding;
        };

        VkBuffer vk_model_uniform_buffer = VK_NULL_HANDLE;
        VmaAllocation vk_model_uniform_buffer_allocation = VK_NULL_HANDLE;
        ModelUniform* vk_model_uniform_buffer_mapped = nullptr;

        // создаёт Scene-буфер (один на сцену) и Model-буфер (для объекта)
        bool createUniformBuffers() {
            {
                const VkBufferCreateInfo buffer_info = {
                    .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                    .size = sizeof(SceneUniforms),
                    .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                    .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                };

                const VmaAllocationCreateInfo alloc_info = {
                    .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT
                           | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
                    .usage = VMA_MEMORY_USAGE_AUTO,
                };

                VmaAllocationInfo allocation_info{};

                if (vmaCreateBuffer(graphics::internal::context.allocator,
                    &buffer_info, &alloc_info,
                    &vk_scene_uniform_buffer,
                    &vk_scene_uniform_buffer_allocation,
                    &allocation_info) != VK_SUCCESS) {
                    std::cerr << "Failed to create scene uniform buffer\n";
                    return false;
                }

                vk_scene_uniform_buffer_mapped =
                    static_cast<SceneUniforms*>(allocation_info.pMappedData);

                vk_scene_uniform_buffer_mapped->view = glm::mat4(1.0f);
                vk_scene_uniform_buffer_mapped->proj = glm::mat4(1.0f);
            }

            {
                const VkBufferCreateInfo buffer_info = {
                    .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                    .size = sizeof(ModelUniform),
                    .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                    .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                };

                const VmaAllocationCreateInfo alloc_info = {
                    .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT
                           | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
                    .usage = VMA_MEMORY_USAGE_AUTO,
                };

                VmaAllocationInfo allocation_info{};

                if (vmaCreateBuffer(graphics::internal::context.allocator,
                    &buffer_info, &alloc_info,
                    &vk_model_uniform_buffer,
                    &vk_model_uniform_buffer_allocation,
                    &allocation_info) != VK_SUCCESS) {
                    std::cerr << "Failed to create model uniform buffer\n";
                    return false;
                }

                vk_model_uniform_buffer_mapped =
                    static_cast<ModelUniform*>(allocation_info.pMappedData);

                vk_model_uniform_buffer_mapped->model = glm::mat4(1.0f);
                vk_model_uniform_buffer_mapped->color = glm::vec3(1.0f);
                vk_model_uniform_buffer_mapped->_padding = 0.0f;
            }

            return true;
        }

        // два set layout: set 0 — Scene, set 1 — Model
        VkDescriptorSetLayout vk_scene_descriptor_set_layout = VK_NULL_HANDLE;
        VkDescriptorSetLayout vk_model_descriptor_set_layout = VK_NULL_HANDLE;
        // пул, из которого выделяются наборы дескрипторов
        VkDescriptorPool vk_descriptor_pool = VK_NULL_HANDLE;
        // сами наборы: один Scene, один Model
        VkDescriptorSet vk_scene_descriptor_set = VK_NULL_HANDLE;
        VkDescriptorSet vk_model_descriptor_set = VK_NULL_HANDLE;
        // pipeline layout знает, какие set layout'ы будут привязаны в пайплайне
        VkPipelineLayout vk_pipeline_layout = VK_NULL_HANDLE;
        // графический пайплайн: шейдеры + настройки рендера
        VkPipeline vk_pipeline = VK_NULL_HANDLE;

        // создаёт set layout'ы, pipeline layout, пул и наборы дескрипторов
        bool createDescriptors() {
            const VkDescriptorSetLayoutBinding scene_binding = {
                .binding = 0,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
            };

            const VkDescriptorSetLayoutCreateInfo scene_layout_info = {
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                .bindingCount = 1,
                .pBindings = &scene_binding,
            };

            if (vkCreateDescriptorSetLayout(graphics::internal::context.device,
                &scene_layout_info, nullptr,
                &vk_scene_descriptor_set_layout) != VK_SUCCESS) {
                std::cerr << "Failed to create scene descriptor set layout\n";
                return false;
            }

            const VkDescriptorSetLayoutBinding model_binding = {
                .binding = 0,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            };

            const VkDescriptorSetLayoutCreateInfo model_layout_info = {
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                .bindingCount = 1,
                .pBindings = &model_binding,
            };

            if (vkCreateDescriptorSetLayout(graphics::internal::context.device,
                &model_layout_info, nullptr,
                &vk_model_descriptor_set_layout) != VK_SUCCESS) {
                std::cerr << "Failed to create model descriptor set layout\n";
                return false;
            }

            const VkDescriptorSetLayout set_layouts[] = {
                vk_scene_descriptor_set_layout,
                vk_model_descriptor_set_layout,
            };

            // Pipeline Layout ссылается на descriptor set layout
            const VkPipelineLayoutCreateInfo pipeline_layout_info = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                .setLayoutCount = 2,
                .pSetLayouts = set_layouts,
            };

            if (vkCreatePipelineLayout(graphics::internal::context.device,
                &pipeline_layout_info, nullptr,
                &vk_pipeline_layout) != VK_SUCCESS) {
                std::cerr << "Failed to create pipeline layout\n";
                return false;
            }

            const VkDescriptorPoolSize pool_size = {
                .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .descriptorCount = 2,
            };

            const VkDescriptorPoolCreateInfo pool_info = {
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                .maxSets = 2,
                .poolSizeCount = 1,
                .pPoolSizes = &pool_size,
            };

            if (vkCreateDescriptorPool(graphics::internal::context.device,
                &pool_info, nullptr,
                &vk_descriptor_pool) != VK_SUCCESS) {
                std::cerr << "Failed to create descriptor pool\n";
                return false;
            }

            // выделение одного Scene set
            const VkDescriptorSetAllocateInfo scene_alloc_info = {
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                .descriptorPool = vk_descriptor_pool,
                .descriptorSetCount = 1,
                .pSetLayouts = &vk_scene_descriptor_set_layout,
            };

            if (vkAllocateDescriptorSets(graphics::internal::context.device,
                &scene_alloc_info,
                &vk_scene_descriptor_set) != VK_SUCCESS) {
                std::cerr << "Failed to allocate scene descriptor set\n";
                return false;
            }

            // выделение одного Model set
            const VkDescriptorSetAllocateInfo model_alloc_info = {
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                .descriptorPool = vk_descriptor_pool,
                .descriptorSetCount = 1,
                .pSetLayouts = &vk_model_descriptor_set_layout,
            };

            if (vkAllocateDescriptorSets(graphics::internal::context.device,
                &model_alloc_info,
                &vk_model_descriptor_set) != VK_SUCCESS) {
                std::cerr << "Failed to allocate model descriptor set\n";
                return false;
            }

            // привязка Scene set к Scene-буферу
            {
                const VkDescriptorBufferInfo buffer_info = {
                    .buffer = vk_scene_uniform_buffer,
                    .offset = 0,
                    .range = sizeof(SceneUniforms),
                };

                const VkWriteDescriptorSet write = {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = vk_scene_descriptor_set,
                    .dstBinding = 0,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    .pBufferInfo = &buffer_info,
                };

                vkUpdateDescriptorSets(graphics::internal::context.device,
                    1, &write, 0, nullptr);
            }

            // привязка Model set к Model-буферу
            {
                const VkDescriptorBufferInfo buffer_info = {
                    .buffer = vk_model_uniform_buffer,
                    .offset = 0,
                    .range = sizeof(ModelUniform),
                };

                const VkWriteDescriptorSet write = {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = vk_model_descriptor_set,
                    .dstBinding = 0,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    .pBufferInfo = &buffer_info,
                };

                vkUpdateDescriptorSets(graphics::internal::context.device,
                    1, &write, 0, nullptr);
            }

            return true;
        }

        // графический пайплайн
        bool createPipeline() {
            // загрузка скомпилированных шейдеров
            VkShaderModule vert_module = loadShaderModule("shaders/octahedron.vert.spv");
            VkShaderModule frag_module = loadShaderModule("shaders/octahedron.frag.spv");

            if (vert_module == VK_NULL_HANDLE || frag_module == VK_NULL_HANDLE) {
                std::cerr << "Failed to load shaders\n";
                return false;
            }

            const VkPipelineShaderStageCreateInfo stages[] = {
                {
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    .stage = VK_SHADER_STAGE_VERTEX_BIT, // вершинный
                    .module = vert_module,
                    .pName = "main",                     // точка входа
                },
                {
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    .stage = VK_SHADER_STAGE_FRAGMENT_BIT, // фрагментный
                    .module = frag_module,
                    .pName = "main",
                },
            };

            const VkVertexInputBindingDescription vertex_binding = {
                .binding = 0,
                .stride = sizeof(Vertex),
                .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
            };

            // атрибуты вершины: позиция (location=0) и цвет (location=1)
            const VkVertexInputAttributeDescription vertex_attributes[] = {
                {
                    .location = 0,
                    .binding = 0,
                    .format = VK_FORMAT_R32G32B32_SFLOAT,
                    .offset = offsetof(Vertex, position),
                },
                {
                    .location = 1,
                    .binding = 0,
                    .format = VK_FORMAT_R32G32B32_SFLOAT,
                    .offset = offsetof(Vertex, color),
                },
            };

            const VkPipelineVertexInputStateCreateInfo vertex_input = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                .vertexBindingDescriptionCount = 1,
                .pVertexBindingDescriptions = &vertex_binding,
                .vertexAttributeDescriptionCount = 2,
                .pVertexAttributeDescriptions = vertex_attributes,
            };

            // сборка треугольников
            const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                .primitiveRestartEnable = VK_FALSE,
            };

            // viewport и scissor задаются динамически в render
            const VkPipelineViewportStateCreateInfo viewport_state = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                .viewportCount = 1,
                .scissorCount = 1,
            };

            // залитые треугольники, отсечение задних граней
            const VkPipelineRasterizationStateCreateInfo rasterization = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                .depthClampEnable = VK_FALSE,
                .rasterizerDiscardEnable = VK_FALSE,
                .polygonMode = VK_POLYGON_MODE_FILL,
                .cullMode = VK_CULL_MODE_BACK_BIT,
                .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
                .depthBiasEnable = VK_FALSE,
                .lineWidth = 1.0f,
            };

            // сглаживание на краях геометрических фигур
            const VkPipelineMultisampleStateCreateInfo multisample = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
                .sampleShadingEnable = VK_FALSE,
            };

            // depth-тест: ближние фрагменты перекрывают дальние
            const VkPipelineDepthStencilStateCreateInfo depth_stencil = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
                .depthTestEnable = VK_TRUE,
                .depthWriteEnable = VK_TRUE,
                .depthCompareOp = VK_COMPARE_OP_LESS,
                .depthBoundsTestEnable = VK_FALSE,
                .stencilTestEnable = VK_FALSE,
            };

            // цвет: без смешивания, все 4 канала
            const VkPipelineColorBlendAttachmentState color_attachment = {
                .blendEnable = VK_FALSE,
                .colorWriteMask = VK_COLOR_COMPONENT_R_BIT
                                | VK_COLOR_COMPONENT_G_BIT
                                | VK_COLOR_COMPONENT_B_BIT
                                | VK_COLOR_COMPONENT_A_BIT,
            };

            const VkPipelineColorBlendStateCreateInfo color_blend = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                .logicOpEnable = VK_FALSE,
                .attachmentCount = 1,
                .pAttachments = &color_attachment,
            };

            // динамические состояния: viewport и scissor
            const VkDynamicState dynamic_states[] = {
                VK_DYNAMIC_STATE_VIEWPORT,
                VK_DYNAMIC_STATE_SCISSOR,
            };

            const VkPipelineDynamicStateCreateInfo dynamic_state = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                .dynamicStateCount = 2,
                .pDynamicStates = dynamic_states,
            };

            // сборка всего
            const VkGraphicsPipelineCreateInfo pipeline_info = {
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
                .subpass = 0,
            };

            // создание пайплайна
            if (vkCreateGraphicsPipelines(graphics::internal::context.device,
                VK_NULL_HANDLE, 1, &pipeline_info,
                nullptr, &vk_pipeline) != VK_SUCCESS) {
                std::cerr << "Failed to create graphics pipeline\n";
                vkDestroyShaderModule(graphics::internal::context.device, vert_module, nullptr);
                vkDestroyShaderModule(graphics::internal::context.device, frag_module, nullptr);
                return false;
            }

            vkDestroyShaderModule(graphics::internal::context.device, vert_module, nullptr);
            vkDestroyShaderModule(graphics::internal::context.device, frag_module, nullptr);

            return true;
        }
    } // namespace

    // создание ключевых объектов и загрузка необходимых функций
    bool initialize() {
        // пайплайн зависит от дескрипторов, дескрипторы - от uniform-буферов
        if (!createVertexBuffer()) return false;
        if (!createIndexBuffer()) return false;
        if (!createUniformBuffers()) return false;
        if (!createDescriptors()) return false;
        if (!createPipeline()) return false;
        return true;
    }

    // уничтожает все объекты в порядке, обратном созданию
    void shutdown() {
        auto& context = graphics::internal::context;
        vkQueueWaitIdle(context.graphics_queue);

        // pipeline
        if (vk_pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(context.device, vk_pipeline, nullptr);
            vk_pipeline = VK_NULL_HANDLE;
        }
        // pipeline layout
        if (vk_pipeline_layout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(context.device, vk_pipeline_layout, nullptr);
            vk_pipeline_layout = VK_NULL_HANDLE;
        }
        // пул дескрипторов
        if (vk_descriptor_pool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(context.device, vk_descriptor_pool, nullptr);
            vk_descriptor_pool = VK_NULL_HANDLE;
            vk_scene_descriptor_set = VK_NULL_HANDLE;
            vk_model_descriptor_set = VK_NULL_HANDLE;
        }
        // set layout
        if (vk_scene_descriptor_set_layout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(context.device, vk_scene_descriptor_set_layout, nullptr);
            vk_scene_descriptor_set_layout = VK_NULL_HANDLE;
        }
        if (vk_model_descriptor_set_layout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(context.device, vk_model_descriptor_set_layout, nullptr);
            vk_model_descriptor_set_layout = VK_NULL_HANDLE;
        }
        // scene uniform buffer
        if (vk_scene_uniform_buffer != VK_NULL_HANDLE) {
            vmaDestroyBuffer(context.allocator,
                vk_scene_uniform_buffer,
                vk_scene_uniform_buffer_allocation);
            vk_scene_uniform_buffer = VK_NULL_HANDLE;
            vk_scene_uniform_buffer_mapped = nullptr;
        }
        // model uniform buffer
        if (vk_model_uniform_buffer != VK_NULL_HANDLE) {
            vmaDestroyBuffer(context.allocator,
                vk_model_uniform_buffer,
                vk_model_uniform_buffer_allocation);
            vk_model_uniform_buffer = VK_NULL_HANDLE;
            vk_model_uniform_buffer_mapped = nullptr;
        }
        // вершинный и индексный буферы
        if (vk_vertex_buffer != VK_NULL_HANDLE) {
            vmaDestroyBuffer(context.allocator, vk_vertex_buffer, vk_vertex_buffer_allocation);
            vk_vertex_buffer = VK_NULL_HANDLE;
        }
        if (vk_index_buffer != VK_NULL_HANDLE) {
            vmaDestroyBuffer(context.allocator, vk_index_buffer, vk_index_buffer_allocation);
            vk_index_buffer = VK_NULL_HANDLE;
        }
    }

    // обновление данных сцены и интерфейса каждый кадр
    void update(double time) {
        // размер окна и соотношение сторон
        const auto& extent = graphics::internal::context.swapchain_extent;
        const float aspect = float(extent.width) / float(extent.height);

        static int projection_mode = 0; // 0 = перспективная, 1 = ортографическая
        static glm::vec3 manual_position = glm::vec3(0.0f); // ручное смещение объекта
        static glm::vec3 manual_rotation = glm::vec3(0.0f); // ручной поворот объекта в градусах
        static glm::vec3 scale = glm::vec3(1.0f); // масштаб объекта
        static glm::vec3 color = glm::vec3(1.0f); // цвет, выставляемый пользователем

        // состояние анимации
        static bool is_playing = false;   // идет ли анимация
        static float anim_speed = 1.0f;   // скорость анимации
        static float anim_radius = 1.5f;  // радиус траектории
        static float anim_height = 0.3f;  // высота траектории по Y
        static float anim_angle = 0.0f;   // текущий угол на траектории

        // deltaTime — сколько секунд прошло с прошлого кадра
        // нужно, чтобы скорость анимации не зависела от FPS
        static double last_time = time;
        const double delta = time - last_time;
        last_time = time;

        // обновление угла анимации
        if (is_playing) {
            // рост угла пропорционален времени и скорости
            anim_angle += float(delta) * anim_speed;
            // угол от 0 до 2pi
            if (anim_angle > 2.0f * 3.14159265f) {
                anim_angle -= 2.0f * 3.14159265f;
            }
        }

        // UI (ImGUI)
        ImGui::Begin("Octahedron controls");

        // проекция
        ImGui::Text("Projection:");
        ImGui::RadioButton("Perspective", &projection_mode, 0);
        ImGui::SameLine();
        ImGui::RadioButton("Orthographic", &projection_mode, 1);

        // трансформации
        ImGui::Separator();
        ImGui::SliderFloat3("Position", &manual_position.x, -3.0f, 3.0f); // сдвиг
        ImGui::SliderFloat3("Rotation", &manual_rotation.x, -180.0f, 180.0f); // поворот
        ImGui::SliderFloat3("Scale", &scale.x, 0.1f, 3.0f); // масштаб

        // цвет
        ImGui::Separator();
        ImGui::Text("Color:");
        ImGui::ColorEdit3("Base color", &color.x); // палитра цветов, меняет color

        // анимация
        ImGui::Separator();
        ImGui::Text("Animation:");

        ImGui::Checkbox("Animation", &is_playing); // галочка вкл/выкл анимации

        ImGui::SliderFloat("Speed", &anim_speed, -3.0f, 3.0f); // скорость анимации
        ImGui::SliderFloat("Radius", &anim_radius, 0.0f, 3.0f); // радиус траектории
        ImGui::SliderFloat("Height", &anim_height, -1.0f, 1.0f); // высота траектории
        ImGui::Text("Angle: %.2f rad", anim_angle); // текущий угол

        ImGui::End(); // закрытие окна

        // model-матрица
        glm::mat4 model = glm::mat4(1.0f);

        // ручное смещение и движение по траектории
        glm::vec3 final_position = manual_position;
        final_position.x += anim_radius * std::cos(anim_angle);
        final_position.z += anim_radius * std::sin(anim_angle * 2.0f); // фигура восьмёрка
        final_position.y += anim_height;

        // сдвиг в финальную позицию
        model = glm::translate(model, final_position);

        // поворот вокруг X
        model = glm::rotate(model, glm::radians(manual_rotation.x), glm::vec3(1, 0, 0));
        // поворот вокруг Y, фигура крутится вокруг своей оси
        model = glm::rotate(model, glm::radians(manual_rotation.y) + anim_angle * 2.0f, glm::vec3(0, 1, 0));
        // поворот вокруг Z
        model = glm::rotate(model, glm::radians(manual_rotation.z), glm::vec3(0, 0, 1));

        // масштаб по всем осям
        model = glm::scale(model, scale);

        // view-матрица, общая для всей сцены
        glm::mat4 view = glm::lookAt(
            glm::vec3(2.5f, 2.0f, 3.5f),
            glm::vec3(0.0f, 0.0f, 0.0f),
            glm::vec3(0.0f, 1.0f, 0.0f)
        );

        // proj-матрица, общая для всей сцены
        glm::mat4 proj;
        if (projection_mode == 0) {
            // перспективная
            proj = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 100.0f);
        }
        else {
            // ортографическая
            float ortho_size = 3.0f; // размер области видимости
            proj = glm::ortho(
                -ortho_size * aspect, ortho_size * aspect,
                -ortho_size, ortho_size,
                0.1f, 100.0f
            );
        }
        proj[1][1] *= -1.0f;

        // заполнение scene-буфера
        vk_scene_uniform_buffer_mapped->view = view;
        vk_scene_uniform_buffer_mapped->proj = proj;

        // заполнение model-буфера
        vk_model_uniform_buffer_mapped->model = model;
        vk_model_uniform_buffer_mapped->color = color;
    }

    // запись команд рендера в командный буфер текущего кадра
    void render(const graphics::internal::FrameData& fd) {
        // начало записи команд
        vkResetCommandBuffer(fd.command_buffer, 0); // очищение буфера от предыдущего кадра

        const VkCommandBufferBeginInfo command_buffer_begin = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, // буфер используется один раз
        };

        // начало записи, все VkCmd* пишутся в буфер
        vkBeginCommandBuffer(fd.command_buffer, &command_buffer_begin);

        // значения очистки
        const VkClearValue clear_values[] = {
            { .color = { .float32 = { 0.1f, 0.1f, 0.1f, 1.0f } } },
            { .depthStencil = { 1.0f, 0 } },
        };

        // начало render pass (определение ресурсов для рендеринга, их обработка и определение операций)
        const VkRenderPassBeginInfo render_pass_begin = {
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .renderPass = graphics::internal::context.render_pass,
            .framebuffer = fd.framebuffer, 
            .renderArea = { .extent = graphics::internal::context.swapchain_extent },
            .clearValueCount = sizeof(clear_values) / sizeof(clear_values[0]),
            .pClearValues = clear_values,
        };

        vkCmdBeginRenderPass(fd.command_buffer, &render_pass_begin, VK_SUBPASS_CONTENTS_INLINE);

        // динамические viewport и scissor
        const VkViewport viewport = {
            .x = 0.0f, .y = 0.0f,
            .width = float(graphics::internal::context.swapchain_extent.width),
            .height = float(graphics::internal::context.swapchain_extent.height),
            .minDepth = 0.0f, .maxDepth = 1.0f,
        };

        const VkRect2D scissor = {
            .extent = graphics::internal::context.swapchain_extent,
        };

        vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);
        vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);

        // установка пайплайна
        vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_pipeline);

        const VkDeviceSize vertex_buffer_offset = 0;
        // привязка вершинного буфера к binding=0
        vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &vk_vertex_buffer, &vertex_buffer_offset);
        // привязка индексного буфера
        vkCmdBindIndexBuffer(fd.command_buffer, vk_index_buffer, 0, VK_INDEX_TYPE_UINT32);

        // scene set (set=0)
        vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
            vk_pipeline_layout,
            0,
            1,
            &vk_scene_descriptor_set,
            0, nullptr);

        // model set (set=1)
        vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
            vk_pipeline_layout,
            1, 1, &vk_model_descriptor_set,
            0, nullptr);

        vkCmdDrawIndexed(fd.command_buffer,
            sizeof(octahedron_indices) / sizeof(octahedron_indices[0]), // количество индексов
            1, 0, 0, 0);

        vkCmdEndRenderPass(fd.command_buffer);
        vkEndCommandBuffer(fd.command_buffer);
    }

} // namespace application