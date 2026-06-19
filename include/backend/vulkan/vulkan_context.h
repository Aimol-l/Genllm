#pragma once


#ifdef BACKEND_VULKAN
#include <string>
#include <mutex>
#include <vector>
#include <cstddef>
#include <memory>
#include <unordered_map>
#include <initializer_list>
#include <vulkan/vulkan.hpp>

#include "core/tensor.hpp"

// ==================== Pipeline 信息 ====================
struct PipelineInfo {
    vk::Pipeline pipeline;
    vk::PipelineLayout layout;
    vk::DescriptorSetLayout ds_layout;
    uint32_t binding_count;        // 是uniform buffer 的数量
    uint32_t push_constant_size;   // push constant 字节大小
};
struct PendingReadback {
    vk::Buffer staging_buf;
    vk::DeviceMemory staging_mem;
    void* staging_ptr;
    void* dst_cpu;
    size_t size;
};
// ==================== 每设备资源 ====================
struct DeviceSlot {
    vk::Device device;
    vk::Queue compute_queue;
    vk::CommandPool command_pool;

    bool is_recording = false;
    vk::CommandBuffer recording_cmd;

    uint32_t subgroup_size = 32;
    uint32_t compute_queue_family = 0;

    vk::PhysicalDevice physical_device;
    vk::DescriptorPool descriptor_pool;

    vk::Fence last_fence = nullptr;
    vk::PipelineLayout current_layout;
    std::vector<PendingReadback> pending_readbacks;
    std::vector<vk::DescriptorSet> pending_frees; // 保存批次执行算子后需要释放的描述符集
    std::unordered_map<std::string, PipelineInfo> pipeline_infos;

    std::unique_ptr<std::mutex> record_mutex = std::make_unique<std::mutex>();
    std::unique_ptr<std::mutex> pipeline_mutex = std::make_unique<std::mutex>();
    std::unique_ptr<std::mutex> pending_frees_mutex = std::make_unique<std::mutex>();

    std::vector<std::tuple<vk::Buffer, vk::DeviceMemory, void*>> pending_staging_frees;

    ~DeviceSlot();
    DeviceSlot(DeviceSlot&& o) noexcept;
    DeviceSlot(const DeviceSlot&) = delete;
    DeviceSlot& operator=(const DeviceSlot&) = delete;
    DeviceSlot& operator=(DeviceSlot&&) = delete;
    DeviceSlot() = default;
};

class VulkanContext {
private:
    vk::Instance instance_;
    std::vector<DeviceSlot> device_slots_;
    std::unordered_map<size_t, int> buffer_device_;

    VkDebugUtilsMessengerEXT debugMessenger_;
    PFN_vkDestroyDebugUtilsMessengerEXT pfnDestroyDebugUtilsMessengerEXT = nullptr;

    const std::vector<const char*> validationLayers_ = {"VK_LAYER_KHRONOS_validation"};
    const std::vector<const char*> deviceExtensions_ = {
        "VK_KHR_shader_float16_int8",
        "VK_KHR_shader_bfloat16",
        "VK_KHR_16bit_storage",
        "VK_KHR_shader_subgroup_extended_types",
        "VK_KHR_cooperative_matrix"
    };

    VulkanContext() { init(); }
    void init();
    void createInstance();
    void setupDebugMessenger();
    void enumerateAndCreateDevices();

public:
    static VulkanContext& get();

    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;
    ~VulkanContext();

    int device_count() const;
    DeviceSlot& slot(int device_id);
    const DeviceSlot& slot(int device_id) const;

    vk::Instance instance() const;
    vk::Device device(int device_id) const;
    vk::PhysicalDevice physical_device(int device_id) const;
    uint32_t subgroup_size(int device_id) const;

    // 创建 计算管线，描述符布局，管线布局
    void registerBuffer(size_t buffer_handle, int device_id) {
        buffer_device_[buffer_handle] = device_id;
    }
    int bufferDeviceId(size_t buffer_handle) const {
        auto it = buffer_device_.find(buffer_handle);
        if (it == buffer_device_.end()) return 0;
        return it->second;
    }

    void registerOp(
        const std::string& name,
        const uint32_t* spv_data,
        size_t spv_size_words,
        uint32_t binding_count,
        uint32_t push_constant_size);
    // ==================== Descriptor Set ====================
    
    bool isRecording(int device_id) const;

    void beginRecording(int device_id);

    vk::DescriptorSet updateDescriptorSets(int device_id,const std::string& name,std::initializer_list<Tensor*> tensors);
    vk::DescriptorSet updateDescriptorSets(int device_id, const std::string& name,std::initializer_list<vk::Buffer> buffers) ;
    vk::DescriptorSet updateDescriptorSets(int device_id, const std::string& name,std::initializer_list<vk::DescriptorBufferInfo> buffer_infos) ;

    // 背后映射到 device_slots_[device_id].pipeline_infos[name].pipeline
    void bindPipeline(int device_id,const std::string& name);
    void bindDescriptorSet(int device_id, vk::DescriptorSet descriptor_set);
    void pushConstants(int device_id, const void* data, size_t size);
    void dispatch(int device_id, uint32_t x, uint32_t y, uint32_t z);

    void deferFreeDescriptorSet(int device_id, vk::DescriptorSet ds);
    void endRecordingAndWait(int device_id);
    void cleanupPendingFrees(int device_id);


    // 暂存 buffer 管理
    vk::Buffer createStagingBuffer(int device_id, size_t size,
                                vk::DeviceMemory* out_memory, void** out_mapped);
    void destroyStagingBuffer(int device_id, vk::Buffer buffer,
                            vk::DeviceMemory memory, void* mapped);
    void cleanupPendingStagingBuffers(int device_id);

    // 获取当前录制的 CommandBuffer
    vk::CommandBuffer cmdBuffer(int device_id) const;

    // 细粒度 Buffer Barrier
    void addBufferBarrier(int device_id, vk::Buffer buffer,
                        vk::AccessFlags srcAccessMask, vk::AccessFlags dstAccessMask,
                        vk::PipelineStageFlags srcStage, vk::PipelineStageFlags dstStage);
};

#endif // BACKEND_VULKAN
