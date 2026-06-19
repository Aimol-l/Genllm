
#ifdef BACKEND_VULKAN

#include <print>
#include <cstring>
#include <format>

#include "vulkan/vulkan.hpp"
#include "backend/vulkan/vulkan_context.h"

// ==================== DeviceSlot ====================

DeviceSlot::~DeviceSlot() {
    if (!device) return;
    device.waitIdle();
    for (auto& [name, info] : pipeline_infos) {
        device.destroyPipeline(info.pipeline);
        device.destroyPipelineLayout(info.layout);
        device.destroyDescriptorSetLayout(info.ds_layout);
    }
    device.destroyDescriptorPool(descriptor_pool);
    device.destroyCommandPool(command_pool);
    device.destroy();
}

DeviceSlot::DeviceSlot(DeviceSlot&& o) noexcept
    : physical_device(o.physical_device)
    , device(std::exchange(o.device, nullptr))
    , compute_queue(o.compute_queue)
    , compute_queue_family(o.compute_queue_family)
    , command_pool(std::exchange(o.command_pool, nullptr))
    , descriptor_pool(std::exchange(o.descriptor_pool, nullptr))
    , subgroup_size(o.subgroup_size)
    , recording_cmd(std::exchange(o.recording_cmd, nullptr))
    , is_recording(std::exchange(o.is_recording, false))
    , last_fence(std::exchange(o.last_fence, nullptr))
    , pipeline_mutex(std::move(o.pipeline_mutex))
    , pipeline_infos(std::move(o.pipeline_infos))
{}

// ==================== VulkanContext ====================

VulkanContext& VulkanContext::get() {
    static VulkanContext instance;
    return instance;
}

VulkanContext::~VulkanContext() {
    device_slots_.clear();
    if (pfnDestroyDebugUtilsMessengerEXT) {
        pfnDestroyDebugUtilsMessengerEXT(static_cast<VkInstance>(instance_), debugMessenger_, nullptr);
    }
    if (instance_) instance_.destroy();
}

int VulkanContext::device_count() const { return static_cast<int>(device_slots_.size()); }
DeviceSlot& VulkanContext::slot(int device_id) { return device_slots_.at(device_id); }
const DeviceSlot& VulkanContext::slot(int device_id) const { return device_slots_.at(device_id); }

vk::Instance VulkanContext::instance() const { return instance_; }
vk::Device VulkanContext::device(int device_id) const { return slot(device_id).device; }
vk::PhysicalDevice VulkanContext::physical_device(int device_id) const { return slot(device_id).physical_device; }
uint32_t VulkanContext::subgroup_size(int device_id) const { return slot(device_id).subgroup_size; }

// ==================== init ====================

void VulkanContext::init() {
    this->createInstance();
    this->setupDebugMessenger();
    this->enumerateAndCreateDevices();
}

void VulkanContext::createInstance() {
    vk::ApplicationInfo app_info;
    app_info.setPApplicationName("Genllm")
            .setApplicationVersion(VK_MAKE_VERSION(1, 0, 0))
            .setPEngineName("GenllmEngine")
            .setEngineVersion(VK_MAKE_VERSION(1, 0, 0))
            .setApiVersion(VK_API_VERSION_1_4);

    vk::InstanceCreateInfo create_info({}, &app_info);

    auto result = vk::createInstance(&create_info, nullptr, &instance_);
    if (result != vk::Result::eSuccess) {
        throw std::runtime_error(std::format("VulkanContext: createInstance failed ({})", vk::to_string(result)));
    }
}

void VulkanContext::setupDebugMessenger() {
    static PFN_vkDebugUtilsMessengerCallbackEXT DebugUtilsMessengerCallback = [](
        VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        VkDebugUtilsMessageTypeFlagsEXT messageTypes,
        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
        void* pUserData)->VkBool32 {
            std::printf("{}", pCallbackData->pMessage);
            return VK_FALSE;
    };
    VkDebugUtilsMessengerCreateInfoEXT debugUtilsMessengerCreateInfo = {
        .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
        .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
        .pfnUserCallback = DebugUtilsMessengerCallback
    };
    pfnDestroyDebugUtilsMessengerEXT = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(static_cast<VkInstance>(instance_), "vkDestroyDebugUtilsMessengerEXT");
    PFN_vkCreateDebugUtilsMessengerEXT vkCreateDebugUtilsMessenger = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(this->instance_, "vkCreateDebugUtilsMessengerEXT"));
    if (vkCreateDebugUtilsMessenger) {
        VkResult result = vkCreateDebugUtilsMessenger(this->instance_, &debugUtilsMessengerCreateInfo, nullptr, &debugMessenger_);
        if (result)
            std::println("[VulkanBase] ERROR Failed to create a debug messenger! Error code: {}", int32_t(result));
    }
}

void VulkanContext::enumerateAndCreateDevices() {
    auto phys_devices = instance_.enumeratePhysicalDevices();

    if (phys_devices.empty()) {
        throw std::runtime_error("VulkanContext: no Vulkan-capable devices");
    }

    for (auto& phy : phys_devices) {
        auto props = phy.getProperties();
        auto families = phy.getQueueFamilyProperties();

        bool has_compute = false;
        uint32_t queue_family = 0;
        for (uint32_t i = 0; i < families.size(); ++i) {
            if (families[i].queueFlags & vk::QueueFlagBits::eCompute) {
                has_compute = true;
                queue_family = i;
                break;
            }
        }
        if (!has_compute) continue;

        vk::PhysicalDeviceSubgroupProperties subgroup_props;
        vk::PhysicalDeviceProperties2 props2({}, &subgroup_props);
        phy.getProperties2(&props2);
        uint32_t sg_size = subgroup_props.subgroupSize;

        float priority = 1.0f;
        vk::DeviceQueueCreateInfo queue_info({}, queue_family, 1, &priority);

        auto ext_props = phy.enumerateDeviceExtensionProperties();

        bool found = true;
        for (const char* extName : this->deviceExtensions_) {
            bool ext_found = false;
            for (const auto& prop : ext_props) {
                if (strcmp(extName, prop.extensionName) == 0) {
                    ext_found = true;
                    break;
                }
            }
            found = found && ext_found;
        }
        if (!found) {
            auto name = std::string(props.deviceName);
            std::println("VulkanContext: {} does not support required extensions", name);
            continue;
        }

        vk::PhysicalDevice16BitStorageFeatures storage16_feat;
        storage16_feat.storageBuffer16BitAccess = VK_TRUE;

        vk::PhysicalDeviceShaderFloat16Int8Features float16_feat;
        float16_feat.shaderFloat16 = VK_TRUE;
        float16_feat.pNext = &storage16_feat;

        vk::PhysicalDeviceShaderBfloat16FeaturesKHR bf16_feat;
        bf16_feat.shaderBFloat16Type = VK_TRUE;
        bf16_feat.pNext = &float16_feat;

        vk::PhysicalDeviceCooperativeMatrixFeaturesKHR coop_feat;
        coop_feat.cooperativeMatrix = VK_TRUE;
        coop_feat.pNext = &bf16_feat;

        vk::DeviceCreateInfo device_info{};
        device_info.setQueueCreateInfos(queue_info)
            .setEnabledExtensionCount(static_cast<uint32_t>(deviceExtensions_.size()))
            .setPpEnabledExtensionNames(deviceExtensions_.data())
            .setPNext(&coop_feat);

        vk::Device dev;
        try {
            dev = phy.createDevice(device_info);
        } catch (const vk::SystemError& e) {
            std::println("Vulkan: skipping device {} ({})", std::string_view(props.deviceName), e.what());
            continue;
        }

        auto queue = dev.getQueue(queue_family, 0);

        vk::CommandPoolCreateInfo cmd_pool_info;
        cmd_pool_info.setQueueFamilyIndex(queue_family)
                     .setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer | vk::CommandPoolCreateFlagBits::eTransient);
        vk::CommandPool cmd_pool;
        try {
           cmd_pool = dev.createCommandPool(cmd_pool_info);
        } catch (const vk::SystemError& e) {
            throw std::runtime_error(std::string("Failed to create command pool: ") + e.what());
        }

        vk::DescriptorPoolSize pool_size(vk::DescriptorType::eStorageBuffer, 4096);
        vk::DescriptorPoolCreateInfo desc_pool_info(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 1024, pool_size);
        desc_pool_info.setFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet);

        vk::DescriptorPool desc_pool;
        try {
            desc_pool = dev.createDescriptorPool(desc_pool_info);
        } catch (const vk::SystemError& e) {
            throw std::runtime_error(std::string("Failed to create descriptor pool: ") + e.what());
        }
        vk::CommandBufferAllocateInfo allocInfo(cmd_pool, vk::CommandBufferLevel::ePrimary, 1);
        auto cmdBuffers = dev.allocateCommandBuffers(allocInfo);
        DeviceSlot slot;
        slot.device = dev;
        slot.physical_device = phy;
        slot.compute_queue = queue;
        slot.command_pool = cmd_pool;
        slot.subgroup_size = sg_size;
        slot.descriptor_pool = desc_pool;
        slot.compute_queue_family = queue_family;
        slot.recording_cmd = cmdBuffers[0];


        device_slots_.push_back(std::move(slot));

        std::println("Vulkan[{}]: {}", device_slots_.size() - 1, std::string_view(props.deviceName));
    }

    if (device_slots_.empty()) {
        throw std::runtime_error("VulkanContext: no suitable GPU found");
    }
}


void VulkanContext::registerOp(
    const std::string& name,
    const uint32_t* spv_data,
    size_t spv_size_words,
    uint32_t binding_count,
    uint32_t push_constant_size)
{
    for (int d = 0; d < this->device_count(); ++d) {
        auto& s = this->slot(d);
        std::lock_guard<std::mutex> lock(*s.pipeline_mutex);
        if (s.pipeline_infos.find(name) != s.pipeline_infos.end()) continue;

        auto dev = s.device;

        std::vector<vk::DescriptorSetLayoutBinding> bindings(binding_count);
        for (uint32_t i = 0; i < binding_count; ++i) {
            bindings[i] = vk::DescriptorSetLayoutBinding(
                i,
                vk::DescriptorType::eStorageBuffer,
                1,
                vk::ShaderStageFlagBits::eCompute
            );
        }
        vk::DescriptorSetLayoutCreateInfo ds_info({}, bindings);
        auto ds_layout = dev.createDescriptorSetLayout(ds_info);

        vk::PushConstantRange pc_range(vk::ShaderStageFlagBits::eCompute, 0, push_constant_size);
        vk::PipelineLayoutCreateInfo layout_info({}, ds_layout, pc_range);
        auto layout = dev.createPipelineLayout(layout_info);

        vk::ShaderModuleCreateInfo shader_info({}, spv_size_words * sizeof(uint32_t), spv_data);
        auto shader_module = dev.createShaderModule(shader_info);

        vk::PipelineShaderStageCreateInfo stage_info({}, vk::ShaderStageFlagBits::eCompute, shader_module, "main");
        vk::ComputePipelineCreateInfo pipe_info({}, stage_info, layout);
        auto [result, pipeline] = dev.createComputePipeline(nullptr, pipe_info);

        dev.destroyShaderModule(shader_module);

        if (result != vk::Result::eSuccess) {
            dev.destroyDescriptorSetLayout(ds_layout);
            dev.destroyPipelineLayout(layout);
            throw std::runtime_error("VulkanContext: registerOp '" + name + "' failed (" + vk::to_string(result) + ")");
        }

        s.pipeline_infos.emplace(name, PipelineInfo{pipeline, layout, ds_layout, binding_count});
    }
}
void VulkanContext::bindPipeline(int device_id, const std::string& name) {
    auto& slot = this->slot(device_id);
    auto& pipe = slot.pipeline_infos[name];
    slot.recording_cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipe.pipeline);
    slot.current_layout = pipe.layout;        // 记录
}

void VulkanContext::bindDescriptorSet(int device_id, vk::DescriptorSet ds) {
    auto& slot = this->slot(device_id);
    slot.recording_cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute,slot.current_layout, 0, ds, {});
}

void VulkanContext::pushConstants(int device_id, const void* data, size_t size) {
    auto& slot = this->slot(device_id);
    slot.recording_cmd.pushConstants(slot.current_layout,vk::ShaderStageFlagBits::eCompute,0, size, data);
}

// ==================== Descriptor Set ====================
void VulkanContext::cleanupPendingFrees(int device_id) {
    auto& slot = this->slot(device_id);
    std::vector<vk::DescriptorSet> toFree;
    {
        std::lock_guard<std::mutex> lock(*slot.pending_frees_mutex);
        toFree.swap(slot.pending_frees);   // 取出所有待释放的集
    }
    if (toFree.empty()) return;
    slot.device.freeDescriptorSets(slot.descriptor_pool, toFree);
}
vk::DescriptorSet VulkanContext::updateDescriptorSets(int device_id, const std::string& name,std::initializer_list<vk::Buffer> buffers) {
    auto& slot = this->slot(device_id);
    auto it = slot.pipeline_infos.find(name);
    if (it == slot.pipeline_infos.end()) {
        throw std::runtime_error("Pipeline not registered: " + name);
    }
    auto& pipeinfo = it->second;
    if (buffers.size() != pipeinfo.binding_count) {
        throw std::runtime_error("Buffer count mismatch for pipeline " + name);
    }
    vk::DescriptorSetAllocateInfo allocInfo(slot.descriptor_pool, 1, &pipeinfo.ds_layout);
    vk::DescriptorSet descSet = slot.device.allocateDescriptorSets(allocInfo).front();

    std::vector<vk::DescriptorBufferInfo> bufferInfos;
    bufferInfos.reserve(buffers.size());
    for (auto& buf : buffers) {
        bufferInfos.emplace_back(buf, 0, VK_WHOLE_SIZE);
    }

    std::vector<vk::WriteDescriptorSet> writes;
    writes.reserve(buffers.size());
    for (uint32_t i = 0; i < bufferInfos.size(); ++i) {
        writes.emplace_back(
            descSet,    // dstSet
            i,          // dstBinding
            0,
            1,
            vk::DescriptorType::eStorageBuffer,
            nullptr,
            &bufferInfos[i],
            nullptr
        );
    }
    slot.device.updateDescriptorSets(writes, {});
    return descSet;
}

vk::DescriptorSet VulkanContext::updateDescriptorSets(int device_id, const std::string& name, std::initializer_list<vk::DescriptorBufferInfo> buffer_infos) {
    auto& slot = this->slot(device_id);
    auto it = slot.pipeline_infos.find(name);
    if (it == slot.pipeline_infos.end()) {
        throw std::runtime_error("Pipeline not registered: " + name);
    }
    auto& pipeinfo = it->second;
    if (buffer_infos.size() != pipeinfo.binding_count) {
        throw std::runtime_error("Buffer info count mismatch for pipeline " + name);
    }
    vk::DescriptorSetAllocateInfo allocInfo(slot.descriptor_pool, 1, &pipeinfo.ds_layout);
    vk::DescriptorSet descSet = slot.device.allocateDescriptorSets(allocInfo).front();

    std::vector<vk::WriteDescriptorSet> writes;
    writes.reserve(buffer_infos.size());
    uint32_t binding = 0;
    for (auto& info : buffer_infos) {
        writes.emplace_back(descSet, binding++, 0, 1,
            vk::DescriptorType::eStorageBuffer, nullptr, &info, nullptr);
    }
    slot.device.updateDescriptorSets(writes, {});
    return descSet;
}

vk::DescriptorSet VulkanContext::updateDescriptorSets(int device_id, const std::string& name,std::initializer_list<Tensor*> tensors) {
    auto& slot = this->slot(device_id);
    auto it = slot.pipeline_infos.find(name);
    if (it == slot.pipeline_infos.end()) {
        throw std::runtime_error("Pipeline not registered: " + name);
    }
    auto& pipeinfo = it->second;
    if (tensors.size() != pipeinfo.binding_count) {
        throw std::runtime_error("Tensor count mismatch for pipeline " + name);
    }
    // 分配描述符集
    vk::DescriptorSetAllocateInfo allocInfo(slot.descriptor_pool, 1, &pipeinfo.ds_layout);
    vk::DescriptorSet descSet = slot.device.allocateDescriptorSets(allocInfo).front();
    // 准备写入：bufferInfo 必须活得比 writes 久
    std::vector<vk::DescriptorBufferInfo> bufferInfos;
    bufferInfos.reserve(tensors.size());
    for (Tensor* tensor : tensors) {
        bufferInfos.emplace_back(
            reinterpret_cast<VkBuffer>(tensor->device_handle),
            tensor->offset,
            tensor->bytes()
        );
    }
    std::vector<vk::WriteDescriptorSet> writes;
    writes.reserve(bufferInfos.size());
    uint32_t binding = 0;
    for (auto& info : bufferInfos) {
        writes.emplace_back(
            descSet, binding++, 0, 1,
            vk::DescriptorType::eStorageBuffer,
            nullptr, &info, nullptr
        );
    }
    slot.device.updateDescriptorSets(writes, nullptr);
    return descSet;
}

void VulkanContext::beginRecording(int device_id) {
    auto& slot = this->slot(device_id);
    std::lock_guard<std::mutex> lock(*slot.record_mutex);
    if (slot.is_recording) {
        throw std::runtime_error("beginRecording: already recording on this device");
    }
    slot.recording_cmd.reset();
    vk::CommandBufferBeginInfo beginInfo{};
    beginInfo.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    slot.recording_cmd.begin(beginInfo);
    slot.is_recording = true;
}
void VulkanContext::endRecordingAndWait(int device_id) {
    auto& slot = this->slot(device_id);
    std::lock_guard<std::mutex> lock(*slot.record_mutex);
    if (!slot.is_recording)     return;
    slot.recording_cmd.end();
    slot.is_recording = false;
    vk::SubmitInfo submitInfo{};
    submitInfo.setCommandBuffers(slot.recording_cmd);
    if (slot.last_fence) {
        vk::Result waitResult = slot.device.waitForFences(slot.last_fence, VK_TRUE, UINT64_MAX);
        if (waitResult != vk::Result::eSuccess)
            throw std::runtime_error("waitForFences failed with error: " + vk::to_string(waitResult));
        slot.device.resetFences(slot.last_fence);
    } else {
        slot.last_fence = slot.device.createFence({});
    }
    slot.compute_queue.submit(submitInfo, slot.last_fence);
    vk::Result waitResult = slot.device.waitForFences(slot.last_fence, VK_TRUE, UINT64_MAX);
    if (waitResult != vk::Result::eSuccess) {
        throw std::runtime_error("waitForFences on current fence failed");
    }
}

bool VulkanContext::isRecording(int device_id) const {
    return slot(device_id).is_recording;
}

void VulkanContext::deferFreeDescriptorSet(int device_id, vk::DescriptorSet ds) {
    auto& slot = this->slot(device_id);
    std::lock_guard<std::mutex> lock(*slot.pending_frees_mutex);
    slot.pending_frees.push_back(ds);
}
void VulkanContext::dispatch(int device_id, uint32_t x, uint32_t y, uint32_t z) {
    auto& slot = this->slot(device_id);
    if (!slot.is_recording) throw std::runtime_error("dispatch: not recording");

    vk::MemoryBarrier2 barrier(
        vk::PipelineStageFlagBits2::eComputeShader,
        vk::AccessFlagBits2::eShaderStorageWrite,
        vk::PipelineStageFlagBits2::eComputeShader,
        vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite);
    vk::DependencyInfo dep({}, barrier, {});
    slot.recording_cmd.pipelineBarrier2(dep);

    slot.recording_cmd.dispatch(x, y, z);
}
vk::Buffer VulkanContext::createStagingBuffer(int device_id, size_t size,
                                               vk::DeviceMemory* out_memory, void** out_mapped) {
    auto& slot = this->slot(device_id);
    vk::BufferCreateInfo bufInfo({}, size,
                                 vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
                                 vk::SharingMode::eExclusive);
    vk::Buffer buffer = slot.device.createBuffer(bufInfo);

    vk::MemoryRequirements memReqs = slot.device.getBufferMemoryRequirements(buffer);
    vk::PhysicalDeviceMemoryProperties memProps = slot.physical_device.getMemoryProperties();

    uint32_t memTypeIndex = 0;
    for (; memTypeIndex < memProps.memoryTypeCount; ++memTypeIndex) {
        if ((memReqs.memoryTypeBits & (1 << memTypeIndex)) &&
            (memProps.memoryTypes[memTypeIndex].propertyFlags &
             (vk::MemoryPropertyFlagBits::eHostVisible |
              vk::MemoryPropertyFlagBits::eHostCoherent))) {
            break;
        }
    }
    if (memTypeIndex == memProps.memoryTypeCount) {
        slot.device.destroyBuffer(buffer);
        throw std::runtime_error("No suitable memory type for staging buffer");
    }

    vk::MemoryAllocateInfo allocInfo(memReqs.size, memTypeIndex);
    vk::DeviceMemory memory = slot.device.allocateMemory(allocInfo);
    slot.device.bindBufferMemory(buffer, memory, 0);

    void* mapped = slot.device.mapMemory(memory, 0, size);
    *out_memory = memory;
    *out_mapped = mapped;
    return buffer;
}

void VulkanContext::destroyStagingBuffer(int device_id, vk::Buffer buffer,
                                         vk::DeviceMemory memory, void* mapped) {
    auto& slot = this->slot(device_id);
    if (mapped) slot.device.unmapMemory(memory);
    slot.device.destroyBuffer(buffer);
    slot.device.freeMemory(memory);
}
vk::CommandBuffer VulkanContext::cmdBuffer(int device_id) const {
    const auto& slot = this->slot(device_id);
    if (!slot.is_recording) {
        throw std::runtime_error("cmdBuffer: not in recording mode");
    }
    return slot.recording_cmd;
}
void VulkanContext::addBufferBarrier(int device_id, vk::Buffer buffer,
                                     vk::AccessFlags srcAccessMask, vk::AccessFlags dstAccessMask,
                                     vk::PipelineStageFlags srcStage, vk::PipelineStageFlags dstStage) {
    auto& slot = this->slot(device_id);
    if (!slot.is_recording) {
        throw std::runtime_error("addBufferBarrier: not in recording mode");
    }
    vk::BufferMemoryBarrier2 barrier(
        static_cast<vk::PipelineStageFlagBits2>(static_cast<uint32_t>(srcStage)),
        static_cast<vk::AccessFlagBits2>(static_cast<uint32_t>(srcAccessMask)),
        static_cast<vk::PipelineStageFlagBits2>(static_cast<uint32_t>(dstStage)),
        static_cast<vk::AccessFlagBits2>(static_cast<uint32_t>(dstAccessMask)),
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        buffer, 0, VK_WHOLE_SIZE);
    vk::DependencyInfo dep({}, {}, barrier);
    slot.recording_cmd.pipelineBarrier2(dep);
}
void VulkanContext::cleanupPendingStagingBuffers(int device_id) {
    auto& slot = this->slot(device_id);
    std::lock_guard<std::mutex> lock(*slot.pending_frees_mutex);

    // 先处理回读：将 staging buffer 数据拷贝到 CPU 内存
    for (auto& rb : slot.pending_readbacks) {
        std::memcpy(rb.dst_cpu, rb.staging_ptr, rb.size);
        slot.device.unmapMemory(rb.staging_mem);
        slot.device.destroyBuffer(rb.staging_buf);
        slot.device.freeMemory(rb.staging_mem);
    }
    slot.pending_readbacks.clear();

    // 再处理普通的 staging buffer（仅用作临时拷贝源，无需回读）
    for (auto& [buf, mem, ptr] : slot.pending_staging_frees) {
        if (ptr) slot.device.unmapMemory(mem);
        slot.device.destroyBuffer(buf);
        slot.device.freeMemory(mem);
    }
    slot.pending_staging_frees.clear();
}
#endif // BACKEND_VULKAN
