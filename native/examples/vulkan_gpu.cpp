// Standalone DLL consumer: native production -> public GPU C API -> native
// byte comparison. Only integer validation flags are mapped on the CPU.
// Shader assets are compiled from interop/common.rs by interop_shaders.
#define VK_NO_PROTOTYPES
#define NOMINMAX
#include "oidn_webgpu_vulkan.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <windows.h>

#define INSTANCE_FUNCTIONS(X)                                                                      \
  X(CreateInstance)                                                                                \
  X(EnumerateInstanceExtensionProperties)                                                          \
  X(DestroyInstance) X(EnumeratePhysicalDevices) X(GetPhysicalDeviceProperties)                    \
      X(GetPhysicalDeviceFeatures2) X(GetPhysicalDeviceQueueFamilyProperties)                      \
          X(GetPhysicalDeviceMemoryProperties) X(EnumerateDeviceExtensionProperties)               \
              X(CreateDevice) X(GetDeviceProcAddr) X(CreateDebugUtilsMessengerEXT)                 \
                  X(DestroyDebugUtilsMessengerEXT)
#define DEVICE_FUNCTIONS(X)                                                                        \
  X(DestroyDevice)                                                                                 \
  X(GetDeviceQueue)                                                                                \
  X(DeviceWaitIdle) X(CreateBuffer) X(DestroyBuffer) X(GetBufferMemoryRequirements)                \
      X(AllocateMemory) X(FreeMemory) X(BindBufferMemory) X(MapMemory) X(UnmapMemory)              \
          X(CreateSemaphore) X(DestroySemaphore) X(GetSemaphoreCounterValue) X(SignalSemaphore)    \
              X(WaitSemaphores) X(QueueSubmit) X(CreateCommandPool) X(DestroyCommandPool)          \
                  X(AllocateCommandBuffers) X(BeginCommandBuffer) X(EndCommandBuffer)              \
                      X(CmdBindPipeline) X(CmdBindDescriptorSets) X(CmdPushConstants)              \
                          X(CmdDispatch) X(CreateDescriptorSetLayout)                              \
                              X(DestroyDescriptorSetLayout) X(CreatePipelineLayout)                \
                                  X(DestroyPipelineLayout) X(CreateShaderModule)                   \
                                      X(DestroyShaderModule) X(CreateComputePipelines)             \
                                          X(DestroyPipeline) X(CreateDescriptorPool)               \
                                              X(DestroyDescriptorPool) X(AllocateDescriptorSets)   \
                                                  X(UpdateDescriptorSets)
#define DECLARE(name) PFN_vk##name vk##name = nullptr;
INSTANCE_FUNCTIONS(DECLARE)
DEVICE_FUNCTIONS(DECLARE)
#undef DECLARE
static PFN_vkGetInstanceProcAddr gipa;
static VkInstance instance;
static VkPhysicalDevice physical;
static VkDevice device;
static uint32_t family;
static VkQueue queues[3];
static std::mutex queue_mutex;
static std::atomic<unsigned> validation_errors{0}, naga_diagnostics{0};
static std::atomic<unsigned> validation_warnings{0}, validation_timeouts{0};
static char error_text[4096];
static void trace(const std::string &text) {
  if (getenv("OIDNW_INTEROP_TRACE"))
    std::cerr << text << std::endl;
}
static void check(int32_t status) {
  if (status)
    throw std::runtime_error("C API " + std::to_string(status) + ": " + error_text);
}
static void vkcheck(VkResult status) {
  if (status)
    throw std::runtime_error("Vulkan " + std::to_string(status));
}
#define API(call) check(call)
#define ERR error_text, sizeof error_text
static void lock_queue(void *) noexcept { queue_mutex.lock(); }
static void unlock_queue(void *) noexcept { queue_mutex.unlock(); }
static VKAPI_ATTR VkBool32 VKAPI_CALL
debug_message(VkDebugUtilsMessageSeverityFlagBitsEXT severity, VkDebugUtilsMessageTypeFlagsEXT,
              const VkDebugUtilsMessengerCallbackDataEXT *data, void *) {
  std::string message(data->pMessage);
  if (uint32_t(data->messageIdNumber) == 0xb210f7c2u &&
      message.find("Workgroup") != std::string::npos &&
      message.find("ArrayStride") != std::string::npos)
    ++naga_diagnostics;
  else if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
    ++validation_errors;
  else if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
    ++validation_warnings;
  if (message.find("timeout waiting for timeline semaphore state") != std::string::npos)
    ++validation_timeouts;
  std::cerr << message << '\n';
  return VK_FALSE;
}
static std::vector<char> read_file(const std::string &path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input)
    throw std::runtime_error("cannot open " + path);
  std::vector<char> result(size_t(input.tellg()));
  input.seekg(0);
  input.read(result.data(), result.size());
  return result;
}
struct Buffer {
  VkBuffer raw;
  VkDeviceMemory memory;
  uint64_t size;
  oidnw_gpu_buffer *imported = nullptr;
};
static Buffer buffer(uint64_t bytes, bool cpu, oidnw_gpu_device *library) {
  Buffer result{};
  result.size = bytes;
  VkBufferCreateInfo create{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  create.size = bytes;
  create.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  create.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  vkcheck(vkCreateBuffer(device, &create, nullptr, &result.raw));
  VkMemoryRequirements requirements;
  vkGetBufferMemoryRequirements(device, result.raw, &requirements);
  VkPhysicalDeviceMemoryProperties properties;
  vkGetPhysicalDeviceMemoryProperties(physical, &properties);
  VkMemoryPropertyFlags wanted =
      cpu ? VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
          : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
  uint32_t index = 0;
  for (; index < properties.memoryTypeCount; ++index)
    if ((requirements.memoryTypeBits & (1u << index)) &&
        (properties.memoryTypes[index].propertyFlags & wanted) == wanted)
      break;
  if (index == properties.memoryTypeCount)
    throw std::runtime_error("no compatible memory type");
  VkMemoryAllocateInfo allocate{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  allocate.allocationSize = requirements.size;
  allocate.memoryTypeIndex = index;
  vkcheck(vkAllocateMemory(device, &allocate, nullptr, &result.memory));
  vkcheck(vkBindBufferMemory(device, result.raw, result.memory, 0));
  if (library)
    API(oidnw_gpu_buffer_import_vulkan(library, result.raw, &create, &result.imported, ERR));
  return result;
}
static VkSemaphore semaphore() {
  VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
  type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  VkSemaphoreCreateInfo create{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  create.pNext = &type;
  VkSemaphore result;
  vkcheck(vkCreateSemaphore(device, &create, nullptr, &result));
  return result;
}
static oidnw_gpu_sync *import_sync(oidnw_gpu_device *library, VkSemaphore raw) {
  VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
  type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  VkSemaphoreCreateInfo create{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  create.pNext = &type;
  oidnw_gpu_sync *result;
  API(oidnw_gpu_sync_import_vulkan(library, raw, &create, &result, ERR));
  return result;
}
static uint64_t value(VkSemaphore raw) {
  uint64_t v;
  vkcheck(vkGetSemaphoreCounterValue(device, raw, &v));
  return v;
}
static void signal(VkSemaphore raw) {
  VkSemaphoreSignalInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
  info.semaphore = raw;
  info.value = 1;
  vkcheck(vkSignalSemaphore(device, &info));
}
static void wait(VkSemaphore raw, uint64_t v) {
  VkSemaphoreWaitInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
  info.semaphoreCount = 1;
  info.pSemaphores = &raw;
  info.pValues = &v;
  vkcheck(vkWaitSemaphores(device, &info, 60000000000ull));
}
struct Point {
  VkSemaphore sync;
  uint64_t value;
};
static void submit(VkQueue queue, VkCommandBuffer command, const std::vector<Point> &waits,
                   Point signal) {
  std::vector<VkSemaphore> sems;
  std::vector<uint64_t> values;
  std::vector<VkPipelineStageFlags> stages;
  for (auto p : waits) {
    sems.push_back(p.sync);
    values.push_back(p.value);
    stages.push_back(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
  }
  VkTimelineSemaphoreSubmitInfo timeline{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
  timeline.waitSemaphoreValueCount = uint32_t(values.size());
  timeline.pWaitSemaphoreValues = values.data();
  timeline.signalSemaphoreValueCount = 1;
  timeline.pSignalSemaphoreValues = &signal.value;
  VkSubmitInfo info{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  info.pNext = &timeline;
  info.waitSemaphoreCount = uint32_t(sems.size());
  info.pWaitSemaphores = sems.data();
  info.pWaitDstStageMask = stages.data();
  info.commandBufferCount = command ? 1 : 0;
  info.pCommandBuffers = &command;
  info.signalSemaphoreCount = 1;
  info.pSignalSemaphores = &signal.sync;
  std::lock_guard<std::mutex> guard(queue_mutex);
  vkcheck(vkQueueSubmit(queue, 1, &info, VK_NULL_HANDLE));
}
struct Batch {
  VkCommandPool commands;
  VkDescriptorPool descriptors;
  VkCommandBuffer command;
};
struct Kernels {
  VkDescriptorSetLayout sets;
  VkPipelineLayout layout;
  VkPipeline pipelines[2];
  void init(const std::string &directory) {
    VkDescriptorSetLayoutBinding bindings[3]{};
    for (uint32_t i = 0; i < 3; ++i) {
      bindings[i].binding = i;
      bindings[i].descriptorCount = 1;
      bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo set{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    set.bindingCount = 3;
    set.pBindings = bindings;
    vkcheck(vkCreateDescriptorSetLayout(device, &set, nullptr, &sets));
    VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, 16};
    VkPipelineLayoutCreateInfo pipeline{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipeline.setLayoutCount = 1;
    pipeline.pSetLayouts = &sets;
    pipeline.pushConstantRangeCount = 1;
    pipeline.pPushConstantRanges = &push;
    vkcheck(vkCreatePipelineLayout(device, &pipeline, nullptr, &layout));
    const char *names[] = {"produce", "consume"};
    for (int i = 0; i < 2; ++i) {
      auto bytes = read_file(directory + "/" + names[i] + ".spv");
      VkShaderModuleCreateInfo shader{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
      shader.codeSize = bytes.size();
      shader.pCode = reinterpret_cast<const uint32_t *>(bytes.data());
      VkShaderModule module;
      vkcheck(vkCreateShaderModule(device, &shader, nullptr, &module));
      VkComputePipelineCreateInfo create{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
      create.layout = layout;
      create.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      create.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
      create.stage.module = module;
      create.stage.pName = names[i];
      vkcheck(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &create, nullptr, &pipelines[i]));
      vkDestroyShaderModule(device, module, nullptr);
    }
  }
  Batch batch(bool consume, std::array<Buffer *, 3> buffers, uint32_t size, uint32_t pattern,
              uint32_t frame) {
    Batch result{};
    VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool.queueFamilyIndex = family;
    vkcheck(vkCreateCommandPool(device, &pool, nullptr, &result.commands));
    VkCommandBufferAllocateInfo allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocate.commandPool = result.commands;
    allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocate.commandBufferCount = 1;
    vkcheck(vkAllocateCommandBuffers(device, &allocate, &result.command));
    VkDescriptorPoolSize count{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
    VkDescriptorPoolCreateInfo descriptors{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    descriptors.maxSets = 1;
    descriptors.poolSizeCount = 1;
    descriptors.pPoolSizes = &count;
    vkcheck(vkCreateDescriptorPool(device, &descriptors, nullptr, &result.descriptors));
    VkDescriptorSetAllocateInfo set{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    set.descriptorPool = result.descriptors;
    set.descriptorSetCount = 1;
    set.pSetLayouts = &sets;
    VkDescriptorSet descriptor;
    vkcheck(vkAllocateDescriptorSets(device, &set, &descriptor));
    VkDescriptorBufferInfo ranges[3]{};
    VkWriteDescriptorSet writes[3]{};
    for (uint32_t i = 0; i < 3; ++i) {
      ranges[i] = {buffers[i]->raw, consume && i == 2 ? 0ull : 256ull,
                   consume && i == 2 ? buffers[i]->size : uint64_t(size) * size * 16};
      writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[i].dstSet = descriptor;
      writes[i].dstBinding = i;
      writes[i].descriptorCount = 1;
      writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      writes[i].pBufferInfo = &ranges[i];
    }
    vkUpdateDescriptorSets(device, 3, writes, 0, nullptr);
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkcheck(vkBeginCommandBuffer(result.command, &begin));
    vkCmdBindPipeline(result.command, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines[consume ? 1 : 0]);
    vkCmdBindDescriptorSets(result.command, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1,
                            &descriptor, 0, nullptr);
    uint32_t params[] = {size, pattern, frame, 0};
    vkCmdPushConstants(result.command, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof params,
                       params);
    vkCmdDispatch(result.command, (size * size + 63) / 64, 1, 1);
    vkcheck(vkEndCommandBuffer(result.command));
    return result;
  }
};
static void release(Batch b) {
  vkDestroyCommandPool(device, b.commands, nullptr);
  vkDestroyDescriptorPool(device, b.descriptors, nullptr);
}
static oidnw_gpu_images images(const std::array<Buffer, 4> &buffers) {
  return {{buffers[0].imported, 256},
          {buffers[1].imported, 256},
          {buffers[2].imported, 256},
          {buffers[3].imported, 256}};
}
int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "usage: vulkan_gpu WEIGHTS SHADER_DIR [frames=1000] [size=16] [tile=0] [fp32=0] "
                 "[shared=0] [checks=1]\n";
    return 2;
  }
  try {
    if (oidnw_api_version() != OIDNW_API_VERSION)
      throw std::runtime_error("header/DLL version mismatch");
    unsigned frames = argc > 3 ? std::stoul(argv[3]) : 1000,
             size = argc > 4 ? std::stoul(argv[4]) : 16;
    unsigned tile = argc > 5 ? std::stoul(argv[5]) : 0;
    bool fp32 = argc > 6 && std::stoi(argv[6]), shared = argc > 7 && std::stoi(argv[7]);
    bool checks = argc <= 8 || std::stoi(argv[8]);
    const bool gpu_validation =
        getenv("VK_LAYER_GPUAV_ENABLE") && !strcmp(getenv("VK_LAYER_GPUAV_ENABLE"), "1");
    if (!frames || frames > 100000 || !size || size > 1024 ||
        uint64_t(frames) * size * size * 16 > 256 * 1024 * 1024)
      throw std::runtime_error("invalid size/frames");
    auto loader = LoadLibraryW(L"vulkan-1.dll");
    if (!loader)
      throw std::runtime_error("Vulkan loader missing");
    gipa = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
        GetProcAddress(loader, "vkGetInstanceProcAddr"));
#define LOAD_GLOBAL(name) vk##name = reinterpret_cast<PFN_vk##name>(gipa(nullptr, "vk" #name));
    LOAD_GLOBAL(CreateInstance)
    LOAD_GLOBAL(EnumerateInstanceExtensionProperties)
#undef LOAD_GLOBAL
    uint32_t count = 0;
    auto extensions = oidnw::vulkan::instance_extensions(gipa, VK_API_VERSION_1_3);
    if (std::none_of(extensions.begin(), extensions.end(), [](const char *name) {
          return !strcmp(name, VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        })) extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    const char *layer = "VK_LAYER_KHRONOS_validation";
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo instance_create{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instance_create.pApplicationInfo = &app;
    instance_create.enabledExtensionCount = uint32_t(extensions.size());
    instance_create.ppEnabledExtensionNames = extensions.data();
    instance_create.enabledLayerCount = 1;
    instance_create.ppEnabledLayerNames = &layer;
    vkcheck(vkCreateInstance(&instance_create, nullptr, &instance));
#define LOAD_INSTANCE(name) vk##name = reinterpret_cast<PFN_vk##name>(gipa(instance, "vk" #name));
    INSTANCE_FUNCTIONS(LOAD_INSTANCE)
#undef LOAD_INSTANCE
    VkDebugUtilsMessengerCreateInfoEXT debug{
        VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
    debug.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT |
                            VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT;
    debug.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                        VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                        VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    debug.pfnUserCallback = debug_message;
    VkDebugUtilsMessengerEXT messenger;
    vkcheck(vkCreateDebugUtilsMessengerEXT(instance, &debug, nullptr, &messenger));
    vkcheck(vkEnumeratePhysicalDevices(instance, &count, nullptr));
    std::vector<VkPhysicalDevice> adapters(count);
    vkcheck(vkEnumeratePhysicalDevices(instance, &count, adapters.data()));
    VkPhysicalDeviceProperties properties{};
    for (auto candidate : adapters) {
      vkGetPhysicalDeviceProperties(candidate, &properties);
      if (strstr(properties.deviceName,
                 getenv("OIDNW_INTEROP_ADAPTER") ? getenv("OIDNW_INTEROP_ADAPTER") : "RTX 4060")) {
        physical = candidate;
        break;
      }
    }
    if (!physical)
      throw std::runtime_error("requested adapter not found");
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, families.data());
    for (family = 0; family < count; ++family)
      if (families[family].queueCount >= 3 && (families[family].queueFlags & 3) == 3)
        break;
    if (family == count)
      throw std::runtime_error("requires three graphics+compute queues");
    oidnw_gpu_vulkan_requirements_desc query;
    oidnw_gpu_vulkan_requirements_desc_default(&query);
    query.flags = fp32 ? OIDNW_GPU_VULKAN_FP32 : 0;
    query.get_instance_proc_addr = gipa;
    query.instance = instance;
    query.physical_device = physical;
    query.instance_create_info = &instance_create;
    auto requirements = oidnw::vulkan::query_requirements(query);
    auto device_create = oidnw::vulkan::device_create_info(requirements);
    float priorities[] = {1, 1, 1};
    VkDeviceQueueCreateInfo queue_create{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue_create.queueFamilyIndex = family;
    queue_create.queueCount = 3;
    queue_create.pQueuePriorities = priorities;
    device_create.queueCreateInfoCount = 1;
    device_create.pQueueCreateInfos = &queue_create;
    vkcheck(vkCreateDevice(physical, &device_create, nullptr, &device));
#define LOAD_DEVICE(name)                                                                          \
  vk##name = reinterpret_cast<PFN_vk##name>(vkGetDeviceProcAddr(device, "vk" #name));
    DEVICE_FUNCTIONS(LOAD_DEVICE)
#undef LOAD_DEVICE
    for (int i = 0; i < 3; ++i)
      vkGetDeviceQueue(device, family, i, &queues[i]);
    oidnw_gpu_vulkan_device_desc desc;
    oidnw_gpu_vulkan_device_desc_default(&desc);
    desc.flags = fp32 ? OIDNW_GPU_VULKAN_FP32 : 0;
    desc.get_instance_proc_addr = gipa;
    desc.instance = instance;
    desc.physical_device = physical;
    desc.device = device;
    desc.queue = queues[shared ? 0 : 1];
    desc.queue_family_index = family;
    desc.queue_index = shared ? 0 : 1;
    desc.instance_create_info = &instance_create;
    desc.device_create_info = &device_create;
    desc.lock = lock_queue;
    desc.unlock = unlock_queue;
    desc.max_pending_submissions = 8;
    trace("import device");
    oidnw_gpu_device *library;
    API(oidnw_gpu_device_import_vulkan(&desc, &library, ERR));
    uint32_t backend, enabled;
    API(oidnw_gpu_device_info(library, nullptr, 0, &backend, &enabled));
    if (backend != OIDNW_BACKEND_VULKAN)
      throw std::runtime_error("wrong backend");
    // Real DLL rejection of a missing HAL feature, extension and aggregate chain.
    oidnw_gpu_device *rejected = nullptr;
    auto core = *device_create.pEnabledFeatures;
    auto original_core = device_create.pEnabledFeatures;
    core.robustBufferAccess = 0;
    device_create.pEnabledFeatures = &core;
    if (oidnw_gpu_device_import_vulkan(&desc, &rejected, ERR) != OIDNW_ERROR_INVALID_ARGUMENT ||
        rejected)
      throw std::runtime_error("missing HAL feature accepted");
    device_create.pEnabledFeatures = original_core;
    auto extension_count = device_create.enabledExtensionCount;
    device_create.enabledExtensionCount = 0;
    if (oidnw_gpu_device_import_vulkan(&desc, &rejected, ERR) != OIDNW_ERROR_INVALID_ARGUMENT ||
        rejected)
      throw std::runtime_error("missing HAL extensions accepted");
    device_create.enabledExtensionCount = extension_count;
    VkPhysicalDeviceVulkan12Features aggregate{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    auto chain = device_create.pNext;
    device_create.pNext = &aggregate;
    if (oidnw_gpu_device_import_vulkan(&desc, &rejected, ERR) != OIDNW_ERROR_INVALID_ARGUMENT ||
        rejected)
      throw std::runtime_error("aggregate features accepted");
    device_create.pNext = chain;
    uint64_t bytes = uint64_t(size) * size * 16;
    std::array<std::array<Buffer, 4>, 3> ring;
    std::vector<Buffer *> allocations;
    for (auto &group : ring)
      for (auto &b : group) {
        b = buffer(256 + bytes, false, library);
        allocations.push_back(&b);
      }
    std::array<Buffer, 3> references;
    for (auto &b : references) {
      b = buffer(256 + bytes, false, library);
      allocations.push_back(&b);
    }
    auto flags = buffer(frames * bytes, true, nullptr);
    allocations.push_back(&flags);
    void *mapped;
    vkcheck(vkMapMemory(device, flags.memory, 0, flags.size, 0, &mapped));
    memset(mapped, 0xff, size_t(flags.size));
    vkUnmapMemory(device, flags.memory);
    VkSemaphore ready = semaphore(), done = semaphore(), consumed = semaphore(), gate = semaphore(),
                final_gate = semaphore();
    auto ready_handle = import_sync(library, ready), done_handle = import_sync(library, done);
    Kernels kernels;
    kernels.init(argv[2]);
    trace("prepare model");
    auto weights = read_file(argv[1]);
    oidnw_gpu_model *model;
    API(oidnw_gpu_model_create(weights.data(), weights.size(), &model, ERR));
    oidnw_gpu_filter_desc filter_desc;
    oidnw_gpu_filter_desc_default(&filter_desc);
    filter_desc.flags = checks ? OIDNW_FILTER_RUNTIME_CHECKS : 0;
    filter_desc.max_tile_size = tile;
    filter_desc.binding_cache_capacity = 3;
    oidnw_gpu_filter *filter;
    API(oidnw_gpu_filter_create(library, model, &filter_desc, &filter, ERR));
    oidnw_gpu_model_destroy(model);
    auto initial = images(ring[0]);
    oidnw_gpu_prepared *prepared;
    API(oidnw_gpu_prepare(filter, size, size, &initial, &prepared, ERR));
    oidnw_gpu_filter_destroy(filter);
    uint64_t last = 0;
    unsigned queue_full = 0;
    auto denoise = [&](oidnw_gpu_images bindings, uint64_t point, float scale) {
      oidnw_gpu_submit_desc submit;
      oidnw_gpu_submit_desc_default(&submit);
      submit.images = bindings;
      submit.input_scale = scale;
      oidnw_gpu_timeline_point w{ready_handle, point}, s{done_handle, point};
      submit.waits = &w;
      submit.wait_count = 1;
      submit.signals = &s;
      submit.signal_count = 1;
      for (;;) {
        uint64_t sequence = 99;
        auto status = oidnw_gpu_submit(prepared, &submit, &sequence, ERR);
        if (status == OIDNW_ERROR_QUEUE_FULL) {
          if (sequence)
            throw std::runtime_error("QUEUE_FULL accepted work");
          ++queue_full;
          uint32_t complete;
          API(oidnw_gpu_wait(library, last, 60000000000ull, &complete, ERR));
          continue;
        }
        API(status);
        if (sequence <= last)
          throw std::runtime_error("non-monotonic sequence");
        last = sequence;
        break;
      }
    };
    const float scales[] = {1, 2, 0.5};
    // Serialized GPU C references; the Rust oracle is also exercised separately.
    for (unsigned pattern = 0; pattern < 3; ++pattern) {
      trace("reference " + std::to_string(pattern));
      auto producer =
          kernels.batch(false, {&ring[0][0], &ring[0][1], &ring[0][2]}, size, pattern, 0);
      submit(queues[0], producer.command, {}, {ready, pattern + 1});
      wait(ready, pattern + 1);
      auto binding = initial;
      binding.output.buffer = references[pattern].imported;
      denoise(binding, pattern + 1, scales[pattern]);
      uint32_t complete;
      API(oidnw_gpu_wait(library, last, 60000000000ull, &complete, ERR));
      wait(done, pattern + 1);
      release(producer);
    }
    // No synchronization may be accepted for bad parameters or aliased output.
    oidnw_gpu_submit_desc bad;
    oidnw_gpu_submit_desc_default(&bad);
    bad.images = initial;
    bad.input_scale = -1;
    uint64_t rejected_sequence = 99;
    if (oidnw_gpu_submit(prepared, &bad, &rejected_sequence, ERR) != OIDNW_ERROR_INVALID_ARGUMENT ||
        rejected_sequence)
      throw std::runtime_error("bad scale accepted");
    bad.input_scale = 1;
    bad.images.output = bad.images.color;
    rejected_sequence = 99;
    if (oidnw_gpu_submit(prepared, &bad, &rejected_sequence, ERR) != OIDNW_ERROR_INVALID_ARGUMENT ||
        rejected_sequence)
      throw std::runtime_error("aliased output accepted");
    std::atomic<bool> gate_safe{false}, gate_released{false};
    std::thread producer_release([&] {
      std::this_thread::sleep_for(std::chrono::milliseconds(250));
      // GPUAV's counter query can itself wait on layer state. Never make the
      // producer's release depend on such a query in that diagnostic protocol.
      gate_safe = gpu_validation || value(done) == 3;
      gate_released = true;
      signal(gate);
    });
    struct Pending {
      uint64_t value;
      Batch producer, consumer;
    };
    std::deque<Pending> pending;
    size_t high_water = 0;
    for (unsigned frame = 0; frame < frames; ++frame) {
      trace("frame " + std::to_string(frame));
      // HAL binary relay chains cannot cross deeply unresolved CPU timeline
      // signals. Resolve the first gate before chaining the second execution.
      if (frame == 1)
        wait(gate, 1);
      if (pending.size() == 64)
        wait(consumed, pending.front().value);
      auto completed = value(consumed);
      while (!pending.empty() && pending.front().value <= completed) {
        release(pending.front().producer);
        release(pending.front().consumer);
        pending.pop_front();
      }
      auto slot = frame % 3, pattern = (frame + frame / 3) % 3;
      auto &group = ring[slot];
      uint64_t n = frame + 1;
      auto producer = kernels.batch(false, {&group[0], &group[1], &group[2]}, size, pattern, 0);
      std::vector<Point> waits{{gate, 1}};
      if (frame >= 3)
        waits.push_back({consumed, n - 3});
      if (frame + 1 == frames)
        waits.push_back({final_gate, 1});
      trace("producer");
      submit(queues[0], producer.command, waits, {ready, n + 3});
      if (gpu_validation) {
        if (frame + 1 == frames)
          signal(final_gate);
        wait(ready, n + 3);
      }
      trace("denoise");
      denoise(images(group), n + 3, scales[pattern]);
      if (gpu_validation) {
        uint32_t complete;
        API(oidnw_gpu_wait(library, last, 60000000000ull, &complete, ERR));
      }
      trace("consumer");
      auto consumer =
          kernels.batch(true, {&group[3], &references[pattern], &flags}, size, pattern, frame);
      submit(queues[2], consumer.command, {{done, n + 3}}, {consumed, n});
      pending.push_back({n, producer, consumer});
      high_water = std::max(high_water, pending.size());
      if (frame == 0 && !gpu_validation) {
        uint32_t complete = 99;
        auto status = oidnw_gpu_wait(library, last, 0, &complete, ERR);
        if (!gate_released && (status != OIDNW_ERROR_TIMEOUT || complete))
          throw std::runtime_error("gated work completed early");
      }
    }
    producer_release.join();
    if (!gate_safe)
      throw std::runtime_error("external signal occurred before producer gate");
    oidnw_gpu_prepared_destroy(prepared);
    for (auto b : allocations)
      oidnw_gpu_buffer_destroy(b->imported);
    oidnw_gpu_sync_destroy(ready_handle);
    oidnw_gpu_sync_destroy(done_handle);
    uint32_t complete = 99;
    if (!gpu_validation) {
      if (oidnw_gpu_wait(library, last, 0, &complete, ERR) != OIDNW_ERROR_TIMEOUT || complete)
        throw std::runtime_error("in-flight resources completed early");
      if (oidnw_gpu_device_destroy(library, ERR) != OIDNW_ERROR_BUSY)
        throw std::runtime_error("in-flight device destroyed");
      signal(final_gate);
    }
    API(oidnw_gpu_wait(library, last, 60000000000ull, &complete, ERR));
    wait(consumed, frames);
    for (auto p : pending) {
      release(p.producer);
      release(p.consumer);
    }
    pending.clear();
    API(oidnw_gpu_device_destroy(library, ERR));
    vkcheck(vkMapMemory(device, flags.memory, 0, flags.size, 0, &mapped));
    auto words = static_cast<uint32_t *>(mapped);
    for (uint64_t i = 0; i < flags.size / 4; ++i)
      if (words[i])
        throw std::runtime_error("GPU consumer mismatch word " + std::to_string(i) + " flags " +
                                 std::to_string(words[i]));
    vkUnmapMemory(device, flags.memory);
    for (auto b : allocations) {
      vkDestroyBuffer(device, b->raw, nullptr);
      vkFreeMemory(device, b->memory, nullptr);
    }
    for (auto s : {ready, done, consumed, gate, final_gate})
      vkDestroySemaphore(device, s, nullptr);
    for (auto p : kernels.pipelines)
      vkDestroyPipeline(device, p, nullptr);
    vkDestroyPipelineLayout(device, kernels.layout, nullptr);
    vkDestroyDescriptorSetLayout(device, kernels.sets, nullptr);
    vkcheck(vkDeviceWaitIdle(device));
    requirements.reset();
    vkDestroyDevice(device, nullptr);
    vkDestroyDebugUtilsMessengerEXT(instance, messenger, nullptr);
    vkDestroyInstance(instance, nullptr);
    if (validation_errors)
      throw std::runtime_error("new Vulkan validation errors: " +
                               std::to_string(validation_errors.load()));
    if (validation_timeouts)
      throw std::runtime_error("validation layer timeline-state timeouts: " +
                               std::to_string(validation_timeouts.load()));
    char dll[MAX_PATH];
    GetModuleFileNameA(GetModuleHandleA("oidn_webgpu.dll"), dll, MAX_PATH);
    std::replace(dll, dll + strlen(dll), '\\', '/');
    std::cout << "{\"backend\":\"Vulkan\",\"formal_c_api\":true,\"dll\":\"" << dll
              << "\",\"frames\":" << frames << ",\"size\":" << size << ",\"tile\":" << tile
              << ",\"precision\":\"" << ((enabled & 1) ? "fp16" : "fp32")
              << "\",\"runtime_checks\":" << (checks ? "true" : "false") << ",\"queue\":\""
              << (shared ? "shared" : "dedicated")
              << "\",\"output_byte_equal\":true,\"cpu_image_uploads\":0,\"cpu_image_readbacks\":0,"
                 "\"validation_errors\":"
              << validation_errors
              << ",\"known_naga_workgroup_layout_diagnostics\":" << naga_diagnostics
              << ",\"validation_warnings\":" << validation_warnings
              << ",\"validation_layer_timeouts\":" << validation_timeouts
              << ",\"gpu_validation_serial_protocol\":" << (gpu_validation ? "true" : "false")
              << ",\"queue_full\":" << queue_full << ",\"host_pending_high_water\":" << high_water
              << ",\"producer_gate_verified\":" << (gpu_validation ? "false" : "true")
              << ",\"in_flight_release_verified\":" << (gpu_validation ? "false" : "true") << "}"
              << std::endl;
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
