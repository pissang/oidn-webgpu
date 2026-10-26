#include "oidn_webgpu_vulkan.hpp"
#include <cstdlib>
#include <cstring>
#include <type_traits>

template<class T> static void ensure(const T &condition) {
  if (!condition) std::abort();
}
struct oidnw_gpu_device { int unused = 0; };
struct oidnw_gpu_vulkan_requirements {
  const char *extensions[2] = {"VK_KHR_library", "VK_KHR_shared"};
  VkPhysicalDeviceFeatures core{};
  VkPhysicalDevice16BitStorageFeatures storage{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
  VkPhysicalDeviceTimelineSemaphoreFeatures timeline{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES};
};
static oidnw_gpu_vulkan_device_desc imported{};
static unsigned destroyed = 0;
static bool locked_during_import = false;
extern "C" {
uint32_t oidnw_api_version() { return OIDNW_API_VERSION; }
void oidnw_gpu_vulkan_device_desc_default(oidnw_gpu_vulkan_device_desc *p) {
  std::memset(p, 0, sizeof *p); p->max_pending_submissions = 64;
}
int32_t oidnw_gpu_device_import_vulkan(const oidnw_gpu_vulkan_device_desc *desc,
    oidnw_gpu_device **out, char *, size_t) {
  imported = *desc;
  desc->lock(desc->user_data);
  locked_during_import = !static_cast<oidnw::vulkan::queue_lock *>(desc->user_data)->try_lock();
  desc->unlock(desc->user_data);
  *out = new oidnw_gpu_device;
  return OIDNW_OK;
}
int32_t oidnw_gpu_wait(oidnw_gpu_device *, uint64_t, uint64_t, uint32_t *done, char *, size_t) {
  *done = 1; return OIDNW_OK;
}
int32_t oidnw_gpu_device_destroy(oidnw_gpu_device *d, char *, size_t) {
  if (d) { ++destroyed; delete d; }
  return OIDNW_OK;
}
int32_t oidnw_gpu_vulkan_requirements_create(const oidnw_gpu_vulkan_requirements_desc *,
    oidnw_gpu_vulkan_requirements **out, char *, size_t) {
  auto value = new oidnw_gpu_vulkan_requirements;
  value->core.robustBufferAccess = VK_TRUE;
  value->storage.storageBuffer16BitAccess = VK_TRUE;
  value->timeline.timelineSemaphore = VK_TRUE;
  value->storage.pNext = &value->timeline;
  *out = value;
  return OIDNW_OK;
}
int32_t oidnw_gpu_vulkan_requirements_get(const oidnw_gpu_vulkan_requirements *r,
    VkDeviceCreateInfo *info, uint32_t *, char *, size_t) {
  *info = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  info->pNext = &r->storage;
  info->enabledExtensionCount = 2;
  info->ppEnabledExtensionNames = r->extensions;
  info->pEnabledFeatures = &r->core;
  return OIDNW_OK;
}
void oidnw_gpu_vulkan_requirements_destroy(oidnw_gpu_vulkan_requirements *r) {
  if (r) { std::memset(static_cast<void *>(r), 0xcd, sizeof *r); delete r; }
}
}

static VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL fake_loader(VkInstance, const char *) { return nullptr; }
static oidnw::vulkan::requirements library() {
  oidnw_gpu_vulkan_requirements_desc query{};
  return oidnw::vulkan::query_requirements(query);
}
template<class F> static void rejects(F &&body) {
  try { body(); ensure(false); }
  catch (const oidnw::gpu::error &e) { ensure(e.code == OIDNW_ERROR_INVALID_ARGUMENT); }
}

int main() {
  using namespace oidnw;
  static_assert(!std::is_copy_constructible_v<vulkan::queue_lock>);
  static_assert(!std::is_copy_constructible_v<vulkan::merged_device_info>);
  static_assert(std::is_nothrow_move_constructible_v<vulkan::merged_device_info>);

  // queue_lock is BasicLockable and its callbacks drive the same mutex.
  {
    vulkan::queue_lock lock;
    { std::lock_guard<vulkan::queue_lock> guard(lock); ensure(!lock.try_lock()); }
    vulkan::queue_lock::lock_callback(&lock);
    ensure(!lock.try_lock());
    vulkan::queue_lock::unlock_callback(&lock);
    ensure(lock.try_lock());
    lock.unlock();
  }

  // import_device fills the descriptor and retains the lock with host_lifetime.
  VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  const vulkan::device_handles handles{fake_loader, reinterpret_cast<VkInstance>(1),
      reinterpret_cast<VkPhysicalDevice>(2), reinterpret_cast<VkDevice>(3), reinterpret_cast<VkQueue>(4), 5, 6};
  {
    auto lock = std::make_shared<vulkan::queue_lock>();
    std::weak_ptr<vulkan::queue_lock> lock_alive = lock;
    auto token = std::make_shared<int>(1);
    std::weak_ptr<int> token_alive = token;
    gpu::device_options options;
    options.host_lifetime = std::move(token);
    auto context = vulkan::import_device(handles, instance_info, device_info, std::move(lock),
        OIDNW_GPU_VULKAN_FP32, std::move(options));
    ensure(imported.get_instance_proc_addr == handles.get_instance_proc_addr);
    ensure(imported.instance == handles.instance && imported.physical_device == handles.physical_device);
    ensure(imported.device == handles.device && imported.queue == handles.queue);
    ensure(imported.queue_family_index == 5 && imported.queue_index == 6);
    ensure(imported.instance_create_info == &instance_info && imported.device_create_info == &device_info);
    ensure(imported.flags == OIDNW_GPU_VULKAN_FP32 && imported.max_pending_submissions == 64);
    ensure(locked_during_import);
    ensure(!lock_alive.expired() && !token_alive.expired());
    ensure(context.close());
    ensure(destroyed == 1 && lock_alive.expired() && token_alive.expired());
  }
  rejects([&] { vulkan::import_device(handles, instance_info, device_info, nullptr); });

  // merge_requirements: union of extensions, OR of features, owned copies.
  char host_name[] = "VK_KHR_shared";
  const char *host_extensions[] = {"VK_KHR_swapchain", host_name};
  VkDeviceQueueCreateInfo queue{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  VkPhysicalDeviceTimelineSemaphoreFeatures host_timeline{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES};
  VkPhysicalDeviceFeatures2 host_core{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &host_timeline};
  host_core.features.shaderInt64 = VK_TRUE;
  VkDeviceCreateInfo host{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &host_core};
  host.queueCreateInfoCount = 1;
  host.pQueueCreateInfos = &queue;
  host.enabledExtensionCount = 2;
  host.ppEnabledExtensionNames = host_extensions;

  auto merged = [&] { auto source = library(); return vulkan::merge_requirements(source, host); }();
  std::memset(host_name, 0, sizeof host_name);
  auto moved = std::move(merged);
  const VkDeviceCreateInfo &info = moved.get();
  ensure(info.queueCreateInfoCount == 1 && info.pQueueCreateInfos == &queue);
  ensure(info.enabledExtensionCount == 3);
  ensure(!std::strcmp(info.ppEnabledExtensionNames[0], "VK_KHR_swapchain"));
  ensure(!std::strcmp(info.ppEnabledExtensionNames[1], "VK_KHR_shared"));
  ensure(!std::strcmp(info.ppEnabledExtensionNames[2], "VK_KHR_library"));
  ensure(info.pEnabledFeatures && info.pEnabledFeatures->shaderInt64 && info.pEnabledFeatures->robustBufferAccess);
  ensure(!info.pEnabledFeatures->geometryShader);
  auto first = static_cast<const VkPhysicalDeviceTimelineSemaphoreFeatures *>(info.pNext);
  ensure(first && first != &host_timeline);
  ensure(first->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES && first->timelineSemaphore);
  auto second = static_cast<const VkPhysicalDevice16BitStorageFeatures *>(first->pNext);
  ensure(second && second->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES);
  ensure(second->storageBuffer16BitAccess && !second->storagePushConstant16 && !second->pNext);
  ensure(!host_timeline.timelineSemaphore && host_timeline.pNext == nullptr);

  // Declarations import would reject fail while merging.
  auto source = library();
  {
    VkDeviceCreateInfo both = host;
    VkPhysicalDeviceFeatures core{};
    both.pEnabledFeatures = &core;
    rejects([&] { vulkan::merge_requirements(source, both); });
  }
  {
    VkPhysicalDeviceVulkan12Features aggregate{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkDeviceCreateInfo bad{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &aggregate};
    rejects([&] { vulkan::merge_requirements(source, bad); });
  }
  {
    VkDeviceGroupDeviceCreateInfo group{VK_STRUCTURE_TYPE_DEVICE_GROUP_DEVICE_CREATE_INFO};
    VkDeviceCreateInfo bad{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &group};
    rejects([&] { vulkan::merge_requirements(source, bad); });
  }
  {
    VkPhysicalDeviceTimelineSemaphoreFeatures b{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES};
    VkPhysicalDeviceTimelineSemaphoreFeatures a{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES, &b};
    VkDeviceCreateInfo bad{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &a};
    rejects([&] { vulkan::merge_requirements(source, bad); });
  }
  {
    const char *names[] = {nullptr};
    VkDeviceCreateInfo bad{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    bad.enabledExtensionCount = 1;
    bad.ppEnabledExtensionNames = names;
    rejects([&] { vulkan::merge_requirements(source, bad); });
  }
  {
    // A host without features gets exactly the library's declaration.
    VkDeviceCreateInfo empty{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    auto only = vulkan::merge_requirements(source, empty);
    ensure(only.get().enabledExtensionCount == 2 && only.get().pEnabledFeatures->robustBufferAccess);
    ensure(only.get().pNext && static_cast<const VkBaseInStructure *>(only.get().pNext)->sType ==
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES);
  }
}
