#ifndef OIDN_WEBGPU_VULKAN_HPP
#define OIDN_WEBGPU_VULKAN_HPP
#include "oidn_webgpu_gpu.hpp"
#include "oidn_webgpu_vulkan.h"
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>
namespace oidnw::vulkan {
inline gpu::device import_device(const oidnw_gpu_vulkan_device_desc &desc,
    gpu::device_options options = {}) {
  gpu::require_version();
  if (!options.on_cleanup_failure) throw gpu::error(OIDNW_ERROR_INVALID_ARGUMENT, "cleanup handler is NULL");
  oidnw_gpu_device *value = nullptr;
  std::array<char, 1024> message{};
  gpu::require(oidnw_gpu_device_import_vulkan(&desc, &value, message.data(), message.size()), message.data());
  return gpu::device::adopt(value, std::move(options));
}
inline gpu::buffer import_buffer(const gpu::device &context, VkBuffer native, const VkBufferCreateInfo &info) {
  oidnw_gpu_buffer *value = nullptr;
  std::array<char, 1024> message{};
  gpu::require(oidnw_gpu_buffer_import_vulkan(context.get(), native, &info, &value,
      message.data(), message.size()), message.data());
  return gpu::buffer(value, context.owner());
}
inline gpu::sync import_sync(const gpu::device &context, VkSemaphore native, const VkSemaphoreCreateInfo &info) {
  oidnw_gpu_sync *value = nullptr;
  std::array<char, 1024> message{};
  gpu::require(oidnw_gpu_sync_import_vulkan(context.get(), native, &info, &value,
      message.data(), message.size()), message.data());
  return gpu::sync(value, context.owner());
}
using requirements = gpu::detail::owned<oidnw_gpu_vulkan_requirements,
    oidnw_gpu_vulkan_requirements_destroy>;
inline std::vector<const char *> instance_extensions(
    PFN_vkGetInstanceProcAddr loader, uint32_t api_version) {
  gpu::require_version();
  std::array<char, 1024> message{};
  size_t count = 0;
  gpu::require(oidnw_gpu_vulkan_instance_extensions(loader, api_version, nullptr, &count,
      message.data(), message.size()), message.data());
  std::vector<const char *> names(count);
  gpu::require(oidnw_gpu_vulkan_instance_extensions(loader, api_version, names.data(), &count,
      message.data(), message.size()), message.data());
  return names;
}
inline requirements query_requirements(const oidnw_gpu_vulkan_requirements_desc &desc) {
  gpu::require_version();
  oidnw_gpu_vulkan_requirements *value = nullptr;
  std::array<char, 1024> message{};
  gpu::require(oidnw_gpu_vulkan_requirements_create(&desc, &value,
      message.data(), message.size()), message.data());
  return requirements(value, {});
}
inline VkDeviceCreateInfo device_create_info(const requirements &source,
    uint32_t *enabled_features = nullptr) {
  VkDeviceCreateInfo result;
  std::array<char, 1024> message{};
  gpu::require(oidnw_gpu_vulkan_requirements_get(source.get(), &result, enabled_features,
      message.data(), message.size()), message.data());
  return result;
}

/* Mutex shared by the host and the library for every access to the imported
 * queue. Lock it (for example with std::lock_guard) around host vkQueueSubmit
 * and vkQueueWaitIdle calls; never call the library while holding it. */
class queue_lock {
public:
  queue_lock() = default;
  queue_lock(const queue_lock &) = delete;
  queue_lock &operator=(const queue_lock &) = delete;
  void lock() { mutex_.lock(); }
  bool try_lock() { return mutex_.try_lock(); }
  void unlock() { mutex_.unlock(); }
  static void lock_callback(void *self) noexcept { static_cast<queue_lock *>(self)->mutex_.lock(); }
  static void unlock_callback(void *self) noexcept { static_cast<queue_lock *>(self)->mutex_.unlock(); }
private:
  std::mutex mutex_;
};

struct device_handles {
  PFN_vkGetInstanceProcAddr get_instance_proc_addr;
  VkInstance instance;
  VkPhysicalDevice physical_device;
  VkDevice device;
  VkQueue queue;
  uint32_t queue_family_index;
  uint32_t queue_index;
};

/* Imports a host device created with instance_info and device_info. The
 * declarations must be the ones passed to vkCreateInstance/vkCreateDevice and
 * stay valid during this call. The returned context retains lock until its
 * native destruction; options.host_lifetime is retained alongside it. */
inline gpu::device import_device(const device_handles &host,
    const VkInstanceCreateInfo &instance_info, const VkDeviceCreateInfo &device_info,
    std::shared_ptr<queue_lock> lock, uint32_t flags = 0, gpu::device_options options = {}) {
  if (!lock) throw gpu::error(OIDNW_ERROR_INVALID_ARGUMENT, "queue lock is NULL");
  oidnw_gpu_vulkan_device_desc desc;
  oidnw_gpu_vulkan_device_desc_default(&desc);
  desc.flags = flags;
  desc.get_instance_proc_addr = host.get_instance_proc_addr;
  desc.instance = host.instance;
  desc.physical_device = host.physical_device;
  desc.device = host.device;
  desc.queue = host.queue;
  desc.queue_family_index = host.queue_family_index;
  desc.queue_index = host.queue_index;
  desc.instance_create_info = &instance_info;
  desc.device_create_info = &device_info;
  desc.lock = &queue_lock::lock_callback;
  desc.unlock = &queue_lock::unlock_callback;
  desc.user_data = lock.get();
  if (options.host_lifetime) {
    using both = std::pair<std::shared_ptr<void>, std::shared_ptr<queue_lock>>;
    options.host_lifetime = std::make_shared<both>(std::move(options.host_lifetime), std::move(lock));
  } else {
    options.host_lifetime = std::move(lock);
  }
  return import_device(desc, std::move(options));
}

namespace detail {
/* Individual feature structures accepted by import (native/src/gpu_capi/
 * vulkan_features.rs). Each is a header followed only by VkBool32 fields.
 * Returns 0 for any other structure. */
inline size_t feature_size(VkStructureType type) noexcept {
  switch (type) {
#define OIDNW_VULKAN_FEATURE(T, S) case S: return sizeof(T);
  OIDNW_VULKAN_FEATURE(VkPhysicalDevice16BitStorageFeatures, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES)
  OIDNW_VULKAN_FEATURE(VkPhysicalDeviceBufferDeviceAddressFeatures, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES)
  OIDNW_VULKAN_FEATURE(VkPhysicalDeviceDescriptorIndexingFeatures, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES)
  OIDNW_VULKAN_FEATURE(VkPhysicalDeviceMultiviewFeatures, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_FEATURES)
  OIDNW_VULKAN_FEATURE(VkPhysicalDeviceSamplerYcbcrConversionFeatures, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLER_YCBCR_CONVERSION_FEATURES)
  OIDNW_VULKAN_FEATURE(VkPhysicalDeviceShaderAtomicInt64Features, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_INT64_FEATURES)
  OIDNW_VULKAN_FEATURE(VkPhysicalDeviceShaderDrawParametersFeatures, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DRAW_PARAMETERS_FEATURES)
  OIDNW_VULKAN_FEATURE(VkPhysicalDeviceShaderFloat16Int8Features, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES)
  OIDNW_VULKAN_FEATURE(VkPhysicalDeviceTimelineSemaphoreFeatures, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES)
  OIDNW_VULKAN_FEATURE(VkPhysicalDeviceVulkanMemoryModelFeatures, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES)
#ifdef VK_VERSION_1_3
  OIDNW_VULKAN_FEATURE(VkPhysicalDeviceImageRobustnessFeatures, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_ROBUSTNESS_FEATURES)
  OIDNW_VULKAN_FEATURE(VkPhysicalDeviceMaintenance4Features, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES)
  OIDNW_VULKAN_FEATURE(VkPhysicalDeviceShaderIntegerDotProductFeatures, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES)
  OIDNW_VULKAN_FEATURE(VkPhysicalDeviceSubgroupSizeControlFeatures, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES)
  OIDNW_VULKAN_FEATURE(VkPhysicalDeviceTextureCompressionASTCHDRFeatures, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TEXTURE_COMPRESSION_ASTC_HDR_FEATURES)
  OIDNW_VULKAN_FEATURE(VkPhysicalDeviceZeroInitializeWorkgroupMemoryFeatures, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ZERO_INITIALIZE_WORKGROUP_MEMORY_FEATURES)
#endif
#ifdef VK_KHR_acceleration_structure
  OIDNW_VULKAN_FEATURE(VkPhysicalDeviceAccelerationStructureFeaturesKHR, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR)
#endif
#ifdef VK_KHR_cooperative_matrix
  OIDNW_VULKAN_FEATURE(VkPhysicalDeviceCooperativeMatrixFeaturesKHR, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR)
#endif
#ifdef VK_KHR_fragment_shader_barycentric
  OIDNW_VULKAN_FEATURE(VkPhysicalDeviceFragmentShaderBarycentricFeaturesKHR, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_KHR)
#endif
#ifdef VK_EXT_mesh_shader
  OIDNW_VULKAN_FEATURE(VkPhysicalDeviceMeshShaderFeaturesEXT, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT)
#endif
#ifdef VK_KHR_portability_subset
  OIDNW_VULKAN_FEATURE(VkPhysicalDevicePortabilitySubsetFeaturesKHR, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PORTABILITY_SUBSET_FEATURES_KHR)
#endif
#ifdef VK_KHR_ray_query
  OIDNW_VULKAN_FEATURE(VkPhysicalDeviceRayQueryFeaturesKHR, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR)
#endif
#ifdef VK_KHR_ray_tracing_pipeline
  OIDNW_VULKAN_FEATURE(VkPhysicalDeviceRayTracingPipelineFeaturesKHR, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR)
#endif
#ifdef VK_KHR_ray_tracing_position_fetch
  OIDNW_VULKAN_FEATURE(VkPhysicalDeviceRayTracingPositionFetchFeaturesKHR, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_POSITION_FETCH_FEATURES_KHR)
#endif
#ifdef VK_EXT_robustness2
  OIDNW_VULKAN_FEATURE(VkPhysicalDeviceRobustness2FeaturesEXT, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT)
#endif
#ifdef VK_EXT_shader_atomic_float
  OIDNW_VULKAN_FEATURE(VkPhysicalDeviceShaderAtomicFloatFeaturesEXT, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT)
#endif
#ifdef VK_EXT_shader_image_atomic_int64
  OIDNW_VULKAN_FEATURE(VkPhysicalDeviceShaderImageAtomicInt64FeaturesEXT, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_IMAGE_ATOMIC_INT64_FEATURES_EXT)
#endif
#undef OIDNW_VULKAN_FEATURE
  default: return 0;
  }
}
inline bool aggregate_features(VkStructureType type) noexcept {
  return type == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 ||
      type == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES ||
      type == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES
#ifdef VK_VERSION_1_3
      || type == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES
#endif
#ifdef VK_VERSION_1_4
      || type == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES
#endif
      ;
}
inline void merge_bools(void *target, const void *source, size_t begin, size_t end) noexcept {
  for (size_t offset = begin; offset + sizeof(VkBool32) <= end; offset += sizeof(VkBool32)) {
    VkBool32 a, b;
    std::memcpy(&a, static_cast<char *>(target) + offset, sizeof a);
    std::memcpy(&b, static_cast<const char *>(source) + offset, sizeof b);
    a = (a || b) ? VK_TRUE : VK_FALSE;
    std::memcpy(static_cast<char *>(target) + offset, &a, sizeof a);
  }
}
}

/* A VkDeviceCreateInfo that enables both the host's and the library's
 * extensions and features. It owns copies of every extension name and feature
 * structure, and borrows the host's queue declarations and layer names. Pass
 * get() to vkCreateDevice and to import_device; moving keeps get() valid. */
class merged_device_info {
public:
  merged_device_info(const merged_device_info &) = delete;
  merged_device_info &operator=(const merged_device_info &) = delete;
  merged_device_info(merged_device_info &&) noexcept = default;
  merged_device_info &operator=(merged_device_info &&) noexcept = default;
  const VkDeviceCreateInfo &get() const noexcept { return info_; }
private:
  merged_device_info() = default;
  friend merged_device_info merge_requirements(const requirements &, const VkDeviceCreateInfo &);
  VkDeviceCreateInfo info_{};
  std::vector<std::string> extension_names_;
  std::vector<const char *> extensions_;
  std::unique_ptr<VkPhysicalDeviceFeatures> core_;
  std::vector<std::unique_ptr<uint64_t[]>> chain_;
};

/* Merges the library's requirements into the host's device declaration.
 * Extensions are deduplicated, core features and same-type feature structures
 * are OR-ed, and library-only feature structures are appended. A host
 * VkPhysicalDeviceFeatures2 is converted to pEnabledFeatures. Throws
 * INVALID_ARGUMENT for structures import rejects: aggregate
 * VkPhysicalDeviceVulkan1xFeatures, other pNext structures, duplicates, and
 * chains longer than 64 nodes. requirements may be destroyed after merging. */
inline merged_device_info merge_requirements(const requirements &library,
    const VkDeviceCreateInfo &host) {
  const VkDeviceCreateInfo required = device_create_info(library);
  merged_device_info result;
  result.info_ = host;

  auto add_extension = [&](const char *name) {
    if (!name) throw gpu::error(OIDNW_ERROR_INVALID_ARGUMENT, "device extension name is NULL");
    for (const auto &existing : result.extension_names_)
      if (existing == name) return;
    result.extension_names_.emplace_back(name);
  };
  for (uint32_t i = 0; i < host.enabledExtensionCount; ++i) add_extension(host.ppEnabledExtensionNames[i]);
  for (uint32_t i = 0; i < required.enabledExtensionCount; ++i) add_extension(required.ppEnabledExtensionNames[i]);
  for (const auto &name : result.extension_names_) result.extensions_.push_back(name.c_str());
  result.info_.enabledExtensionCount = uint32_t(result.extensions_.size());
  result.info_.ppEnabledExtensionNames = result.extensions_.empty() ? nullptr : result.extensions_.data();

  auto merge_core = [&](const VkPhysicalDeviceFeatures *features) {
    if (!features) return;
    if (!result.core_) result.core_ = std::make_unique<VkPhysicalDeviceFeatures>();
    detail::merge_bools(result.core_.get(), features, 0, sizeof(VkPhysicalDeviceFeatures));
  };
  merge_core(host.pEnabledFeatures);
  merge_core(required.pEnabledFeatures);

  std::vector<VkStructureType> types;
  bool host_features2 = false;
  auto merge_chain = [&](const void *head, bool from_host) {
    for (auto node = static_cast<const VkBaseInStructure *>(head); node; node = node->pNext) {
      if (from_host && node->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2) {
        // Equivalent to pEnabledFeatures, which is the only form import accepts.
        if (host.pEnabledFeatures || host_features2)
          throw gpu::error(OIDNW_ERROR_INVALID_ARGUMENT,
              "host declares core features twice (pEnabledFeatures and VkPhysicalDeviceFeatures2)");
        host_features2 = true;
        merge_core(&reinterpret_cast<const VkPhysicalDeviceFeatures2 *>(node)->features);
        continue;
      }
      if (detail::aggregate_features(node->sType))
        throw gpu::error(OIDNW_ERROR_INVALID_ARGUMENT,
            "aggregate Vulkan feature structures are unsupported; chain individual feature structures");
      const size_t size = detail::feature_size(node->sType);
      if (!size)
        throw gpu::error(OIDNW_ERROR_INVALID_ARGUMENT,
            "unsupported VkDeviceCreateInfo pNext structure; import accepts individual feature structures only");
      size_t index = 0;
      while (index < types.size() && types[index] != node->sType) ++index;
      if (index < types.size()) {
        if (from_host) throw gpu::error(OIDNW_ERROR_INVALID_ARGUMENT, "duplicate feature structure in host chain");
        detail::merge_bools(result.chain_[index].get(), node, sizeof(VkBaseInStructure), size);
        continue;
      }
      if (types.size() == 64) throw gpu::error(OIDNW_ERROR_INVALID_ARGUMENT, "feature chain too long");
      std::unique_ptr<uint64_t[]> copy(new uint64_t[(size + 7) / 8]());
      std::memcpy(copy.get(), node, size);
      result.chain_.push_back(std::move(copy));
      types.push_back(node->sType);
    }
  };
  merge_chain(host.pNext, true);
  merge_chain(required.pNext, false);
  result.info_.pEnabledFeatures = result.core_.get();
  void *next = nullptr;
  for (size_t i = result.chain_.size(); i-- > 0;) {
    reinterpret_cast<VkBaseOutStructure *>(result.chain_[i].get())->pNext =
        static_cast<VkBaseOutStructure *>(next);
    next = result.chain_[i].get();
  }
  result.info_.pNext = next;
  return result;
}
}
#endif
