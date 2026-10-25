#ifndef OIDN_WEBGPU_VULKAN_HPP
#define OIDN_WEBGPU_VULKAN_HPP
#include "oidn_webgpu_gpu.hpp"
#include "oidn_webgpu_vulkan.h"
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
}
#endif
