#ifndef OIDN_WEBGPU_METAL_HPP
#define OIDN_WEBGPU_METAL_HPP
#include "oidn_webgpu_gpu.hpp"
#include "oidn_webgpu_metal.h"
namespace oidnw::metal {
inline gpu::device import_device(const oidnw_gpu_metal_device_desc &desc,
    gpu::device_options options = {}) {
  gpu::require_version();
  if (!options.on_cleanup_failure) throw gpu::error(OIDNW_ERROR_INVALID_ARGUMENT, "cleanup handler is NULL");
  oidnw_gpu_device *value = nullptr;
  std::array<char, 1024> message{};
  gpu::require(oidnw_gpu_device_import_metal(&desc, &value, message.data(), message.size()), message.data());
  return gpu::device::adopt(value, std::move(options));
}
inline gpu::buffer import_buffer(const gpu::device &context, void *native, uint64_t bytes) {
  oidnw_gpu_buffer *value = nullptr;
  std::array<char, 1024> message{};
  gpu::require(oidnw_gpu_buffer_import_metal(context.get(), native, bytes, &value,
      message.data(), message.size()), message.data());
  return gpu::buffer(value, context.owner());
}
inline gpu::sync import_sync(const gpu::device &context, void *native) {
  oidnw_gpu_sync *value = nullptr;
  std::array<char, 1024> message{};
  gpu::require(oidnw_gpu_sync_import_metal(context.get(), native, &value,
      message.data(), message.size()), message.data());
  return gpu::sync(value, context.owner());
}
}
#endif
