#ifndef OIDN_WEBGPU_DX12_HPP
#define OIDN_WEBGPU_DX12_HPP
#include "oidn_webgpu_gpu.hpp"
#include "oidn_webgpu_dx12.h"
namespace oidnw::dx12 {
inline gpu::device create_device(const oidnw_gpu_dx12_device_desc &desc,
    gpu::device_options options = {}) {
  gpu::require_version();
  if (!options.on_cleanup_failure) throw gpu::error(OIDNW_ERROR_INVALID_ARGUMENT, "cleanup handler is NULL");
  oidnw_gpu_device *value = nullptr;
  std::array<char, 1024> message{};
  gpu::require(oidnw_gpu_device_create_dx12(&desc, &value, message.data(), message.size()), message.data());
  return gpu::device::adopt(value, std::move(options));
}
inline gpu::buffer import_buffer(const gpu::device &context, ID3D12Resource *native) {
  oidnw_gpu_buffer *value = nullptr;
  std::array<char, 1024> message{};
  gpu::require(oidnw_gpu_buffer_import_dx12(context.get(), native, &value,
      message.data(), message.size()), message.data());
  return gpu::buffer(value, context.owner());
}
inline gpu::sync import_sync(const gpu::device &context, ID3D12Fence *native) {
  oidnw_gpu_sync *value = nullptr;
  std::array<char, 1024> message{};
  gpu::require(oidnw_gpu_sync_import_dx12(context.get(), native, &value,
      message.data(), message.size()), message.data());
  return gpu::sync(value, context.owner());
}
}
#endif
