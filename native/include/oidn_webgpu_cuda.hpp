#ifndef OIDN_WEBGPU_CUDA_HPP
#define OIDN_WEBGPU_CUDA_HPP
#include <cuda_runtime_api.h>
#include "oidn_webgpu_gpu.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace oidnw::cuda {
namespace detail { struct state; }
enum class code { success, invalid_argument, unsupported, busy, timeout, cuda_error, backend_error, closed };
struct status {
  code value = code::success;
  std::string message;
  explicit operator bool() const noexcept { return value == code::success; }
};
enum class format { rgba32f };
struct image {
  void *data = nullptr;
  uint64_t bytes = 0;
  uint64_t offset = 0;
  uint64_t row_stride = 0;
  uint32_t width = 0, height = 0;
  format pixel_format = format::rgba32f;
};
struct images { image color, albedo, normal, output; };
struct options {
  uint32_t width = 0, height = 0, max_tile_size = 0;
  uint32_t slots = 3;
  bool hdr = true;
  uint32_t transfer = OIDNW_TRANSFER_PU;
  bool fp32 = false, runtime_checks = true, debug = false, gpu_validation = false;
  std::string agility_sdk_path;
  uint32_t agility_sdk_version = 0;
};
struct statistics {
  uint64_t input_copy_bytes = 0, output_copy_bytes = 0;
  uint64_t copy_enqueues = 0, shared_enqueues = 0;
  uint64_t shared_allocation_bytes = 0;
};
struct runtime_info {
  std::string device_name, core_dll_path;
  uint32_t api_version = 0, agility_sdk_version = 0, msvc_version = 0;
  oidnw_gpu_prepared_info workspace{};
};
enum class task_phase { not_accepted, accepted, accepted_error, completed };
class task {
  std::shared_ptr<detail::state> state_;
  uint32_t slot_ = 0;
  uint64_t generation_ = 0;
  task(std::shared_ptr<detail::state>, uint32_t, uint64_t, status);
  friend class adapter;
public:
  status result;
  task() noexcept = default;
  task(const task &) = delete;
  task &operator=(const task &) = delete;
  task(task &&) noexcept;
  task &operator=(task &&) noexcept;
  ~task();
  bool accepted() const noexcept;
  uint64_t sequence() const noexcept;
  task_phase phase() const;
  status query(bool &completed) const;
  status wait(uint64_t timeout_ns = UINT64_MAX) const;
  status depend(cudaStream_t stream) const;
};
class shared_images {
  std::shared_ptr<detail::state> state_;
  uint32_t slot_ = 0;
  uint64_t generation_ = 0;
  shared_images(std::shared_ptr<detail::state>, uint32_t, uint64_t);
  friend class adapter;
public:
  status result;
  shared_images() noexcept = default;
  shared_images(const shared_images &) = delete;
  shared_images &operator=(const shared_images &) = delete;
  shared_images(shared_images &&) noexcept;
  shared_images &operator=(shared_images &&) noexcept;
  ~shared_images();
  explicit operator bool() const noexcept;
  images views() const;
};
class adapter {
  std::shared_ptr<detail::state> state_;
public:
  adapter(const options &, const void *weights, size_t weights_bytes);
  adapter(const adapter &) = delete;
  adapter &operator=(const adapter &) = delete;
  adapter(adapter &&) noexcept;
  adapter &operator=(adapter &&) noexcept;
  ~adapter();
  task enqueue_copy(const images &, cudaStream_t, float input_scale = 1);
  shared_images acquire();
  task enqueue_shared(shared_images &, cudaStream_t, float input_scale = 1);
  status release(shared_images &, cudaStream_t consumer_stream);
  /* Requires released task handles/leases and completed CUDA consumer retirement.
   * Reuses devices and weights. Failure preserves the previous image pool. */
  status resize(uint32_t width, uint32_t height);
  status drain(uint64_t timeout_ns = UINT64_MAX);
  status close(uint64_t timeout_ns = UINT64_MAX);
  statistics stats() const;
  runtime_info info() const;
  bool uses_fp16() const;
};
}
#endif
