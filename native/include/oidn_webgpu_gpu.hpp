#ifndef OIDN_WEBGPU_GPU_HPP
#define OIDN_WEBGPU_GPU_HPP
#include "oidn_webgpu_gpu.h"
#include <array>
#include <atomic>
#include <cstdio>
#include <exception>
#include <memory>
#include <stdexcept>
#include <utility>

namespace oidnw::gpu {
inline const char *status_name(int32_t code) noexcept {
  switch (code) {
    case OIDNW_OK: return "OK";
    case OIDNW_ERROR_INVALID_ARGUMENT: return "INVALID_ARGUMENT";
    case OIDNW_ERROR_UNAVAILABLE: return "UNAVAILABLE";
    case OIDNW_ERROR_MODEL: return "MODEL";
    case OIDNW_ERROR_GPU: return "GPU";
    case OIDNW_ERROR_BUSY: return "BUSY";
    case OIDNW_ERROR_DEVICE_LOST: return "DEVICE_LOST";
    case OIDNW_ERROR_TIMEOUT: return "TIMEOUT";
    case OIDNW_ERROR_QUEUE_FULL: return "QUEUE_FULL";
    default: return "UNKNOWN";
  }
}
struct device_info {
  std::array<char, 256> name{};
  uint32_t backend = OIDNW_BACKEND_AUTO;
  uint32_t features = 0;
};
struct status {
  int32_t code = OIDNW_OK;
  std::array<char, 1024> message{};
  explicit operator bool() const noexcept { return code == OIDNW_OK; }
};
struct submission : status {
  uint64_t sequence = 0;
  bool accepted() const noexcept { return sequence != 0; }
};
struct completion : status {
  bool completed = false;
};
class error : public std::runtime_error {
public:
  int32_t code;
  error(int32_t value, const char *message) : std::runtime_error(message), code(value) {}
};
inline void require(int32_t code, const char *message) {
  if (code != OIDNW_OK) throw error(code, message);
}
inline void require_version() {
  if (!OIDNW_API_VERSION_IS_COMPATIBLE(oidnw_api_version()))
    throw error(OIDNW_ERROR_INVALID_ARGUMENT, "header/library version mismatch");
}

struct detached_device {
  oidnw_gpu_device *handle = nullptr;
  std::shared_ptr<void> host_lifetime;
};
/* Cleanup failure transfers the native context and host-lifetime token to the
 * handler. Both remain owned until safe native destruction. The default
 * terminates; handle recovery explicitly or install a retaining noexcept handler. */
using cleanup_failure_handler = void (*)(detached_device, const status &) noexcept;
namespace detail {
inline void terminate_cleanup(detached_device, const status &) noexcept { std::terminate(); }
}
struct device_options {
  cleanup_failure_handler on_cleanup_failure = detail::terminate_cleanup;
  std::shared_ptr<void> host_lifetime;
};
namespace detail {
inline status failure(int32_t code, const char *message) noexcept {
  status result;
  result.code = code;
  std::snprintf(result.message.data(), result.message.size(), "%s", message);
  return result;
}
struct device_state {
  oidnw_gpu_device *handle = nullptr;
  std::atomic<uint64_t> latest{0};
  cleanup_failure_handler on_failure = terminate_cleanup;
  std::shared_ptr<void> host_lifetime;
  void record(uint64_t sequence) noexcept {
    auto previous = latest.load(std::memory_order_relaxed);
    while (previous < sequence && !latest.compare_exchange_weak(previous, sequence,
        std::memory_order_release, std::memory_order_relaxed)) {}
  }
  completion wait(uint64_t sequence, uint64_t timeout) const noexcept {
    completion result;
    uint32_t done = 0;
    result.code = oidnw_gpu_wait(handle, sequence, timeout, &done,
        result.message.data(), result.message.size());
    result.completed = done != 0;
    return result;
  }
  status close(uint64_t timeout) noexcept {
    if (!handle) return {};
    const auto sequence = latest.load(std::memory_order_acquire);
    if (sequence) {
      const auto result = wait(sequence, timeout);
      if (!result.completed) return result;
    }
    status result;
    result.code = oidnw_gpu_device_destroy(handle, result.message.data(), result.message.size());
    if (result) { handle = nullptr; host_lifetime.reset(); }
    return result;
  }
  ~device_state() noexcept {
    if (!handle) return;
    const auto result = close(UINT64_MAX);
    if (!result) on_failure({std::exchange(handle, nullptr), std::move(host_lifetime)}, result);
  }
};

template<class T, void (*Destroy)(T *)> class owned {
protected:
  T *handle_ = nullptr;
  std::shared_ptr<device_state> owner_;
public:
  owned() noexcept = default;
  explicit owned(T *value, std::shared_ptr<device_state> owner) noexcept
      : handle_(value), owner_(std::move(owner)) {}
  owned(const owned &) = delete;
  owned &operator=(const owned &) = delete;
  owned(owned &&other) noexcept
      : handle_(std::exchange(other.handle_, nullptr)), owner_(std::move(other.owner_)) {}
  owned &operator=(owned &&other) noexcept {
    if (this != &other) {
      reset();
      handle_ = std::exchange(other.handle_, nullptr);
      owner_ = std::move(other.owner_);
    }
    return *this;
  }
  ~owned() noexcept { reset(); }
  T *get() const noexcept { return handle_; }
  explicit operator bool() const noexcept { return handle_ != nullptr; }
  const std::shared_ptr<device_state> &owner() const noexcept { return owner_; }
  void reset() noexcept {
    if (auto value = std::exchange(handle_, nullptr)) Destroy(value);
    owner_.reset();
  }
};
}
class model : public detail::owned<oidnw_gpu_model, oidnw_gpu_model_destroy> {
  using base = detail::owned<oidnw_gpu_model, oidnw_gpu_model_destroy>;
public:
  using base::base;
  uint32_t input_count() const noexcept { return oidnw_gpu_model_input_count(get()); }
};
using filter = detail::owned<oidnw_gpu_filter, oidnw_gpu_filter_destroy>;
using buffer = detail::owned<oidnw_gpu_buffer, oidnw_gpu_buffer_destroy>;
using sync = detail::owned<oidnw_gpu_sync, oidnw_gpu_sync_destroy>;

/* Move-only device owner. Child wrappers retain its state. Last-owner cleanup
 * drains accepted C++ submissions and may block; producers must signal without
 * depending on this destructor. Borrowed native objects outlive that cleanup.
 * close() is the explicit, fallible shutdown path. */
class device {
  std::shared_ptr<detail::device_state> state_;
public:
  device() noexcept = default;
  device(const device &) = delete;
  device &operator=(const device &) = delete;
  device(device &&) noexcept = default;
  device &operator=(device &&) noexcept = default;
  /* Takes ownership of a fresh context with no children or prior submissions. */
  static device adopt(oidnw_gpu_device *handle, device_options options = {}) {
    if (!handle || !options.on_cleanup_failure) throw error(OIDNW_ERROR_INVALID_ARGUMENT, "device/cleanup handler is NULL");
    device result;
    try {
      result.state_ = std::make_shared<detail::device_state>();
    } catch (...) {
      status cleanup;
      cleanup.code = oidnw_gpu_device_destroy(handle, cleanup.message.data(), cleanup.message.size());
      if (!cleanup) options.on_cleanup_failure({handle, std::move(options.host_lifetime)}, cleanup);
      throw;
    }
    result.state_->handle = handle;
    result.state_->on_failure = options.on_cleanup_failure;
    result.state_->host_lifetime = std::move(options.host_lifetime);
    return result;
  }
  oidnw_gpu_device *get() const noexcept { return state_ ? state_->handle : nullptr; }
  explicit operator bool() const noexcept { return get() != nullptr; }
  const std::shared_ptr<detail::device_state> &owner() const noexcept { return state_; }
  completion poll(uint64_t sequence) const noexcept {
    completion result;
    uint32_t done = 0;
    result.code = oidnw_gpu_poll(get(), sequence, &done, result.message.data(), result.message.size());
    result.completed = done != 0;
    return result;
  }
  completion wait(uint64_t sequence, uint64_t timeout = UINT64_MAX) const noexcept {
    if (!state_) {
      completion result;
      static_cast<status &>(result) = detail::failure(OIDNW_ERROR_INVALID_ARGUMENT, "device is empty");
      return result;
    }
    return state_->wait(sequence, timeout);
  }
  device_info info() const {
    device_info result;
    require(oidnw_gpu_device_info(get(), result.name.data(), result.name.size(),
        &result.backend, &result.features), "device info failed");
    return result;
  }
  oidnw_gpu_limits limits() const {
    oidnw_gpu_limits result{};
    std::array<char, 1024> message{};
    require(oidnw_gpu_get_limits(get(), &result, message.data(), message.size()), message.data());
    return result;
  }
  status close(uint64_t timeout = UINT64_MAX) noexcept {
    if (!state_ || !get()) return {};
    if (state_.use_count() != 1)
      return detail::failure(OIDNW_ERROR_BUSY, "release child wrappers before closing the device");
    return state_->close(timeout);
  }
  /* Host has established that no GPU work can access these allocations. */
  status close_quiescent() noexcept {
    if (!state_ || !get()) return {};
    if (state_.use_count() != 1)
      return detail::failure(OIDNW_ERROR_BUSY, "release child wrappers before closing the device");
    status result;
    result.code = oidnw_gpu_device_destroy_quiescent(get(), result.message.data(), result.message.size());
    if (result) { state_->handle = nullptr; state_->host_lifetime.reset(); }
    return result;
  }
  /* Transfers both context and host token for deferred native cleanup. */
  [[nodiscard]] detached_device release() {
    if (!state_) return {};
    if (state_.use_count() != 1)
      throw error(OIDNW_ERROR_BUSY, "release child wrappers before transferring the device");
    return {std::exchange(state_->handle, nullptr), std::move(state_->host_lifetime)};
  }
  void track(uint64_t accepted_sequence) const noexcept {
    if (state_) state_->record(accepted_sequence);
  }
};

class workspace : public detail::owned<oidnw_gpu_prepared, oidnw_gpu_prepared_destroy> {
  using base = detail::owned<oidnw_gpu_prepared, oidnw_gpu_prepared_destroy>;
public:
  using base::base;
  oidnw_gpu_prepared_info info() const {
    oidnw_gpu_prepared_info result{};
    std::array<char, 1024> message{};
    require(oidnw_gpu_prepared_get_info(get(), &result, message.data(), message.size()), message.data());
    return result;
  }
  submission submit(const oidnw_gpu_submit_desc &desc) const noexcept {
    submission result;
    result.code = oidnw_gpu_submit(get(), &desc, &result.sequence,
        result.message.data(), result.message.size());
    if (result.accepted() && owner_) owner_->record(result.sequence);
    return result;
  }
  submission submit(const oidnw_gpu_images &images,
      const oidnw_gpu_timeline_point &wait, const oidnw_gpu_timeline_point &signal,
      float input_scale = 1.0f) const noexcept {
    oidnw_gpu_submit_desc desc;
    oidnw_gpu_submit_desc_default(&desc);
    desc.images = images;
    desc.input_scale = input_scale;
    desc.waits = &wait;
    desc.wait_count = 1;
    desc.signals = &signal;
    desc.signal_count = 1;
    return submit(desc);
  }
};
inline oidnw_gpu_filter_desc filter_options() noexcept {
  oidnw_gpu_filter_desc result;
  oidnw_gpu_filter_desc_default(&result);
  return result;
}
inline oidnw_gpu_submit_desc submit_options() noexcept {
  oidnw_gpu_submit_desc result;
  oidnw_gpu_submit_desc_default(&result);
  return result;
}
inline model load_model(const void *weights, size_t bytes) {
  require_version();
  oidnw_gpu_model *value = nullptr;
  std::array<char, 1024> message{};
  require(oidnw_gpu_model_create(weights, bytes, &value, message.data(), message.size()), message.data());
  return model(value, {});
}
inline filter make_filter(const device &context, const model &weights,
    const oidnw_gpu_filter_desc &options = filter_options()) {
  oidnw_gpu_filter *value = nullptr;
  std::array<char, 1024> message{};
  require(oidnw_gpu_filter_create(context.get(), weights.get(), &options, &value,
      message.data(), message.size()), message.data());
  return filter(value, context.owner());
}
inline void warm_up(const filter &source, uint32_t width, uint32_t height) {
  std::array<char, 1024> message{};
  require(oidnw_gpu_filter_warm_up(source.get(), width, height, message.data(), message.size()), message.data());
}
inline workspace prepare(const filter &source, uint32_t width, uint32_t height,
    const oidnw_gpu_images &images) {
  oidnw_gpu_prepared *value = nullptr;
  std::array<char, 1024> message{};
  require(oidnw_gpu_prepare(source.get(), width, height, &images, &value,
      message.data(), message.size()), message.data());
  return workspace(value, source.owner());
}
}
#endif
