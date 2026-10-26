#include "oidn_webgpu_gpu.hpp"
#include <cstdlib>
#include <cstring>
#include <type_traits>

template<class T> static void ensure(const T &condition) {
  if (!condition) std::abort();
}
struct oidnw_gpu_device { unsigned children = 0; uint64_t issued = 0, done = 0, failed = 0; bool ready = true, lost = false; };
struct oidnw_gpu_model {};
struct oidnw_gpu_filter { oidnw_gpu_device *device; };
struct oidnw_gpu_prepared { oidnw_gpu_device *device; };
struct oidnw_gpu_buffer { oidnw_gpu_device *device; };
struct oidnw_gpu_sync { oidnw_gpu_device *device; };
static unsigned destroyed = 0, waits = 0, model_frees = 0;
static bool fail_next = false;
static oidnw::gpu::detached_device orphan;
static void retain_orphan(oidnw::gpu::detached_device value, const oidnw::gpu::status &status) noexcept {
  ensure(status.code == OIDNW_ERROR_DEVICE_LOST);
  orphan = std::move(value);
}
extern "C" {
uint32_t oidnw_api_version() { return OIDNW_API_VERSION; }
void oidnw_gpu_filter_desc_default(oidnw_gpu_filter_desc *p) { std::memset(p, 0, sizeof *p); p->hdr = 1; }
void oidnw_gpu_submit_desc_default(oidnw_gpu_submit_desc *p) { std::memset(p, 0, sizeof *p); p->input_scale = 1; }
int32_t oidnw_gpu_model_create(const void *bytes, size_t, oidnw_gpu_model **out, char *, size_t) {
  *out = nullptr;
  if (!bytes) return OIDNW_ERROR_MODEL;
  *out = new oidnw_gpu_model;
  return OIDNW_OK;
}
void oidnw_gpu_model_destroy(oidnw_gpu_model *p) { if (p) { ++model_frees; delete p; } }
int32_t oidnw_gpu_filter_create(oidnw_gpu_device *d, const oidnw_gpu_model *, const oidnw_gpu_filter_desc *, oidnw_gpu_filter **out, char *, size_t) {
  *out = new oidnw_gpu_filter{d}; ++d->children; return OIDNW_OK;
}
void oidnw_gpu_filter_destroy(oidnw_gpu_filter *p) { if (p) { --p->device->children; delete p; } }
int32_t oidnw_gpu_prepare(oidnw_gpu_filter *f, uint32_t, uint32_t, const oidnw_gpu_images *, oidnw_gpu_prepared **out, char *, size_t) {
  *out = new oidnw_gpu_prepared{f->device}; ++f->device->children; return OIDNW_OK;
}
int32_t oidnw_gpu_prepared_get_info(const oidnw_gpu_prepared *p, oidnw_gpu_prepared_info *out, char *, size_t) {
  if (!p || out->next) return OIDNW_ERROR_INVALID_ARGUMENT;
  *out = {}; out->precision = OIDNW_PRECISION_FP16; out->kernels = OIDNW_GPU_KERNEL_PATH_SUBGROUP_MATRIX;
  out->tile_count = 1; out->workspace_bytes = 4096; out->shared_weight_bytes = 8192; out->runtime_checks = 1;
  return OIDNW_OK;
}
void oidnw_gpu_prepared_destroy(oidnw_gpu_prepared *p) { if (p) { --p->device->children; delete p; } }
void oidnw_gpu_buffer_destroy(oidnw_gpu_buffer *p) { if (p) { --p->device->children; delete p; } }
void oidnw_gpu_sync_destroy(oidnw_gpu_sync *p) { if (p) { --p->device->children; delete p; } }
int32_t oidnw_gpu_submit(oidnw_gpu_prepared *p, const oidnw_gpu_submit_desc *, uint64_t *seq, char *message, size_t size) {
  *seq = 0;
  if (!p || p->device->lost) return OIDNW_ERROR_DEVICE_LOST;
  *seq = ++p->device->issued;
  if (fail_next) {
    fail_next = false; p->device->failed = *seq;
    std::snprintf(message, size, "%s", "encoding failed");
    return OIDNW_ERROR_GPU;
  }
  return OIDNW_OK;
}
int32_t oidnw_gpu_wait(oidnw_gpu_device *d, uint64_t seq, uint64_t, uint32_t *done, char *, size_t) {
  ++waits; *done = 0;
  if (!d || !seq || seq > d->issued) return OIDNW_ERROR_INVALID_ARGUMENT;
  if (d->lost) return OIDNW_ERROR_DEVICE_LOST;
  if (!d->ready) return OIDNW_ERROR_TIMEOUT;
  d->done = seq; *done = 1;
  return seq == d->failed ? OIDNW_ERROR_GPU : OIDNW_OK;
}
int32_t oidnw_gpu_poll(oidnw_gpu_device *d, uint64_t seq, uint32_t *done, char *error, size_t size) {
  return oidnw_gpu_wait(d, seq, 0, done, error, size);
}
int32_t oidnw_gpu_device_destroy(oidnw_gpu_device *d, char *, size_t) {
  if (!d) return OIDNW_OK;
  if (d->children || d->done != d->issued) return OIDNW_ERROR_BUSY;
  ++destroyed; delete d; return OIDNW_OK;
}
int32_t oidnw_gpu_device_destroy_quiescent(oidnw_gpu_device *d, char *error, size_t size) {
  if (!d->lost) return OIDNW_ERROR_INVALID_ARGUMENT;
  d->done = d->issued;
  return oidnw_gpu_device_destroy(d, error, size);
}
int32_t oidnw_gpu_get_limits(const oidnw_gpu_device *, oidnw_gpu_limits *out, char *, size_t) {
  *out = {256, 1024 * 1024, 1024 * 1024, 8}; return OIDNW_OK;
}
}

int main() {
  using namespace oidnw::gpu;
  static_assert(!std::is_copy_constructible_v<device>);
  static_assert(!std::is_copy_constructible_v<workspace>);
  static_assert(std::is_nothrow_move_constructible_v<workspace>);
  static_assert(std::is_nothrow_move_assignable_v<buffer>);
  oidnw_gpu_images images{};
  std::weak_ptr<int> host_lifetime;
  {
    auto host = std::make_shared<int>(7);
    host_lifetime = host;
    device_options options;
    options.host_lifetime = host;
    auto context = device::adopt(new oidnw_gpu_device, std::move(options));
    host.reset();
    auto weights = load_model("x", 1);
    auto source = make_filter(context, weights);
    auto prepared = prepare(source, 16, 16, images);
    const auto info = prepared.info();
    ensure(info.precision == OIDNW_PRECISION_FP16 && info.kernels == OIDNW_GPU_KERNEL_PATH_SUBGROUP_MATRIX);
    ensure(info.tile_count == 1 && info.workspace_bytes == 4096 && info.shared_weight_bytes == 8192 && info.runtime_checks);
    ++context.get()->children;
    buffer color(new oidnw_gpu_buffer{context.get()}, context.owner());
    buffer moved = std::move(color);
    ensure(!color && moved);
    auto sent = prepared.submit(submit_options());
    ensure(sent && sent.accepted() && sent.sequence == 1);
    for (uint64_t n = 2; n <= 1000; ++n) {
      auto next = prepared.submit(submit_options());
      ensure(next && next.sequence == n);
    }
    context.track(sent.sequence);
    ensure(context.close(0).code == OIDNW_ERROR_BUSY);
    ensure(context.limits().image_offset_alignment == 256);
    weights.reset(); source.reset();
    context = device{};
    ensure(destroyed == 0 && !host_lifetime.expired());
    prepared.reset();
    ensure(destroyed == 0);
    moved.reset();
    ensure(destroyed == 1 && waits == 1 && model_frees == 1);
    ensure(host_lifetime.expired());
  }
  {
    auto context = device::adopt(new oidnw_gpu_device);
    auto weights = load_model("x", 1);
    auto source = make_filter(context, weights);
    auto prepared = prepare(source, 16, 16, images);
    fail_next = true;
    auto sent = prepared.submit(submit_options());
    ensure(sent.code == OIDNW_ERROR_GPU && sent.accepted());
    ensure(std::strcmp(sent.message.data(), "encoding failed") == 0);
    auto result = context.wait(sent.sequence);
    ensure(result.code == OIDNW_ERROR_GPU && result.completed);
    source.reset(); prepared.reset();
    ensure(context.close(0) && !context);
  }
  {
    auto context = device::adopt(new oidnw_gpu_device);
    auto weights = load_model("x", 1);
    auto source = make_filter(context, weights);
    auto prepared = prepare(source, 16, 16, images);
    auto sent = prepared.submit(submit_options());
    context.get()->ready = false;
    ensure(context.wait(sent.sequence, 0).code == OIDNW_ERROR_TIMEOUT);
    source.reset(); prepared.reset();
    ensure(context.close(0).code == OIDNW_ERROR_TIMEOUT && context);
    auto raw = context.release();
    ensure(!context);
    raw.handle->ready = true;
    uint32_t done = 0;
    ensure(oidnw_gpu_wait(raw.handle, sent.sequence, 0, &done, nullptr, 0) == OIDNW_OK);
    ensure(oidnw_gpu_device_destroy(raw.handle, nullptr, 0) == OIDNW_OK);
  }
  {
    auto context = device::adopt(new oidnw_gpu_device);
    auto weights = load_model("x", 1);
    auto source = make_filter(context, weights);
    auto prepared = prepare(source, 16, 16, images);
    auto sent = prepared.submit(submit_options());
    context.get()->lost = true;
    auto result = context.wait(sent.sequence, 0);
    ensure(result.code == OIDNW_ERROR_DEVICE_LOST && !result.completed);
    source.reset(); prepared.reset();
    ensure(context.close_quiescent());
  }
  {
    auto host = std::make_shared<int>(9);
    host_lifetime = host;
    auto context = device::adopt(new oidnw_gpu_device, {retain_orphan, host});
    host.reset();
    auto weights = load_model("x", 1);
    auto source = make_filter(context, weights);
    auto prepared = prepare(source, 16, 16, images);
    prepared.submit(submit_options());
    context.get()->lost = true;
  }
  ensure(orphan.handle && !host_lifetime.expired());
  ensure(oidnw_gpu_device_destroy_quiescent(orphan.handle, nullptr, 0) == OIDNW_OK);
  orphan = {};
  ensure(host_lifetime.expired());
  {
    auto context = device::adopt(new oidnw_gpu_device);
    try { auto weights = load_model(nullptr, 0); ensure(false); }
    catch (const error &e) { ensure(e.code == OIDNW_ERROR_MODEL); }
    ensure(context.close());
  }
  ensure(destroyed == 6);
}
