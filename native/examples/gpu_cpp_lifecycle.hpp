#ifndef OIDNW_EXAMPLE_GPU_CPP_LIFECYCLE_HPP
#define OIDNW_EXAMPLE_GPU_CPP_LIFECYCLE_HPP
#include "oidn_webgpu_gpu.hpp"
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>

namespace cpp_lifecycle {
inline void ensure(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}
inline void check(const oidnw::gpu::status &result) {
  oidnw::gpu::require(result.code, result.message.data());
}
inline oidnw::gpu::detached_device deferred;
inline int32_t cleanup_code = OIDNW_OK;
inline void retain_cleanup(oidnw::gpu::detached_device device,
                           const oidnw::gpu::status &result) noexcept {
  deferred = std::move(device);
  cleanup_code = result.code;
}

template<class Factory>
void run(Factory factory, const void *weights, size_t weight_bytes,
         uint32_t size, uint32_t tile, bool checks) {
  using namespace oidnw::gpu;
  char message[1024]{};
  oidnw_gpu_model *invalid = nullptr;
  auto expected = oidnw_gpu_model_create(nullptr, 0, &invalid, message, sizeof message);
  ensure(expected != OIDNW_OK && !invalid, "invalid model accepted");
  try { auto model = load_model(nullptr, 0); ensure(false, "model exception missing"); }
  catch (const error &e) {
    ensure(e.code == expected && !std::strcmp(e.what(), message), "model exception changed");
  }
  {
    auto host = factory();
    std::weak_ptr<void> lifetime = host;
    auto context = host->context({retain_cleanup, host});
    auto image = host->image(context, 0);
    auto ready = host->sync(context, false);
    auto raw = context.get();
    auto moved = std::move(context);
    ensure(!context && moved.get() == raw, "device move changed context");
    ensure(moved.close(0).code == OIDNW_ERROR_BUSY, "children did not retain context");
    moved = oidnw::gpu::device{};
    host.reset();
    ensure(!lifetime.expired(), "host released before children");
    oidnw_gpu_limits limits{};
    require(oidnw_gpu_get_limits(raw, &limits, message, sizeof message), message);
    image.reset();
    ensure(!lifetime.expired(), "sync did not retain host");
    ready.reset();
    ensure(lifetime.expired() && !deferred.handle, "last child cleanup failed");
  }
  {
    auto host = factory();
    std::weak_ptr<void> lifetime = host;
    auto context = host->context({retain_cleanup, host});
    std::array<oidnw::gpu::buffer, 4> images;
    for (unsigned i = 0; i < 4; ++i) images[i] = host->image(context, i);
    auto ready = host->sync(context, false), done = host->sync(context, true);
    oidnw_gpu_images bindings{{images[0].get(), 256}, {images[1].get(), 256},
                             {images[2].get(), 256}, {images[3].get(), 256}};
    auto model = load_model(weights, weight_bytes);
    auto options = filter_options();
    options.max_tile_size = tile;
    options.flags = checks ? OIDNW_FILTER_RUNTIME_CHECKS : 0;
    auto filter = make_filter(context, model, options);
    auto prepared = prepare(filter, size, size, bindings);
    model.reset(); filter.reset();
    host->start();
    std::mutex mutex;
    std::condition_variable changed;
    bool release = false;
    std::thread producer([&] {
      std::unique_lock<std::mutex> lock(mutex);
      changed.wait_for(lock, std::chrono::seconds(5), [&] { return release; });
      host->unblock();
    });
    struct Join {
      std::thread &thread;
      std::mutex &mutex;
      std::condition_variable &changed;
      bool &release;
      void finish() {
        { std::lock_guard<std::mutex> lock(mutex); release = true; }
        changed.notify_one();
        if (thread.joinable()) thread.join();
      }
      ~Join() { finish(); }
    } join{producer, mutex, changed, release};
    auto sent = prepared.submit(bindings, {ready.get(), 1}, {done.get(), 1}, 0.5f);
    check(sent);
    ensure(sent.accepted(), "C++ submit lost sequence");
    auto pending = context.wait(sent.sequence, 0);
    ensure(pending.code == OIDNW_ERROR_TIMEOUT && !pending.completed,
           "C++ gated wait did not time out");
    prepared.reset();
    for (auto &image : images) image.reset();
    ready.reset(); done.reset();
    auto raw = context.get();
    auto closed = context.close(0);
    ensure(closed.code == OIDNW_ERROR_TIMEOUT && context.get() == raw,
           "timed-out close lost ownership");
    join.finish();
    auto completed = context.wait(sent.sequence, UINT64_C(60000000000));
    check(completed);
    ensure(completed.completed, "C++ wait missed completion");
    host->verify();
    host.reset();
    ensure(!lifetime.expired(), "close lost host token");
    check(context.close(0));
    ensure(!context && lifetime.expired(), "retry close retained context");
  }
  {
    auto host = factory();
    auto context = host->context({retain_cleanup, host});
    ensure(context.close_quiescent().code == OIDNW_ERROR_INVALID_ARGUMENT && context,
           "healthy context accepted quiescent recovery");
    auto detached = context.release();
    ensure(!context && detached.handle && detached.host_lifetime, "release lost ownership");
    require(oidnw_gpu_device_destroy(detached.handle, message, sizeof message), message);
    detached = {};
  }
  {
    auto host = factory();
    std::weak_ptr<void> lifetime = host;
    oidnw_gpu_filter *raw_filter = nullptr;
    {
      auto context = host->context({retain_cleanup, host});
      auto model = load_model(weights, weight_bytes);
      auto options = filter_options();
      require(oidnw_gpu_filter_create(context.get(), model.get(), &options, &raw_filter,
                                     message, sizeof message), message);
      host.reset();
    }
    ensure(deferred.handle && cleanup_code == OIDNW_ERROR_BUSY && !lifetime.expired(),
           "cleanup callback lost busy context/host");
    oidnw_gpu_filter_destroy(raw_filter);
    require(oidnw_gpu_device_destroy(deferred.handle, message, sizeof message), message);
    deferred = {};
    ensure(lifetime.expired(), "deferred cleanup retained host");
  }
}
}
#endif
