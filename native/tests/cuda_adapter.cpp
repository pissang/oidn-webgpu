#include "oidn_webgpu_cuda.hpp"
#include "oidn_webgpu_gpu.h"
#include <cuda.h>
#include <array>
#include <atomic>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>
namespace bridge = oidnw::cuda;
static unsigned fault = 0;
static std::atomic<bool> entered{false}, resume{false};
extern "C" CUresult CUDAAPI oidnw_cuda_test_luid(char *luid, unsigned *node, CUdevice device) {
  auto status = cuDeviceGetLuid(luid, node, device);
  if (fault == 5 && status == CUDA_SUCCESS) for (unsigned i = 0; i < 8; ++i) luid[i] = char(0xff);
  return status;
}
extern "C" int32_t oidnw_cuda_test_poll(oidnw_gpu_device *device, uint64_t sequence,
    uint32_t *completed, char *error, size_t capacity) {
  auto status = oidnw_gpu_poll(device, sequence, completed, error, capacity);
  if (fault == 4 && *completed) {
    if (capacity) { std::strncpy(error, "controlled completion error", capacity); error[capacity - 1] = 0; }
    return OIDNW_ERROR_GPU;
  }
  return status;
}
extern "C" int32_t oidnw_cuda_test_submit(oidnw_gpu_prepared *prepared,
    const oidnw_gpu_submit_desc *desc, uint64_t *sequence, char *error, size_t capacity) {
  if (fault == 1) {
    *sequence = 0;
    if (capacity) { std::strncpy(error, "controlled rejection", capacity); error[capacity - 1] = 0; }
    return OIDNW_ERROR_INVALID_ARGUMENT;
  }
  if (fault == 3) {
    entered.store(true);
    while (!resume.load()) std::this_thread::yield();
  }
  auto status = oidnw_gpu_submit(prepared, desc, sequence, error, capacity);
  if (fault == 2 && *sequence) {
    if (capacity) { std::strncpy(error, "controlled accepted error", capacity); error[capacity - 1] = 0; }
    return OIDNW_ERROR_GPU;
  }
  return status;
}
static void ensure(bool condition, const char *message) { if (!condition) throw std::runtime_error(message); }
static void cu(cudaError_t result) { ensure(result == cudaSuccess, cudaGetErrorString(result)); }
static void check(bridge::status status) { ensure(bool(status), status.message.c_str()); }
int main(int argc, char **argv) {
  try {
    ensure(argc == 2, "supply model path");
    cu(cudaSetDevice(0));
    std::ifstream file(argv[1], std::ios::binary | std::ios::ate);
    ensure(bool(file), "cannot open model");
    std::vector<char> weights(size_t(file.tellg()));
    file.seekg(0); file.read(weights.data(), weights.size());
    cudaStream_t stream; cu(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    std::array<void *, 4> allocations{};
    for (auto &p : allocations) { cu(cudaMalloc(&p, 4608)); cu(cudaMemsetAsync(p, 0, 4608, stream)); }
    auto views = [&] {
      auto image = [&](unsigned i) { return bridge::image{allocations[i], 4608, 256, 256, 16, 16}; };
      return bridge::images{image(0), image(1), image(2), image(3)};
    }();
    bridge::options options; options.width = options.height = 16;
    fault = 5;
    bool unmatched = false;
    try { bridge::adapter rejected(options, weights.data(), weights.size()); }
    catch (const std::exception &e) { unmatched = std::string(e.what()) == "no DX12 adapter with the CUDA device LUID"; }
    ensure(unmatched, "unmatched LUID did not reject device creation"); fault = 0;
    for (unsigned mode : {1u, 2u, 4u}) {
      bridge::adapter adapter(options, weights.data(), weights.size());
      fault = mode;
      auto task = adapter.enqueue_copy(views, stream);
      ensure(task.accepted() && (mode == 4 || !task.result) &&
          (mode == 1 || task.sequence() != 0), "accepted error lost tracking");
      if (mode != 4)
        ensure(task.depend(stream).value == bridge::code::backend_error, "failed task exposed output dependency");
      cu(cudaStreamSynchronize(stream));
      ensure(!task.wait(), "error result was discarded");
      bool complete = false; auto queried = task.query(complete);
      ensure(complete && !queried && task.phase() == bridge::task_phase::completed, "failed task did not finish");
      ensure(task.depend(stream).value == bridge::code::backend_error, "completed error exposed output dependency");
      auto rejected = adapter.enqueue_copy(views, stream);
      ensure(!rejected.accepted() && !rejected.result, "adapter continued after failure");
      task = bridge::task{};
      check(adapter.close());
      bridge::adapter shared_adapter(options, weights.data(), weights.size());
      auto lease = shared_adapter.acquire(); check(lease.result);
      auto shared_views = lease.views();
      for (auto view : {shared_views.color, shared_views.albedo, shared_views.normal})
        cu(cudaMemsetAsync(view.data, 0, view.bytes, stream));
      auto shared_task = shared_adapter.enqueue_shared(lease, stream);
      ensure(shared_task.accepted() && (mode == 4 || !shared_task.result), "shared failure lost its owner");
      cu(cudaStreamSynchronize(stream));
      ensure(!shared_task.wait(), "shared error result was discarded");
      complete = false; auto shared_status = shared_task.query(complete);
      ensure(complete && !shared_status, "shared failed task did not finish");
      check(shared_adapter.release(lease, stream)); shared_task = bridge::task{};
      check(shared_adapter.close());
    }
    fault = 0;
    {
      bridge::adapter adapter(options, weights.data(), weights.size());
      bridge::task pending;
      fault = 3;
      std::thread caller([&] { cu(cudaSetDevice(0)); pending = adapter.enqueue_copy(views, stream); });
      while (!entered.load()) std::this_thread::yield();
      auto rejected = adapter.acquire();
      bool busy = !rejected && rejected.result.value == bridge::code::busy &&
          adapter.close(0).value == bridge::code::busy;
      resume.store(true); caller.join(); fault = 0;
      ensure(busy, "concurrent call was not rejected");
      check(pending.result); check(pending.wait()); pending = bridge::task{}; check(adapter.close());
    }
    {
      auto adapter = std::make_unique<bridge::adapter>(options, weights.data(), weights.size());
      auto lease = adapter->acquire(); check(lease.result);
      auto views = lease.views();
      for (auto view : {views.color, views.albedo, views.normal})
        cu(cudaMemsetAsync(view.data, 0, view.bytes, stream));
      auto task = adapter->enqueue_shared(lease, stream); check(task.result);
      adapter.reset();
      check(task.depend(stream));
      cu(cudaMemsetAsync(static_cast<char *>(lease.views().output.data) + views.output.offset,
          0, 16 * 16 * 16, stream));
      check(task.wait());
    }
    cu(cudaStreamSynchronize(stream));
    for (auto p : allocations) cu(cudaFree(p));
    cu(cudaStreamDestroy(stream));
    std::cout << "CUDA adapter rejection, accepted-error and child-owner lifecycle: OK\n";
    return 0;
  } catch (const std::exception &e) { std::cerr << e.what() << std::endl; return 1; }
}
