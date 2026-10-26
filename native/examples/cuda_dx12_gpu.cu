#include "oidn_webgpu_cuda.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <array>
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <utility>
namespace bridge = oidnw::cuda;
__global__ static void guards(const unsigned char *buffer, uint64_t bytes, unsigned *counts);
__global__ static void produce(float4 *color, float4 *albedo, float4 *normal, unsigned n,
                              unsigned p, unsigned width, unsigned pitch) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float x = float((i * 13 + p * 17) % 97) / 97.0f;
  unsigned j = (i / width) * pitch + i % width;
  color[j] = make_float4(x + 0.125f * p, x * 0.5f + 0.1f, x * 0.25f + 0.2f, 1);
  albedo[j] = make_float4(0.2f + x * 0.5f, 0.4f, 0.6f, 1);
  normal[j] = make_float4(0, 0, 1, 1);
}
__global__ static void verify(const float4 *output, const float4 *reference, unsigned n,
                              unsigned *counts, unsigned width, unsigned pitch) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n * 4) return;
  unsigned pixel = i / 4;
  unsigned j = ((pixel / width) * pitch + pixel % width) * 4 + i % 4;
  float a = reinterpret_cast<const float *>(output)[j];
  float b = reinterpret_cast<const float *>(reference)[i];
  if (!isfinite(a)) atomicAdd(counts, 1);
  if (__float_as_uint(a) != __float_as_uint(b)) atomicAdd(counts + 1, 1);
  atomicAdd(counts + 2, 1);
}
__global__ static void delay(unsigned long long ticks) {
  auto start = clock64();
  while (clock64() - start < ticks) {}
}
static void ensure(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
static void cu(cudaError_t result) {
  if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
static void check(const bridge::status &result) { if (!result) throw std::runtime_error(result.message); }
static void driver(CUresult result) {
  if (result != CUDA_SUCCESS) {
    const char *message = nullptr; cuGetErrorString(result, &message);
    throw std::runtime_error(message ? message : "driver error");
  }
}
static float4 *pixels(const bridge::image &view) {
  return reinterpret_cast<float4 *>(static_cast<char *>(view.data) + view.offset);
}
struct Allocation {
  void *pointer = nullptr;
  ~Allocation() { if (pointer && cudaFree(pointer) != cudaSuccess) std::terminate(); }
};
struct Runtime {
  cudaStream_t producer = nullptr, consumer = nullptr;
  std::array<cudaEvent_t, 3> consumed{};
  std::array<std::array<Allocation, 4>, 3> ordinary;
  std::array<Allocation, 7> references;
  Allocation counts;
  Runtime() {
    cu(cudaStreamCreateWithFlags(&producer, cudaStreamNonBlocking));
    cu(cudaStreamCreateWithFlags(&consumer, cudaStreamNonBlocking));
    for (auto &e : consumed) cu(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
  }
  ~Runtime() {
    // Shutdown covers application kernels as well as adapter work.
    if (cudaStreamSynchronize(producer) != cudaSuccess || cudaStreamSynchronize(consumer) != cudaSuccess)
      std::terminate();
    for (auto e : consumed) if (cudaEventDestroy(e) != cudaSuccess) std::terminate();
    if (cudaStreamDestroy(producer) != cudaSuccess || cudaStreamDestroy(consumer) != cudaSuccess) std::terminate();
  }
};
static void negative_paths(bridge::adapter &adapter, bridge::images views,
                           cudaStream_t stream, const bridge::options &options,
                           const std::vector<char> &weights) {
  auto expect = [&](const bridge::images &v) {
    auto t = adapter.enqueue_copy(v, stream);
    ensure(!t.accepted() && !t.result, "invalid image was accepted");
  };
  auto bad = views; bad.output.offset = 4; expect(bad);
  bad = views; bad.color.row_stride = 1; expect(bad);
  bad = views; bad.output = bad.color; expect(bad);
  bad = views; bad.color.pixel_format = static_cast<bridge::format>(99); expect(bad);
  bad = views; bad.normal.bytes = 1; expect(bad);
  void *managed = nullptr; cu(cudaMallocManaged(&managed, views.color.bytes));
  bad = views; bad.color.data = managed; expect(bad); cu(cudaFree(managed));
  void *host = nullptr; cu(cudaMallocHost(&host, views.color.bytes));
  bad = views; bad.color.data = host; expect(bad); cu(cudaFreeHost(host));
  void *pooled = nullptr; cu(cudaMallocAsync(&pooled, views.color.bytes, stream));
  bad = views; bad.color.data = pooled; expect(bad); cu(cudaFreeAsync(pooled, stream));
  cu(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  auto captured = adapter.enqueue_copy(views, stream);
  ensure(!captured.accepted() && captured.result.value == bridge::code::unsupported, "capture accepted");
  cudaGraph_t graph = nullptr; cu(cudaStreamEndCapture(stream, &graph)); cu(cudaGraphDestroy(graph));
  CUcontext primary, other; CUdevice device;
  driver(cuCtxGetCurrent(&primary)); driver(cuCtxGetDevice(&device));
  driver(cuCtxCreate(&other, 0, device));
  CUstream foreign; driver(cuStreamCreate(&foreign, CU_STREAM_NON_BLOCKING));
  CUdeviceptr foreign_memory; driver(cuMemAlloc(&foreign_memory, views.color.bytes));
  auto wrong = adapter.acquire();
  ensure(!wrong && wrong.result.value == bridge::code::unsupported, "foreign context accepted");
  bool rejected_context = false;
  try { bridge::adapter rejected(options, weights.data(), weights.size()); }
  catch (const std::exception &) { rejected_context = true; }
  CUcontext restored; driver(cuCtxGetCurrent(&restored));
  ensure(rejected_context && restored == other, "construction changed a foreign context");
  driver(cuCtxSetCurrent(primary));
  bad = views; bad.color.data = reinterpret_cast<void *>(foreign_memory); expect(bad);
  auto wrong_stream = adapter.enqueue_copy(views, reinterpret_cast<cudaStream_t>(foreign));
  ensure(!wrong_stream.accepted() && !wrong_stream.result, "foreign stream accepted");
  driver(cuCtxSetCurrent(other)); driver(cuStreamDestroy(foreign)); driver(cuMemFree(foreign_memory));
  driver(cuCtxSetCurrent(primary)); driver(cuCtxDestroy(other));
  bridge::adapter other_adapter(options, weights.data(), weights.size());
  auto lease = other_adapter.acquire(); check(lease.result); ensure(bool(lease), "acquire failed");
  auto cross = adapter.enqueue_shared(lease, stream);
  ensure(!cross.accepted() && cross.result.value == bridge::code::invalid_argument, "cross-adapter images accepted");
  ensure(other_adapter.close(0).value == bridge::code::busy, "held images did not keep adapter alive");
  check(other_adapter.release(lease, stream));
  check(other_adapter.close(UINT64_MAX));
}
int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "usage: cuda_dx12_gpu WEIGHTS [frames=1000] [size=16] [tile=0] [fp32=0] "
                 "[copy=0] [checks=1] [gpu_validation=0] [diagnostic=0]\n";
    return 2;
  }
  try {
    unsigned frames = argc > 2 ? std::stoul(argv[2]) : 1000;
    unsigned size = argc > 3 ? std::stoul(argv[3]) : 16;
    bool copy = argc > 6 && std::stoi(argv[6]);
    bool diagnostic = argc > 9 && std::stoi(argv[9]);
    bridge::options options;
    options.width = options.height = size;
    options.max_tile_size = argc > 4 ? std::stoul(argv[4]) : 0;
    options.fp32 = argc > 5 && std::stoi(argv[5]);
    options.runtime_checks = argc <= 7 || std::stoi(argv[7]);
    options.gpu_validation = argc > 8 && std::stoi(argv[8]);
    options.debug = true;
    ensure(frames >= 4 && frames <= 100000 && size && size <= 1024 &&
        (uint64_t(frames) + 1) * size * size * 4 <= UINT32_MAX, "invalid dimensions/frame count");
    cu(cudaSetDevice(0));
    cudaDeviceProp properties{}; cu(cudaGetDeviceProperties(&properties, 0));
    std::cerr << "GPU: " << properties.name << ", mode=" << (copy ? "copy" : "shared") << std::endl;
    std::ifstream file(argv[1], std::ios::binary | std::ios::ate);
    ensure(bool(file), "cannot open model");
    std::vector<char> weights(static_cast<size_t>(file.tellg()));
    file.seekg(0); file.read(weights.data(), weights.size()); ensure(bool(file), "cannot read model");
    Runtime runtime;
    unsigned n = size * size;
    uint64_t bytes = uint64_t(n) * 16;
    uint64_t ordinary_stride = uint64_t(size + 2) * 16;
    uint64_t ordinary_bytes = 256 + ordinary_stride * size + 256;
    for (auto &slot : runtime.ordinary) for (auto &a : slot) cu(cudaMalloc(&a.pointer, ordinary_bytes));
    for (auto &a : runtime.references) cu(cudaMalloc(&a.pointer, bytes));
    cu(cudaMalloc(&runtime.counts.pointer, sizeof(unsigned) * 4));
    auto counts = static_cast<unsigned *>(runtime.counts.pointer);
    cu(cudaMemsetAsync(counts, 0, sizeof(unsigned) * 4, runtime.consumer));
    auto ordinary = [&](unsigned i) {
      auto image = [&](unsigned j) {
        return bridge::image{runtime.ordinary[i][j].pointer, ordinary_bytes, 256,
            ordinary_stride, size, size, bridge::format::rgba32f};
      };
      return bridge::images{image(0), image(1), image(2), image(3)};
    };
    bridge::adapter adapter(options, weights.data(), weights.size());
    auto info = adapter.info();
    int runtime_version, driver_version;
    cu(cudaRuntimeGetVersion(&runtime_version)); cu(cudaDriverGetVersion(&driver_version));
    std::cerr << "DLL: " << info.core_dll_path << "; API=" << info.api_version
        << "; Agility=" << info.agility_sdk_version << "; MSVC=" << info.msvc_version
        << "; CUDA runtime/driver=" << runtime_version << '/' << driver_version << std::endl;
    negative_paths(adapter, ordinary(0), runtime.producer, options, weights);
    std::cerr << "adapter ready; rejection paths complete" << std::endl;
    const float scales[7] = {1, 2, 0.5f, 4, 0.25f, 1.5f, 0.75f};
    auto production = [&](const bridge::images &views, unsigned pattern) {
      for (auto image : {views.color, views.albedo, views.normal, views.output})
        cu(cudaMemsetAsync(image.data, 0xa5, image.bytes, runtime.producer));
      cu(cudaMemset2DAsync(pixels(views.output), views.output.row_stride, 0xff, size * 16, size, runtime.producer));
      produce<<<(n + 63) / 64, 64, 0, runtime.producer>>>(pixels(views.color), pixels(views.albedo),
          pixels(views.normal), n, pattern, size, unsigned(views.color.row_stride / 16));
      cu(cudaGetLastError());
    };
    auto enqueue = [&](bridge::shared_images &lease, const bridge::images &views, unsigned pattern) {
      auto task = copy ? adapter.enqueue_copy(views, runtime.producer, scales[pattern]) :
                         adapter.enqueue_shared(lease, runtime.producer, scales[pattern]);
      // Accepted errors keep their task owner until its work can be drained.
      check(task.result); ensure(task.accepted(), "successful enqueue was not accepted");
      return task;
    };
    for (unsigned p = 0; p < 7; ++p) {
      auto lease = adapter.acquire();
      check(lease.result); ensure(bool(lease), "reference acquire failed");
      auto views = lease.views();
      production(views, p);
      auto task = adapter.enqueue_shared(lease, runtime.producer, scales[p]);
      check(task.result); ensure(task.accepted(), "reference enqueue rejected");
      check(task.depend(runtime.consumer));
      cu(cudaMemcpy2DAsync(runtime.references[p].pointer, size * 16, pixels(views.output),
          views.output.row_stride, size * 16, size, cudaMemcpyDeviceToDevice, runtime.consumer));
      check(adapter.release(lease, runtime.consumer));
      cu(cudaStreamSynchronize(runtime.consumer));
      check(task.wait());
    }
    std::cerr << "serial references complete" << std::endl;
    bool retirement_pending = false;
    if (!copy) {
      auto first = adapter.acquire(), second = adapter.acquire(), third = adapter.acquire();
      check(first.result); check(second.result); check(third.result);
      auto views = first.views(); production(views, 0);
      auto task = enqueue(first, views, 0); check(task.wait());
      check(task.depend(runtime.consumer));
      delay<<<1, 1, 0, runtime.consumer>>>(uint64_t(properties.clockRate) * 500);
      cu(cudaGetLastError());
      verify<<<(n * 4 + 63) / 64, 64, 0, runtime.consumer>>>(pixels(views.output),
          static_cast<float4 *>(runtime.references[0].pointer), n, counts, size, size);
      cu(cudaGetLastError());
      cu(cudaEventRecord(runtime.consumed[0], runtime.consumer));
      check(adapter.release(first, runtime.consumer)); task = bridge::task{};
      auto busy = adapter.acquire();
      retirement_pending = !busy && busy.result.value == bridge::code::busy;
      if (!retirement_pending) {
        ensure(diagnostic && bool(busy), "slot reused before CUDA consumer retired");
        cu(cudaEventQuery(runtime.consumed[0]));
        check(adapter.release(busy, runtime.consumer));
      }
      check(adapter.release(second, runtime.producer)); check(adapter.release(third, runtime.producer));
      cu(cudaStreamSynchronize(runtime.consumer)); check(adapter.drain());
    }
    auto initial_stats = adapter.stats();
    std::array<bridge::task, 3> pending;
    unsigned backpressure = 0, pending_waits = 0;
    for (unsigned frame = 0; frame < frames; ++frame) {
      unsigned slot = frame % 3, pattern = frame % 7;
      if (frame >= 3) {
        if (copy) {
          auto busy = adapter.enqueue_copy(ordinary(slot), runtime.producer);
          ensure(!busy.accepted() && busy.result.value == bridge::code::busy, "pool grew beyond three slots");
        } else {
          auto busy = adapter.acquire();
          ensure(!busy && busy.result.value == bridge::code::busy, "shared pool grew beyond three slots");
        }
        ++backpressure;
        cu(cudaEventSynchronize(runtime.consumed[slot]));
        check(pending[slot].wait()); pending[slot] = bridge::task{};
      }
      auto lease = copy ? bridge::shared_images{} : adapter.acquire();
      if (!copy) { check(lease.result); ensure(bool(lease), "frame acquire failed"); }
      auto views = copy ? ordinary(slot) : lease.views();
      if (!frame || frame + 1 == frames) {
        delay<<<1, 1, 0, runtime.producer>>>(uint64_t(properties.clockRate) * 500);
        cu(cudaGetLastError());
      }
      production(views, pattern);
      auto task = enqueue(lease, views, pattern);
      if (!frame) {
        auto result = task.wait(0);
        if (result.value == bridge::code::timeout) ++pending_waits;
        else { check(result); ensure(diagnostic, "delayed producer did not keep task pending"); }
      }
      cudaStream_t consumer = frame % 2 ? runtime.consumer : runtime.producer;
      if (consumer != runtime.producer) check(task.depend(consumer));
      if (frame % 101 == 0) {
        delay<<<1, 1, 0, consumer>>>(uint64_t(properties.clockRate) * 5);
        cu(cudaGetLastError());
      }
      verify<<<(n * 4 + 63) / 64, 64, 0, consumer>>>(pixels(views.output),
          static_cast<float4 *>(runtime.references[pattern].pointer), n, counts,
          size, unsigned(views.output.row_stride / 16));
      cu(cudaGetLastError());
      for (auto image : {views.color, views.albedo, views.normal, views.output}) {
        guards<<<1, 256, 0, consumer>>>(static_cast<unsigned char *>(image.data), image.bytes, counts);
        cu(cudaGetLastError());
      }
      if (!copy) check(adapter.release(lease, consumer));
      cu(cudaEventRecord(runtime.consumed[slot], consumer));
      pending[slot] = std::move(task);
    }
    auto timed = adapter.close(0);
    bool close_timeout = timed.value == bridge::code::timeout;
    ensure(close_timeout || (diagnostic && timed.value == bridge::code::busy),
        "close did not preserve an in-flight task on timeout");
    for (unsigned i = 0; i < 3; ++i) {
      cu(cudaEventSynchronize(runtime.consumed[i]));
      check(pending[i].wait()); pending[i] = bridge::task{};
    }
    check(adapter.drain()); check(adapter.close()); check(adapter.close(0));
    auto final_stats = adapter.stats();
    uint64_t input_copies = final_stats.input_copy_bytes - initial_stats.input_copy_bytes;
    uint64_t output_copies = final_stats.output_copy_bytes - initial_stats.output_copy_bytes;
    ensure(input_copies == (copy ? uint64_t(frames) * bytes * 3 : 0) &&
        output_copies == (copy ? uint64_t(frames) * bytes : 0), "wrong bridge copy accounting");
    std::array<unsigned, 4> result{};
    cu(cudaMemcpy(result.data(), counts, sizeof(result), cudaMemcpyDeviceToHost));
    ensure(!result[0] && !result[1] && !result[3] && result[2] == (frames + unsigned(!copy)) * n * 4,
        "GPU finite/bitwise/coverage/guard comparison failed");
    std::cout << "{\"mode\":\"" << (copy ? "copy" : "shared") << "\",\"frames\":" << frames
        << ",\"size\":" << size << ",\"tile\":" << options.max_tile_size << ",\"precision\":\""
        << (adapter.uses_fp16() ? "fp16" : "fp32") << "\",\"runtime_checks\":" << options.runtime_checks
        << ",\"gpu_validation\":" << options.gpu_validation << ",\"input_copy_bytes\":" << input_copies
        << ",\"output_copy_bytes\":" << output_copies << ",\"shared_allocation_bytes\":" << final_stats.shared_allocation_bytes
        << ",\"backpressure\":" << backpressure << ",\"pending_waits\":" << pending_waits
        << ",\"retirement_pending\":" << retirement_pending << ",\"close_timeout\":" << close_timeout
        << ",\"diagnostic\":" << diagnostic
        << ",\"output_byte_equal\":true,\"cpu_image_transfers\":0}" << std::endl;
    return 0;
  } catch (const std::exception &e) { std::cerr << e.what() << std::endl; return 1; }
}
__global__ static void guards(const unsigned char *buffer, uint64_t bytes, unsigned *counts) {
  unsigned i = threadIdx.x;
  if (buffer[i] != 0xa5 || buffer[bytes - 256 + i] != 0xa5) atomicAdd(counts + 3, 1);
}
