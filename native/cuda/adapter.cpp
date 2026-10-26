#define NOMINMAX
#define INITGUID
#include "oidn_webgpu_cuda.hpp"
#include "oidn_webgpu_dx12.hpp"
#include <cuda.h>
#include <d3d12sdklayers.h>
#include <dxgi1_4.h>
#include <wrl/client.h>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>
namespace oidnw::cuda::detail {
using Microsoft::WRL::ComPtr;
struct failure : std::runtime_error {
  code value;
  failure(code c, const std::string &m) : std::runtime_error(m), value(c) {}
};
static void require(bool ok, code c, const char *message) { if (!ok) throw failure(c, message); }
static void hr(HRESULT r) {
  if (FAILED(r)) throw failure(code::backend_error, "HRESULT " + std::to_string(uint32_t(r)));
}
static void cu(cudaError_t r) {
  if (r != cudaSuccess) throw failure(code::cuda_error, cudaGetErrorString(r));
}
static void driver(CUresult r) {
  if (r == CUDA_SUCCESS) return;
  const char *message = nullptr;
  cuGetErrorString(r, &message);
  throw failure(code::cuda_error, message ? message : "CUDA driver error");
}
static void cleanup(cudaError_t r) noexcept { if (r != cudaSuccess) std::terminate(); }
static void check(const gpu::status &s) {
  if (!s) throw failure(s.code == OIDNW_ERROR_TIMEOUT ? code::timeout : code::backend_error, s.message.data());
}
static status caught() noexcept {
  try { throw; }
  catch (const failure &e) {
    try { return {e.value, e.what()}; } catch (...) { return {e.value, {}}; }
  }
  catch (const std::exception &e) {
    try { return {code::backend_error, e.what()}; } catch (...) { return {code::backend_error, {}}; }
  }
  catch (...) { return {code::backend_error, {}}; }
}
struct Handle {
  HANDLE value = nullptr;
  ~Handle() { if (value) CloseHandle(value); }
};
#include "resources.hpp"
struct context_scope {
  bool pushed = false;
  explicit context_scope(CUcontext context) {
    CUcontext current = nullptr;
    driver(cuCtxGetCurrent(&current));
    if (context != current) { driver(cuCtxPushCurrent(context)); pushed = true; }
  }
  ~context_scope() { if (pushed) { CUcontext old; if (cuCtxPopCurrent(&old) != CUDA_SUCCESS) std::terminate(); } }
};
struct native_owner {
  CUcontext context = nullptr;
  CUdevice device = 0;
  bool retained = false;
  HMODULE layers = nullptr;
  ComPtr<ID3D12DeviceFactory> factory;
  ComPtr<ID3D12Device> host;
  struct storage {
    std::array<SharedBuffer, 4> buffers;
    SharedFence ready, done;
    cudaEvent_t completed = nullptr, retired = nullptr;
    ~storage() {
      if (completed) cleanup(cudaEventDestroy(completed));
      if (retired) cleanup(cudaEventDestroy(retired));
    }
  };
  std::vector<std::unique_ptr<storage>> slots;
  ~native_owner() {
    if (context) {
      context_scope scope(context);
      slots.clear(); host.Reset(); factory.Reset();
    }
    if (layers) FreeLibrary(layers);
    if (retained && cuDevicePrimaryCtxRelease(device) != CUDA_SUCCESS) std::terminate();
  }
};
struct slot {
  std::array<gpu::buffer, 4> buffers;
  gpu::sync ready, done;
  uint64_t generation = 0, point = 0, sequence = 0;
  bool allocated = false, shared = false, leased = false, task_held = false;
  bool enqueued = false, event_recorded = false, retired_recorded = false, uncertain = false;
  status result;
};
struct state {
  std::mutex mutex;
  options opts;
  std::shared_ptr<native_owner> native = std::make_shared<native_owner>();
  gpu::device context;
  gpu::model model;
  gpu::filter filter;
  gpu::workspace workspace;
  std::vector<slot> slots;
  statistics counters;
  bool closed = false, poisoned = false, fp16 = false;
  void current() const {
    CUcontext active = nullptr;
    driver(cuCtxGetCurrent(&active));
    require(active == native->context, code::unsupported, "adapter requires its CUDA primary context current");
    require(!closed, code::closed, "adapter is closed");
  }
  void stream(cudaStream_t value) const {
    current();
    require(value && value != cudaStreamLegacy && value != cudaStreamPerThread,
        code::unsupported, "an explicit CUDA stream is required");
    CUcontext owner = nullptr;
    driver(cuStreamGetCtx(reinterpret_cast<CUstream>(value), &owner));
    require(owner == native->context, code::unsupported, "stream belongs to another CUDA context");
    cudaStreamCaptureStatus capture;
    cu(cudaStreamIsCapturing(value, &capture));
    require(capture == cudaStreamCaptureStatusNone, code::unsupported, "CUDA Graph capture is unsupported");
  }
  oidnw_gpu_images bindings(uint32_t i) const {
    auto &b = slots[i].buffers;
    return {{b[0].get(), 256}, {b[1].get(), 256}, {b[2].get(), 256}, {b[3].get(), 256}};
  }
  images views(uint32_t i) const {
    auto view = [&](uint32_t b) {
      auto &buffer = native->slots[i]->buffers[b];
      return image{buffer.mapped, buffer.logical_bytes, 256, uint64_t(opts.width) * 16,
          opts.width, opts.height, format::rgba32f};
    };
    return {view(0), view(1), view(2), view(3)};
  }
  void initialize(const options &options, const void *weights, size_t bytes) {
    opts = options;
    require(opts.width && opts.height && opts.width <= 16384 && opts.height <= 16384 &&
        opts.slots >= 1 && opts.slots <= 64, code::invalid_argument, "invalid dimensions/slot count");
    require(opts.transfer <= OIDNW_TRANSFER_LOG, code::invalid_argument, "unknown transfer");
    require(!opts.gpu_validation || opts.debug, code::invalid_argument, "GPU validation requires debug");
    driver(cuInit(0));
    CUcontext active = nullptr;
    driver(cuCtxGetCurrent(&active));
    require(active != nullptr, code::unsupported, "initialize CUDA primary context before creating adapter");
    driver(cuCtxGetDevice(&native->device));
    driver(cuDevicePrimaryCtxRetain(&native->context, native->device));
    native->retained = true;
    require(active == native->context, code::unsupported, "only the CUDA primary context is supported");
    if (opts.agility_sdk_path.empty()) {
      const char *path = getenv("WGPU_DX12_AGILITY_SDK_PATH");
      if (path) opts.agility_sdk_path = path;
    }
    if (!opts.agility_sdk_version) {
      const char *version = getenv("WGPU_DX12_AGILITY_SDK_VERSION");
      if (version) opts.agility_sdk_version = std::stoul(version);
    }
    require(!opts.agility_sdk_path.empty() && opts.agility_sdk_version, code::invalid_argument,
        "provide matching Agility SDK directory and version in options or environment");
    const auto &path = opts.agility_sdk_path;
    require(path.size() > 3 && path[1] == ':' && (path.back() == '/' || path.back() == '\\'),
        code::invalid_argument, "Agility SDK directory must be absolute with trailing separator");
    if (opts.debug) {
      native->layers = LoadLibraryA((path + "d3d12SDKLayers.dll").c_str());
      require(native->layers != nullptr, code::backend_error, "Agility SDK debug layer missing");
    }
    ComPtr<ID3D12SDKConfiguration1> configuration;
    hr(D3D12GetInterface(CLSID_D3D12SDKConfiguration, IID_PPV_ARGS(&configuration)));
    hr(configuration->CreateDeviceFactory(opts.agility_sdk_version, path.c_str(), IID_PPV_ARGS(&native->factory)));
    if (opts.debug) {
      ComPtr<ID3D12Debug> debug;
      hr(native->factory->GetConfigurationInterface(CLSID_D3D12Debug, IID_PPV_ARGS(&debug)));
      debug->EnableDebugLayer();
      if (opts.gpu_validation) {
        ComPtr<ID3D12Debug1> validation;
        hr(debug.As(&validation)); validation->SetEnableGPUBasedValidation(TRUE);
      }
    }
    char luid[8]{}; unsigned node = 0;
    driver(cuDeviceGetLuid(luid, &node, native->device));
    require(node == 1, code::unsupported, "linked CUDA node adapters are unsupported");
    ComPtr<IDXGIFactory4> dxgi;
    hr(CreateDXGIFactory2(0, IID_PPV_ARGS(&dxgi)));
    for (UINT i = 0; !native->host; ++i) {
      ComPtr<IDXGIAdapter1> adapter;
      HRESULT r = dxgi->EnumAdapters1(i, &adapter);
      require(r != DXGI_ERROR_NOT_FOUND, code::unsupported, "no DX12 adapter with the CUDA device LUID");
      hr(r);
      DXGI_ADAPTER_DESC1 info{}; hr(adapter->GetDesc1(&info));
      if (memcmp(&info.AdapterLuid.LowPart, luid, 4) || memcmp(&info.AdapterLuid.HighPart, luid + 4, 4)) continue;
      hr(native->factory->CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&native->host)));
    }
    require(native->host->GetNodeCount() == 1, code::unsupported, "linked DX12 node adapters are unsupported");
    oidnw_gpu_dx12_device_desc desc;
    oidnw_gpu_dx12_device_desc_default(&desc);
    desc.host_device = native->host.Get();
    desc.agility_sdk_path = path.c_str(); desc.agility_sdk_version = opts.agility_sdk_version;
    desc.max_pending_submissions = opts.slots;
    desc.flags = (opts.fp32 ? OIDNW_GPU_DX12_FP32 : 0) | (opts.debug ? OIDNW_GPU_DX12_DEBUG : 0) |
        (opts.gpu_validation ? OIDNW_GPU_DX12_GPU_VALIDATION : 0);
    gpu::device_options lifetime; lifetime.host_lifetime = native;
    context = dx12::create_device(desc, lifetime);
    uint32_t enabled = 0;
    gpu::require(oidnw_gpu_device_info(context.get(), nullptr, 0, nullptr, &enabled), "device info failed");
    fp16 = (enabled & OIDNW_FEATURE_SHADER_F16) != 0;
    require(opts.fp32 || fp16, code::unsupported, "requested FP16 is unavailable");
    model = gpu::load_model(weights, bytes);
    auto filter_options = gpu::filter_options();
    filter_options.hdr = opts.hdr ? 1 : 0;
    filter_options.transfer = opts.transfer;
    filter_options.flags = opts.runtime_checks ? OIDNW_FILTER_RUNTIME_CHECKS : 0;
    filter_options.max_tile_size = opts.max_tile_size;
    filter_options.binding_cache_capacity = opts.slots;
    filter = gpu::make_filter(context, model, filter_options);
    rebuild_images(opts.width, opts.height);
  }
  void rebuild_images(uint32_t width, uint32_t height) {
    std::vector<std::unique_ptr<native_owner::storage>> new_native;
    std::vector<slot> new_slots(opts.slots);
    uint64_t allocation_bytes = 0;
    for (uint32_t i = 0; i < opts.slots; ++i) {
      new_native.push_back(std::make_unique<native_owner::storage>());
      auto &storage = *new_native.back();
      auto &slot = new_slots[i];
      for (unsigned b = 0; b < 4; ++b) {
        storage.buffers[b].create(native->host.Get(), uint64_t(width) * height * 16);
        allocation_bytes += storage.buffers[b].allocation_bytes;
        slot.buffers[b] = dx12::import_buffer(context, storage.buffers[b].resource.Get());
      }
      storage.ready.create(native->host.Get()); storage.done.create(native->host.Get());
      slot.ready = dx12::import_sync(context, storage.ready.fence.Get());
      slot.done = dx12::import_sync(context, storage.done.fence.Get());
      cu(cudaEventCreateWithFlags(&storage.completed, cudaEventDisableTiming));
      cu(cudaEventCreateWithFlags(&storage.retired, cudaEventDisableTiming));
    }
    auto &buffers = new_slots[0].buffers;
    oidnw_gpu_images images = {{buffers[0].get(), 256}, {buffers[1].get(), 256},
        {buffers[2].get(), 256}, {buffers[3].get(), 256}};
    auto new_workspace = gpu::prepare(filter, width, height, images);
    workspace = std::move(new_workspace);
    slots = std::move(new_slots);
    native->slots = std::move(new_native);
    opts.width = width; opts.height = height;
    counters.shared_allocation_bytes = allocation_bytes;
  }
  void validate(const image &view) const {
    require(view.data && view.width == opts.width && view.height == opts.height &&
        view.pixel_format == format::rgba32f && view.offset % 16 == 0 &&
        view.row_stride >= uint64_t(opts.width) * 16 && view.row_stride % 16 == 0,
        code::invalid_argument, "images require matching dimensions and aligned RGBA32F rows");
    require(view.row_stride <= UINT64_MAX / opts.height && view.offset <= view.bytes &&
        view.row_stride * (opts.height - 1) <= view.bytes - view.offset &&
        uint64_t(opts.width) * 16 <= view.bytes - view.offset - view.row_stride * (opts.height - 1),
        code::invalid_argument, "image range exceeds declared bytes");
    CUdeviceptr pointer = reinterpret_cast<CUdeviceptr>(view.data);
    CUmemorytype type; unsigned managed = 0; CUcontext owner = nullptr; int device = -1;
    driver(cuPointerGetAttribute(&type, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, pointer));
    require(type == CU_MEMORYTYPE_DEVICE, code::unsupported, "host and non-device pointers are unsupported");
    driver(cuPointerGetAttribute(&managed, CU_POINTER_ATTRIBUTE_IS_MANAGED, pointer));
    driver(cuPointerGetAttribute(&owner, CU_POINTER_ATTRIBUTE_CONTEXT, pointer));
    driver(cuPointerGetAttribute(&device, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, pointer));
    require(type == CU_MEMORYTYPE_DEVICE && !managed && owner == native->context && device == native->device,
        code::unsupported, "only same-device primary-context CUDA device allocations are supported");
    CUmemoryPool pool = nullptr;
    driver(cuPointerGetAttribute(&pool, CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE, pointer));
    require(pool == nullptr, code::unsupported, "CUDA memory-pool allocations are unsupported");
    CUdeviceptr base; size_t size;
    driver(cuMemGetAddressRange(&base, &size, pointer));
    require(pointer >= base && pointer - base <= size && view.bytes <= size - (pointer - base),
        code::invalid_argument, "image bytes exceed CUDA allocation");
  }
  bool complete(uint32_t i, bool retirement = false) {
    auto &slot = slots[i];
    if (slot.uncertain) throw failure(code::cuda_error, "CUDA completion is uncertain; resources retained");
    if (slot.enqueued) {
      if (!slot.event_recorded) return false;
      auto r = cudaEventQuery(native->slots[i]->completed);
      if (r == cudaErrorNotReady) return false;
      if (r != cudaSuccess) { slot.uncertain = true; poisoned = true; }
      cu(r);
      if (slot.sequence) {
        auto result = context.poll(slot.sequence);
        if (!result) {
          slot.result = {code::backend_error, result.message.data()};
          poisoned = true;
        }
        if (!result.completed) { check(result); return false; }
      }
    }
    if (retirement && slot.retired_recorded) {
      auto r = cudaEventQuery(native->slots[i]->retired);
      if (r == cudaErrorNotReady) return false;
      if (r != cudaSuccess) { slot.uncertain = true; poisoned = true; }
      cu(r);
    }
    return true;
  }
  uint32_t reserve(bool shared) {
    require(!poisoned, code::backend_error, "adapter stopped after an enqueue failure");
    for (uint32_t i = 0; i < slots.size(); ++i) {
      auto &slot = slots[i];
      if (slot.allocated && !slot.leased && !slot.task_held && complete(i, true)) slot.allocated = false;
      if (slot.allocated) continue;
      slot.allocated = true; slot.shared = shared; slot.leased = shared; slot.task_held = !shared;
      slot.enqueued = slot.event_recorded = slot.retired_recorded = slot.uncertain = false;
      slot.sequence = 0; slot.result = {}; ++slot.generation;
      return i;
    }
    throw failure(code::busy, "all adapter slots are held or in flight");
  }
  void enqueue(uint32_t i, const images *source, cudaStream_t stream, float scale) {
    auto &slot = slots[i]; auto &storage = *native->slots[i];
    slot.enqueued = true; slot.task_held = true;
    try {
      require(slot.point < UINT64_MAX - 1, code::invalid_argument, "fence timeline exhausted; create a new adapter");
      if (source) {
        const image *inputs[] = {&source->color, &source->albedo, &source->normal};
        for (unsigned b = 0; b < 3; ++b) {
          const auto &view = *inputs[b];
          cu(cudaMemcpy2DAsync(storage.buffers[b].pixels(), uint64_t(opts.width) * 16,
              static_cast<char *>(view.data) + view.offset, view.row_stride,
              uint64_t(opts.width) * 16, opts.height, cudaMemcpyDeviceToDevice, stream));
          counters.input_copy_bytes += uint64_t(opts.width) * opts.height * 16;
        }
      }
      storage.ready.signal(stream, ++slot.point);
      auto submitted = workspace.submit(bindings(i), {slot.ready.get(), slot.point}, {slot.done.get(), slot.point}, scale);
      slot.sequence = submitted.sequence;
      check(submitted);
      require(submitted.accepted(), code::backend_error, "DLL did not accept the submission");
      storage.done.wait(stream, slot.point);
      if (source) {
        auto &view = source->output;
        cu(cudaMemcpy2DAsync(static_cast<char *>(view.data) + view.offset, view.row_stride,
            storage.buffers[3].pixels(), uint64_t(opts.width) * 16, uint64_t(opts.width) * 16,
            opts.height, cudaMemcpyDeviceToDevice, stream));
        counters.output_copy_bytes += uint64_t(opts.width) * opts.height * 16;
        ++counters.copy_enqueues;
      } else ++counters.shared_enqueues;
      cu(cudaEventRecord(storage.completed, stream)); slot.event_recorded = true;
    } catch (...) {
      slot.result = caught(); poisoned = true;
      // No CUDA wait is issued after a failed DLL submit, including accepted errors.
      auto r = cudaEventRecord(storage.completed, stream);
      slot.event_recorded = r == cudaSuccess;
      slot.uncertain = r != cudaSuccess;
    }
  }
  status wait_slot(uint32_t i, uint64_t timeout, bool retirement = false) {
    auto start = std::chrono::steady_clock::now();
    for (;;) {
      if (complete(i, retirement)) return slots[i].result;
      if (timeout != UINT64_MAX && uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - start).count()) >= timeout) return {code::timeout, "CUDA task timeout"};
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
  }
  void abandon(uint32_t i, uint64_t generation, bool lease) noexcept {
    try {
      context_scope scope(native->context);
      std::lock_guard<std::mutex> lock(mutex);
      auto &slot = slots[i];
      if (slot.generation != generation) std::terminate();
      if (lease && slot.leased) {
        // An omitted explicit release needs context-wide shutdown proof for unknown consumers.
        driver(cuCtxSynchronize());
        auto result = wait_slot(i, UINT64_MAX);
        require(result.value != code::timeout, result.value, result.message.c_str());
        slot.leased = false;
      }
      if (!lease) slot.task_held = false;
    } catch (...) { std::terminate(); }
  }
  status drain(uint64_t timeout, bool close) {
    for (auto &slot : slots) if (slot.leased) return {code::busy, "release shared image leases before drain/close"};
    auto start = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < slots.size(); ++i) {
      if (!slots[i].allocated) continue;
      uint64_t remaining = timeout;
      if (timeout != UINT64_MAX) {
        auto elapsed = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start).count());
        remaining = elapsed >= timeout ? 0 : timeout - elapsed;
      }
      auto result = wait_slot(i, remaining, true);
      if (result.value == code::timeout) return result;
    }
    if (close) {
      for (auto &slot : slots) if (slot.task_held) return {code::busy, "release task handles before close"};
      if (opts.debug) {
        ComPtr<ID3D12InfoQueue> info; hr(native->host.As(&info));
        for (uint64_t i = 0; i < info->GetNumStoredMessagesAllowedByRetrievalFilter(); ++i) {
          SIZE_T bytes = 0; hr(info->GetMessage(i, nullptr, &bytes));
          std::vector<uint64_t> data((bytes + 7) / 8);
          auto message = reinterpret_cast<D3D12_MESSAGE *>(data.data());
          hr(info->GetMessage(i, message, &bytes));
          require(message->Severity > D3D12_MESSAGE_SEVERITY_ERROR, code::backend_error, message->pDescription);
        }
      }
      workspace.reset(); filter.reset(); model.reset();
      for (auto &slot : slots) { for (auto &b : slot.buffers) b.reset(); slot.ready.reset(); slot.done.reset(); }
      check(context.close(timeout));
      native->slots.clear(); native->host.Reset(); native->factory.Reset();
      closed = true;
    }
    return {};
  }
  ~state() noexcept {
    if (!native->context || !context || closed) return;
    try {
      context_scope scope(native->context);
      for (auto &slot : slots) slot.leased = slot.task_held = false;
      auto result = drain(UINT64_MAX, true);
      if (!result) std::terminate();
    } catch (...) { std::terminate(); }
  }
};
static std::unique_lock<std::mutex> lock(state &s) {
  std::unique_lock<std::mutex> result(s.mutex, std::try_to_lock);
  require(result.owns_lock(), code::busy, "concurrent host calls must be serialized");
  return result;
}
}
namespace oidnw::cuda {
using detail::require;
adapter::adapter(const options &opts, const void *weights, size_t bytes) : state_(std::make_shared<detail::state>()) {
  state_->initialize(opts, weights, bytes);
}
adapter::adapter(adapter &&) noexcept = default;
adapter &adapter::operator=(adapter &&) noexcept = default;
adapter::~adapter() = default;
task::task(std::shared_ptr<detail::state> state, uint32_t i, uint64_t generation, status value)
    : state_(std::move(state)), slot_(i), generation_(generation), result(std::move(value)) {}
task::task(task &&other) noexcept : state_(std::move(other.state_)), slot_(other.slot_),
    generation_(other.generation_), result(std::move(other.result)) {}
task &task::operator=(task &&other) noexcept {
  if (this != &other) { task old(std::move(*this)); state_ = std::move(other.state_);
    slot_ = other.slot_; generation_ = other.generation_; result = std::move(other.result); }
  return *this;
}
task::~task() { if (state_) state_->abandon(slot_, generation_, false); }
bool task::accepted() const noexcept { return bool(state_); }
uint64_t task::sequence() const noexcept { return state_ ? state_->slots[slot_].sequence : 0; }
status task::query(bool &completed) const {
  completed = false;
  try {
    require(bool(state_), code::invalid_argument, "task was not accepted");
    auto lock = detail::lock(*state_); state_->current();
    completed = state_->complete(slot_); return state_->slots[slot_].result;
  } catch (...) { return detail::caught(); }
}
task_phase task::phase() const {
  if (!accepted()) return task_phase::not_accepted;
  bool completed = false; auto status = query(completed);
  if (completed) return task_phase::completed;
  return status ? task_phase::accepted : task_phase::accepted_error;
}
status task::wait(uint64_t timeout) const {
  try {
    require(bool(state_), code::invalid_argument, "task was not accepted");
    auto lock = detail::lock(*state_); state_->current(); return state_->wait_slot(slot_, timeout);
  } catch (...) { return detail::caught(); }
}
status task::depend(cudaStream_t stream) const {
  try {
    require(bool(state_), code::invalid_argument, "task was not accepted");
    auto lock = detail::lock(*state_); state_->stream(stream);
    require(bool(result) && bool(state_->slots[slot_].result) && state_->slots[slot_].event_recorded, code::backend_error,
        "failed task has no output dependency; retain it and drain");
    detail::cu(cudaStreamWaitEvent(stream, state_->native->slots[slot_]->completed, 0)); return {};
  } catch (...) { return detail::caught(); }
}
shared_images::shared_images(std::shared_ptr<detail::state> state, uint32_t i, uint64_t generation)
    : state_(std::move(state)), slot_(i), generation_(generation) {}
shared_images::shared_images(shared_images &&other) noexcept : state_(std::move(other.state_)),
    slot_(other.slot_), generation_(other.generation_), result(std::move(other.result)) {}
shared_images &shared_images::operator=(shared_images &&other) noexcept {
  if (this != &other) { shared_images old(std::move(*this)); state_ = std::move(other.state_);
    slot_ = other.slot_; generation_ = other.generation_; result = std::move(other.result); }
  return *this;
}
shared_images::~shared_images() { if (state_) state_->abandon(slot_, generation_, true); }
shared_images::operator bool() const noexcept { return bool(state_); }
images shared_images::views() const {
  require(bool(state_), code::invalid_argument, "shared images are empty or released");
  auto lock = detail::lock(*state_); state_->current(); return state_->views(slot_);
}
shared_images adapter::acquire() {
  try {
    require(bool(state_), code::closed, "adapter is empty");
    auto lock = detail::lock(*state_); state_->current();
    auto i = state_->reserve(true); return shared_images(state_, i, state_->slots[i].generation);
  } catch (...) { shared_images result; result.result = detail::caught(); return result; }
}
task adapter::enqueue_copy(const images &views, cudaStream_t stream, float scale) {
  task accepted;
  try {
    require(bool(state_), code::closed, "adapter is empty");
    auto lock = detail::lock(*state_); state_->stream(stream);
    require(std::isfinite(scale) && scale > 0, code::invalid_argument, "input scale must be positive finite");
    for (auto view : {&views.color, &views.albedo, &views.normal, &views.output}) state_->validate(*view);
    auto start = [](const image &v) { return reinterpret_cast<uintptr_t>(v.data) + v.offset; };
    auto end = [&](const image &v) { return start(v) + v.row_stride * (v.height - 1) + uint64_t(v.width) * 16; };
    for (auto input : {&views.color, &views.albedo, &views.normal})
      require(end(*input) <= start(views.output) || end(views.output) <= start(*input),
          code::invalid_argument, "output overlaps an input");
    auto i = state_->reserve(false);
    accepted = task(state_, i, state_->slots[i].generation, {});
    state_->enqueue(i, &views, stream, scale);
    accepted.result = state_->slots[i].result;
    return accepted;
  } catch (...) { accepted.result = detail::caught(); return accepted; }
}
task adapter::enqueue_shared(shared_images &images, cudaStream_t stream, float scale) {
  task accepted;
  try {
    require(state_ && images.state_ == state_, code::invalid_argument, "shared images belong to another adapter");
    auto lock = detail::lock(*state_); state_->stream(stream);
    require(std::isfinite(scale) && scale > 0, code::invalid_argument, "input scale must be positive finite");
    auto &slot = state_->slots[images.slot_];
    require(slot.leased && !slot.enqueued && slot.generation == images.generation_,
        code::invalid_argument, "shared images have already been submitted");
    accepted = task(state_, images.slot_, images.generation_, {});
    state_->enqueue(images.slot_, nullptr, stream, scale);
    accepted.result = slot.result;
    return accepted;
  } catch (...) { accepted.result = detail::caught(); return accepted; }
}
status adapter::release(shared_images &images, cudaStream_t stream) {
  try {
    require(state_ && images.state_ == state_, code::invalid_argument, "shared images belong to another adapter");
    auto lock = detail::lock(*state_); state_->stream(stream);
    auto &slot = state_->slots[images.slot_];
    require(slot.leased && slot.generation == images.generation_, code::invalid_argument, "stale lease");
    if (slot.enqueued) {
      if (!slot.result) {
        require(state_->complete(images.slot_), code::busy, "failed shared task is still in flight; retain its lease");
      } else {
        require(slot.event_recorded, code::backend_error, "task has no completion event");
        detail::cu(cudaStreamWaitEvent(stream, state_->native->slots[images.slot_]->completed, 0));
      }
    }
    detail::cu(cudaEventRecord(state_->native->slots[images.slot_]->retired, stream));
    slot.retired_recorded = true; slot.leased = false; images.state_.reset(); return {};
  } catch (...) { return detail::caught(); }
}
status adapter::resize(uint32_t width, uint32_t height) {
  try {
    require(bool(state_), code::closed, "adapter is empty");
    auto lock = detail::lock(*state_); state_->current();
    require(width && height && width <= 16384 && height <= 16384,
        code::invalid_argument, "invalid image dimensions");
    require(!state_->poisoned, code::backend_error, "adapter stopped after an enqueue failure");
    for (uint32_t i = 0; i < state_->slots.size(); ++i) {
      auto &slot = state_->slots[i];
      require(!slot.leased && !slot.task_held && (!slot.allocated || state_->complete(i, true)),
          code::busy, "release tasks/leases and finish consumer retirement before resize");
    }
    require(!state_->poisoned, code::backend_error, "adapter stopped after a completion failure");
    if (width != state_->opts.width || height != state_->opts.height) state_->rebuild_images(width, height);
    return {};
  } catch (...) { return detail::caught(); }
}
status adapter::drain(uint64_t timeout) {
  try {
    require(bool(state_), code::closed, "adapter is empty");
    auto lock = detail::lock(*state_); state_->current(); return state_->drain(timeout, false);
  } catch (...) { return detail::caught(); }
}
status adapter::close(uint64_t timeout) {
  try {
    if (!state_) return {};
    auto lock = detail::lock(*state_);
    if (state_->closed) return {};
    state_->current(); return state_->drain(timeout, true);
  } catch (...) { return detail::caught(); }
}
statistics adapter::stats() const {
  require(bool(state_), code::closed, "adapter is empty");
  auto lock = detail::lock(*state_); return state_->counters;
}
runtime_info adapter::info() const {
  require(bool(state_), code::closed, "adapter is empty");
  auto lock = detail::lock(*state_);
  char path[32768]{}, name[256]{};
  require(GetModuleFileNameA(GetModuleHandleA("oidn_webgpu.dll"), path, sizeof(path)) > 0,
      code::backend_error, "loaded core DLL path unavailable");
  detail::driver(cuDeviceGetName(name, sizeof(name), state_->native->device));
  return {name, path, oidnw_api_version(), state_->opts.agility_sdk_version, _MSC_FULL_VER, state_->workspace.info()};
}
bool adapter::uses_fp16() const {
  require(bool(state_), code::closed, "adapter is empty");
  auto lock = detail::lock(*state_); return state_->fp16;
}
}
