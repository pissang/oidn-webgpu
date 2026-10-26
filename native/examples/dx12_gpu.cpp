// Native DX12 DLL consumer. Both devices use the same Agility SDK and LUID.
// Images stay in shared committed UAV buffers; the CPU maps integer flags only.
#define NOMINMAX
#define INITGUID
#include "oidn_webgpu_dx12.hpp"
#include "gpu_cpp_lifecycle.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <d3d12sdklayers.h>
#include <d3dcompiler.h>
#include <deque>
#include <dxgi1_4.h>
#include <fstream>
#include <initguid.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <wrl/client.h>
using Microsoft::WRL::ComPtr;
static char error_text[4096];
#define ERR error_text, sizeof error_text
static void check(int status) {
  if (status)
    throw std::runtime_error("C API " + std::to_string(status) + ": " + error_text);
}
static void hr(HRESULT result) {
  if (FAILED(result))
    throw std::runtime_error("HRESULT " + std::to_string(uint32_t(result)));
}
#define API(call) check(call)
static void trace(const std::string &text) {
  if (getenv("OIDNW_INTEROP_TRACE"))
    std::cerr << text << std::endl;
}
static std::vector<char> read_file(const std::string &path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input)
    throw std::runtime_error("cannot open " + path);
  std::vector<char> bytes(size_t(input.tellg()));
  input.seekg(0);
  input.read(bytes.data(), bytes.size());
  return bytes;
}
static uint64_t value(ID3D12Fence *fence) {
  auto v = fence->GetCompletedValue();
  if (v == UINT64_MAX)
    throw std::runtime_error("device removed; fence UINT64_MAX");
  return v;
}
static void wait(ID3D12Fence *fence, uint64_t v) {
  if (value(fence) >= v)
    return;
  HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
  if (!event)
    throw std::runtime_error("CreateEvent failed");
  auto result = fence->SetEventOnCompletion(v, event);
  if (FAILED(result)) {
    CloseHandle(event);
    hr(result);
  }
  auto status = WaitForSingleObject(event, 60000);
  CloseHandle(event);
  if (status != WAIT_OBJECT_0)
    throw std::runtime_error("host fence timeout");
  if (value(fence) < v)
    throw std::runtime_error("event did not prove fence completion");
}
struct Buffer {
  ComPtr<ID3D12Resource> raw;
  oidnw_gpu_buffer *imported = nullptr;
};
static Buffer buffer(ID3D12Device *device, uint64_t bytes, bool shared, bool readback,
                     oidnw_gpu_device *library) {
  D3D12_HEAP_PROPERTIES heap{};
  heap.Type = readback ? D3D12_HEAP_TYPE_READBACK : D3D12_HEAP_TYPE_DEFAULT;
  D3D12_RESOURCE_DESC desc{};
  desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  desc.Width = bytes;
  desc.Height = 1;
  desc.DepthOrArraySize = 1;
  desc.MipLevels = 1;
  desc.SampleDesc.Count = 1;
  desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  desc.Flags = readback ? D3D12_RESOURCE_FLAG_NONE : D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
  Buffer result;
  hr(device->CreateCommittedResource(
      &heap, shared ? D3D12_HEAP_FLAG_SHARED : D3D12_HEAP_FLAG_NONE, &desc,
      readback ? D3D12_RESOURCE_STATE_COPY_DEST : D3D12_RESOURCE_STATE_COMMON, nullptr,
      IID_PPV_ARGS(&result.raw)));
  if (library)
    API(oidnw_gpu_buffer_import_dx12(library, result.raw.Get(), &result.imported, ERR));
  return result;
}
static void uav(ID3D12GraphicsCommandList *list, ID3D12Resource *resource) {
  D3D12_RESOURCE_BARRIER b{};
  b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
  b.UAV.pResource = resource;
  list->ResourceBarrier(1, &b);
}
struct Batch {
  ComPtr<ID3D12CommandAllocator> allocator;
  ComPtr<ID3D12GraphicsCommandList> list;
};
struct Kernels {
  ComPtr<ID3D12RootSignature> root;
  std::array<ComPtr<ID3D12PipelineState>, 3> pipelines;
  void init(ID3D12Device *device, const std::string &directory) {
    D3D12_ROOT_PARAMETER params[4]{};
    for (unsigned i = 0; i < 3; ++i) {
      params[i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
      params[i].Descriptor.ShaderRegister = i;
    }
    params[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[3].Constants.Num32BitValues = 4;
    D3D12_ROOT_SIGNATURE_DESC desc{};
    desc.NumParameters = 4;
    desc.pParameters = params;
    ComPtr<ID3DBlob> code, errors;
    hr(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &code, &errors));
    hr(device->CreateRootSignature(0, code->GetBufferPointer(), code->GetBufferSize(),
                                   IID_PPV_ARGS(&root)));
    auto source = read_file(directory + "/host.hlsl");
    const char *entries[] = {"produce", "consume", "initialize"};
    for (unsigned i = 0; i < 3; ++i) {
      code.Reset();
      errors.Reset();
      auto result = D3DCompile(source.data(), source.size(), nullptr, nullptr, nullptr, entries[i],
                               "cs_5_1", D3DCOMPILE_ENABLE_STRICTNESS, 0, &code, &errors);
      if (FAILED(result) && errors)
        std::cerr << static_cast<const char *>(errors->GetBufferPointer());
      hr(result);
      D3D12_COMPUTE_PIPELINE_STATE_DESC pipeline{};
      pipeline.pRootSignature = root.Get();
      pipeline.CS = {code->GetBufferPointer(), code->GetBufferSize()};
      hr(device->CreateComputePipelineState(&pipeline, IID_PPV_ARGS(&pipelines[i])));
    }
  }
  Batch batch(ID3D12Device *device, unsigned kind, std::array<ID3D12Resource *, 3> buffers,
              std::array<unsigned, 4> params) {
    Batch result;
    hr(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                      IID_PPV_ARGS(&result.allocator)));
    hr(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, result.allocator.Get(),
                                 pipelines[kind].Get(), IID_PPV_ARGS(&result.list)));
    auto list = result.list.Get();
    list->SetComputeRootSignature(root.Get());
    for (unsigned i = 0; i < 3; ++i)
      list->SetComputeRootUnorderedAccessView(i, buffers[i]->GetGPUVirtualAddress() +
                                                     (i == 2 && kind != 0 ? 0 : 256));
    list->SetComputeRoot32BitConstants(3, 4, params.data(), 0);
    if (kind == 2) {
      unsigned groups = (params[0] + 63) / 64;
      list->Dispatch(std::min(groups, 65535u), (groups + 65534) / 65535, 1);
      uav(list, buffers[2]);
    } else {
      list->Dispatch((params[0] * params[0] + 63) / 64, 1, 1);
      if (kind == 1)
        uav(list, buffers[2]);
      else
        for (auto b : buffers)
          uav(list, b);
    }
    hr(list->Close());
    return result;
  }
};
static void submit(ID3D12CommandQueue *queue, const Batch &batch) {
  ID3D12CommandList *list = batch.list.Get();
  queue->ExecuteCommandLists(1, &list);
}
static oidnw_gpu_images images(const std::array<Buffer, 4> &group) {
  return {{group[0].imported, 256},
          {group[1].imported, 256},
          {group[2].imported, 256},
          {group[3].imported, 256}};
}
struct CppHost {
  oidnw_gpu_dx12_device_desc desc;
  ComPtr<ID3D12Device> host;
  ComPtr<ID3D12CommandQueue> producer, consumer;
  std::array<Buffer, 4> buffers;
  ComPtr<ID3D12Fence> ready, done, gate, verified;
  Kernels &kernels;
  Buffer reference, flags, readback;
  uint32_t size;
  std::vector<Batch> batches;
  CppHost(oidnw_gpu_dx12_device_desc description, ID3D12Device *device,
          ID3D12CommandQueue *p, ID3D12CommandQueue *c, Kernels &k,
          const Buffer &r, const Buffer &f, const Buffer &read, uint32_t edge)
      : desc(description), host(device), producer(p), consumer(c), kernels(k),
        reference(r), flags(f), readback(read), size(edge) {
    for (auto &b : buffers) b = buffer(host.Get(), 256 + uint64_t(size) * size * 16,
                                      true, false, nullptr);
    for (auto fence : {&ready, &done, &gate, &verified})
      hr(host->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(fence)));
  }
  oidnw::gpu::device context(oidnw::gpu::device_options options) {
    return oidnw::dx12::create_device(desc, std::move(options));
  }
  oidnw::gpu::buffer image(const oidnw::gpu::device &context, unsigned i) {
    return oidnw::dx12::import_buffer(context, buffers[i].raw.Get());
  }
  oidnw::gpu::sync sync(const oidnw::gpu::device &context, bool output) {
    return oidnw::dx12::import_sync(context, output ? done.Get() : ready.Get());
  }
  void start() {
    batches.push_back(kernels.batch(host.Get(), 0,
        {buffers[0].raw.Get(), buffers[1].raw.Get(), buffers[2].raw.Get()}, {size, 2, 0, 0}));
    hr(producer->Wait(gate.Get(), 1));
    submit(producer.Get(), batches.back());
    hr(producer->Signal(ready.Get(), 1));
  }
  void unblock() { hr(gate->Signal(1)); }
  void verify() {
    batches.push_back(kernels.batch(host.Get(), 1,
        {buffers[3].raw.Get(), reference.raw.Get(), flags.raw.Get()}, {size, 2, 0, 0}));
    hr(consumer->Wait(done.Get(), 1));
    submit(consumer.Get(), batches.back());
    Batch copy;
    hr(host->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&copy.allocator)));
    hr(host->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, copy.allocator.Get(), nullptr,
                               IID_PPV_ARGS(&copy.list)));
    auto bytes = size_t(size) * size * 16;
    copy.list->CopyBufferRegion(readback.raw.Get(), 0, flags.raw.Get(), 0, bytes);
    hr(copy.list->Close());
    batches.push_back(copy);
    submit(consumer.Get(), copy);
    hr(consumer->Signal(verified.Get(), 1));
    wait(verified.Get(), 1);
    void *mapped;
    D3D12_RANGE range{0, bytes};
    hr(readback.raw->Map(0, &range, &mapped));
    auto words = static_cast<const uint32_t *>(mapped);
    bool valid = std::all_of(words, words + bytes / 4, [](uint32_t word) { return word == 0; });
    D3D12_RANGE empty{};
    readback.raw->Unmap(0, &empty);
    cpp_lifecycle::ensure(valid, "C++ DX12 output mismatch");
  }
};
int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "usage: dx12_gpu WEIGHTS SHADER_DIR [frames=1000] [size=16] [tile=0] [fp32=0] "
                 "[unused=0] [checks=1]\n";
    return 2;
  }
  try {
    if (!OIDNW_API_VERSION_IS_COMPATIBLE(oidnw_api_version()))
      throw std::runtime_error("header/DLL version mismatch");
    unsigned frames = argc > 3 ? std::stoul(argv[3]) : 1000,
             size = argc > 4 ? std::stoul(argv[4]) : 16, tile = argc > 5 ? std::stoul(argv[5]) : 0;
    bool fp32 = argc > 6 && std::stoi(argv[6]), checks = argc <= 8 || std::stoi(argv[8]);
    if (!frames || frames > 100000 || !size || size > 1024 ||
        uint64_t(frames) * size * size * 16 > 256 * 1024 * 1024)
      throw std::runtime_error("invalid frames/size");
    const char *sdk_path = getenv("WGPU_DX12_AGILITY_SDK_PATH"),
               *sdk_version = getenv("WGPU_DX12_AGILITY_SDK_VERSION");
    if (!sdk_path || !sdk_version)
      throw std::runtime_error("set matching absolute WGPU_DX12_AGILITY_SDK_PATH and VERSION");
    bool debug = !getenv("OIDNW_INTEROP_NO_DEBUG"),
         gpu_validation = getenv("WGPU_GPU_BASED_VALIDATION") &&
                          !strcmp(getenv("WGPU_GPU_BASED_VALIDATION"), "1");
    HMODULE layers = nullptr;
    if (debug) {
      layers = LoadLibraryA((std::string(sdk_path) + "d3d12SDKLayers.dll").c_str());
      if (!layers)
        throw std::runtime_error("Agility SDK layers missing");
    }
    ComPtr<ID3D12SDKConfiguration1> config;
    hr(D3D12GetInterface(CLSID_D3D12SDKConfiguration, IID_PPV_ARGS(&config)));
    ComPtr<ID3D12DeviceFactory> factory;
    hr(config->CreateDeviceFactory(std::stoul(sdk_version), sdk_path, IID_PPV_ARGS(&factory)));
    ComPtr<ID3D12Debug> debug_interface;
    if (debug) {
      hr(factory->GetConfigurationInterface(CLSID_D3D12Debug, IID_PPV_ARGS(&debug_interface)));
      debug_interface->EnableDebugLayer();
    }
    if (gpu_validation && !debug)
      throw std::runtime_error("GPU validation requires debug layer");
    if (gpu_validation) {
      ComPtr<ID3D12Debug1> dbg;
      hr(debug_interface.As(&dbg));
      dbg->SetEnableGPUBasedValidation(TRUE);
    }
    const char *requested_adapter = getenv("OIDNW_INTEROP_ADAPTER");
    std::string match = requested_adapter ? requested_adapter : "RTX 4060";
    int length = MultiByteToWideChar(CP_UTF8, 0, match.c_str(), -1, nullptr, 0);
    std::wstring adapter_match(length, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, match.c_str(), -1, adapter_match.data(), length);
    adapter_match.resize(length - 1);
    ComPtr<IDXGIFactory4> dxgi;
    hr(CreateDXGIFactory2(0, IID_PPV_ARGS(&dxgi)));
    ComPtr<IDXGIAdapter1> adapter;
    DXGI_ADAPTER_DESC1 adapter_desc{};
    for (unsigned i = 0;; ++i) {
      hr(dxgi->EnumAdapters1(i, &adapter));
      hr(adapter->GetDesc1(&adapter_desc));
      if (std::wstring(adapter_desc.Description).find(adapter_match) != std::wstring::npos)
        break;
      adapter.Reset();
    }
    ComPtr<ID3D12Device> host;
    hr(factory->CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&host)));
    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    ComPtr<ID3D12CommandQueue> producer, consumer;
    hr(host->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&producer)));
    hr(host->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&consumer)));
    oidnw_gpu_dx12_device_desc desc;
    oidnw_gpu_dx12_device_desc_default(&desc);
    desc.host_device = host.Get();
    desc.agility_sdk_path = sdk_path;
    desc.agility_sdk_version = std::stoul(sdk_version);
    desc.flags = (fp32 ? 1 : 0) | (debug ? 4 : 0) | (gpu_validation ? 8 : 0);
    desc.max_pending_submissions = 8;
    trace("create independent library device");
    oidnw_gpu_device *library;
    API(oidnw_gpu_device_create_dx12(&desc, &library, ERR));
    uint32_t backend, enabled;
    API(oidnw_gpu_device_info(library, nullptr, 0, &backend, &enabled));
    if (backend != OIDNW_BACKEND_DX12)
      throw std::runtime_error("wrong backend");
    auto mismatched = desc;
    ++mismatched.agility_sdk_version;
    oidnw_gpu_device *rejected = nullptr;
    if (oidnw_gpu_device_create_dx12(&mismatched, &rejected, ERR) != OIDNW_ERROR_INVALID_ARGUMENT ||
        rejected)
      throw std::runtime_error("SDK mismatch accepted");
    uint64_t bytes = uint64_t(size) * size * 16;
    std::array<std::array<Buffer, 4>, 3> ring;
    std::array<Buffer, 3> references;
    std::vector<Buffer *> allocations;
    for (auto &group : ring)
      for (auto &b : group) {
        b = buffer(host.Get(), 256 + bytes, true, false, library);
        allocations.push_back(&b);
      }
    for (auto &b : references) {
      b = buffer(host.Get(), 256 + bytes, true, false, library);
      allocations.push_back(&b);
    }
    oidnw_gpu_buffer *duplicate = nullptr;
    if (oidnw_gpu_buffer_import_dx12(library, ring[0][0].raw.Get(), &duplicate, ERR) !=
            OIDNW_ERROR_INVALID_ARGUMENT ||
        duplicate)
      throw std::runtime_error("duplicate resource accepted");
    auto invalid = buffer(host.Get(), 256 + bytes, false, false, nullptr);
    if (oidnw_gpu_buffer_import_dx12(library, invalid.raw.Get(), &duplicate, ERR) == 0 || duplicate)
      throw std::runtime_error("non-shared buffer accepted");
    auto flags = buffer(host.Get(), frames * bytes, false, false, nullptr),
         readback = buffer(host.Get(), frames * bytes, false, true, nullptr);
    ComPtr<ID3D12Fence> ready, done, consumed, gate, final_gate;
    for (auto p : {&ready, &done, &consumed, &gate, &final_gate})
      hr(host->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(p)));
    oidnw_gpu_sync *ready_handle, *done_handle;
    API(oidnw_gpu_sync_import_dx12(library, ready.Get(), &ready_handle, ERR));
    API(oidnw_gpu_sync_import_dx12(library, done.Get(), &done_handle, ERR));
    Kernels kernels;
    kernels.init(host.Get(), argv[2]);
    auto initialize =
        kernels.batch(host.Get(), 2, {ring[0][0].raw.Get(), ring[0][1].raw.Get(), flags.raw.Get()},
                      {frames * size * size, 0, 0, 0});
    submit(consumer.Get(), initialize);
    hr(consumer->Signal(consumed.Get(), 1));
    wait(consumed.Get(), 1);
    auto weights = read_file(argv[1]);
    oidnw_gpu_model *model;
    API(oidnw_gpu_model_create(weights.data(), weights.size(), &model, ERR));
    oidnw_gpu_filter_desc filter_desc;
    oidnw_gpu_filter_desc_default(&filter_desc);
    filter_desc.flags = checks ? OIDNW_FILTER_RUNTIME_CHECKS : 0;
    filter_desc.max_tile_size = tile;
    filter_desc.binding_cache_capacity = 3;
    oidnw_gpu_filter *filter;
    API(oidnw_gpu_filter_create(library, model, &filter_desc, &filter, ERR));
    oidnw_gpu_model_destroy(model);
    auto initial = images(ring[0]);
    oidnw_gpu_prepared *prepared;
    trace("prepare workspace");
    API(oidnw_gpu_prepare(filter, size, size, &initial, &prepared, ERR));
    oidnw_gpu_filter_destroy(filter);
    uint64_t last = 0;
    unsigned queue_full = 0;
    auto denoise = [&](oidnw_gpu_images bindings, uint64_t point, float scale) {
      oidnw_gpu_submit_desc submit;
      oidnw_gpu_submit_desc_default(&submit);
      submit.images = bindings;
      submit.input_scale = scale;
      oidnw_gpu_timeline_point w{ready_handle, point}, s{done_handle, point};
      submit.waits = &w;
      submit.wait_count = 1;
      submit.signals = &s;
      submit.signal_count = 1;
      for (;;) {
        uint64_t sequence = 99;
        auto status = oidnw_gpu_submit(prepared, &submit, &sequence, ERR);
        if (status == OIDNW_ERROR_QUEUE_FULL) {
          if (sequence)
            throw std::runtime_error("QUEUE_FULL accepted work");
          ++queue_full;
          uint32_t complete;
          API(oidnw_gpu_wait(library, last, 60000000000ull, &complete, ERR));
          continue;
        }
        API(status);
        if (sequence <= last)
          throw std::runtime_error("sequence did not increase");
        last = sequence;
        break;
      }
    };
    const float scales[] = {1, 2, 0.5};
    for (unsigned pattern = 0; pattern < 3; ++pattern) {
      trace("reference " + std::to_string(pattern));
      auto batch = kernels.batch(host.Get(), 0,
                                 {ring[0][0].raw.Get(), ring[0][1].raw.Get(), ring[0][2].raw.Get()},
                                 {size, pattern, 0, 0});
      submit(producer.Get(), batch);
      hr(producer->Signal(ready.Get(), pattern + 1));
      wait(ready.Get(), pattern + 1);
      auto binding = initial;
      binding.output.buffer = references[pattern].imported;
      denoise(binding, pattern + 1, scales[pattern]);
      uint32_t complete;
      API(oidnw_gpu_wait(library, last, 60000000000ull, &complete, ERR));
      if (value(done.Get()) < pattern + 1)
        throw std::runtime_error("wait missed external signal");
    }
    if (!gpu_validation)
      cpp_lifecycle::run([&] {
        return std::make_shared<CppHost>(desc, host.Get(), producer.Get(), consumer.Get(),
                                        kernels, references[2], flags, readback, size);
      }, weights.data(), weights.size(), size, tile, checks);
    oidnw_gpu_submit_desc bad;
    oidnw_gpu_submit_desc_default(&bad);
    bad.images = initial;
    bad.input_scale = -1;
    uint64_t rejected_sequence = 99;
    if (oidnw_gpu_submit(prepared, &bad, &rejected_sequence, ERR) != OIDNW_ERROR_INVALID_ARGUMENT ||
        rejected_sequence)
      throw std::runtime_error("bad parameters accepted");
    bad.input_scale = 1;
    bad.images.output = bad.images.color;
    rejected_sequence = 99;
    if (oidnw_gpu_submit(prepared, &bad, &rejected_sequence, ERR) != OIDNW_ERROR_INVALID_ARGUMENT ||
        rejected_sequence)
      throw std::runtime_error("aliased output accepted");
    std::atomic<bool> gate_safe{false}, gate_released{false};
    std::thread release([&] {
      std::this_thread::sleep_for(std::chrono::milliseconds(250));
      gate_safe = value(done.Get()) == 3;
      gate_released = true;
      hr(gate->Signal(1));
    });
    struct Pending {
      uint64_t value;
      Batch producer, consumer;
    };
    std::deque<Pending> pending;
    size_t high_water = 0;
    for (unsigned frame = 0; frame < frames; ++frame) {
      trace("frame " + std::to_string(frame));
      if (pending.size() == 64)
        wait(consumed.Get(), pending.front().value);
      auto completed = value(consumed.Get());
      while (!pending.empty() && pending.front().value <= completed)
        pending.pop_front();
      unsigned slot = frame % 3, pattern = (frame + frame / 3) % 3;
      uint64_t n = frame + 1;
      auto &group = ring[slot];
      auto production =
          kernels.batch(host.Get(), 0, {group[0].raw.Get(), group[1].raw.Get(), group[2].raw.Get()},
                        {size, pattern, 0, 0});
      hr(producer->Wait(gate.Get(), 1));
      if (frame >= 3)
        hr(producer->Wait(consumed.Get(), n - 2));
      if (frame + 1 == frames)
        hr(producer->Wait(final_gate.Get(), 1));
      submit(producer.Get(), production);
      hr(producer->Signal(ready.Get(), n + 3));
      denoise(images(group), n + 3, scales[pattern]);
      auto consumption = kernels.batch(
          host.Get(), 1, {group[3].raw.Get(), references[pattern].raw.Get(), flags.raw.Get()},
          {size, pattern, frame, 0});
      hr(consumer->Wait(done.Get(), n + 3));
      submit(consumer.Get(), consumption);
      hr(consumer->Signal(consumed.Get(), n + 1));
      pending.push_back({n + 1, production, consumption});
      high_water = std::max(high_water, pending.size());
      if (frame == 0) {
        uint32_t complete = 99;
        auto status = oidnw_gpu_wait(library, last, 0, &complete, ERR);
        if (!gate_released && (status != OIDNW_ERROR_TIMEOUT || complete))
          throw std::runtime_error("producer gate ignored");
      }
    }
    release.join();
    if (!gate_safe)
      throw std::runtime_error("external signal preceded producer gate");
    oidnw_gpu_prepared_destroy(prepared);
    for (auto b : allocations)
      oidnw_gpu_buffer_destroy(b->imported);
    oidnw_gpu_sync_destroy(ready_handle);
    oidnw_gpu_sync_destroy(done_handle);
    // Pending C API records retain these original COM identities and allocations.
    for (auto &group : ring)
      for (unsigned i = 0; i < 3; ++i)
        group[i].raw.Reset();
    uint32_t complete = 99;
    if (oidnw_gpu_wait(library, last, 0, &complete, ERR) != OIDNW_ERROR_TIMEOUT || complete)
      throw std::runtime_error("final producer completed early");
    if (oidnw_gpu_device_destroy(library, ERR) != OIDNW_ERROR_BUSY)
      throw std::runtime_error("in-flight device destroyed");
    hr(final_gate->Signal(1));
    API(oidnw_gpu_wait(library, last, 60000000000ull, &complete, ERR));
    if (value(done.Get()) < frames + 3)
      throw std::runtime_error("external signals incomplete");
    wait(consumed.Get(), frames + 1);
    pending.clear();
    Batch copy;
    hr(host->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&copy.allocator)));
    hr(host->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, copy.allocator.Get(), nullptr,
                               IID_PPV_ARGS(&copy.list)));
    copy.list->CopyBufferRegion(readback.raw.Get(), 0, flags.raw.Get(), 0, frames * bytes);
    hr(copy.list->Close());
    submit(consumer.Get(), copy);
    hr(consumer->Signal(consumed.Get(), frames + 2));
    wait(consumed.Get(), frames + 2);
    API(oidnw_gpu_device_destroy(library, ERR));
    void *mapped;
    D3D12_RANGE range{0, size_t(frames * bytes)};
    hr(readback.raw->Map(0, &range, &mapped));
    auto words = static_cast<const uint32_t *>(mapped);
    for (uint64_t i = 0; i < frames * bytes / 4; ++i)
      if (words[i])
        throw std::runtime_error("GPU consumer mismatch word " + std::to_string(i) + " flags " +
                                 std::to_string(words[i]));
    D3D12_RANGE empty{};
    readback.raw->Unmap(0, &empty);
    // Library maintenance checks the debug queues of BOTH devices before
    // retiring work. This final host scan also catches consumer/copy errors.
    unsigned validation_errors = 0;
    ComPtr<ID3D12InfoQueue> info;
    if (debug) {
      hr(host.As(&info));
      for (uint64_t i = 0; i < info->GetNumStoredMessagesAllowedByRetrievalFilter(); ++i) {
        SIZE_T bytes = 0;
        hr(info->GetMessage(i, nullptr, &bytes));
        std::vector<size_t> data((bytes + sizeof(size_t) - 1) / sizeof(size_t));
        auto message = reinterpret_cast<D3D12_MESSAGE *>(data.data());
        hr(info->GetMessage(i, message, &bytes));
        if (message->Severity == D3D12_MESSAGE_SEVERITY_ERROR ||
            message->Severity == D3D12_MESSAGE_SEVERITY_CORRUPTION) {
          ++validation_errors;
          std::cerr << message->pDescription << std::endl;
        }
      }
    }
    if (validation_errors)
      throw std::runtime_error("DX12 debug validation errors");
    char dll[MAX_PATH];
    GetModuleFileNameA(GetModuleHandleA("oidn_webgpu.dll"), dll, MAX_PATH);
    std::replace(dll, dll + strlen(dll), '\\', '/');
    std::cout << "{\"backend\":\"DX12\",\"formal_c_api\":true,\"route\":\"independent-device-"
                 "shared-memory\",\"dll\":\""
              << dll << "\",\"cpp_lifecycle\":" << (gpu_validation ? "false" : "true")
              << ",\"frames\":" << frames << ",\"size\":" << size << ",\"tile\":" << tile
              << ",\"precision\":\"" << ((enabled & 1) ? "fp16" : "fp32")
              << "\",\"runtime_checks\":" << (checks ? "true" : "false")
              << ",\"debug_layer\":" << (debug ? "true" : "false")
              << ",\"gpu_validation\":" << (gpu_validation ? "true" : "false")
              << ",\"validation_errors\":" << (debug ? std::to_string(validation_errors) : "null")
              << ",\"output_byte_equal\":true,\"cpu_image_uploads\":0,\"cpu_image_readbacks\":0,"
                 "\"queue_full\":"
              << queue_full << ",\"host_pending_high_water\":" << high_water
              << ",\"producer_gate_verified\":true,\"in_flight_release_verified\":true}"
              << std::endl;
    // layers stays loaded through the COM objects' process-lifetime cleanup.
    (void)layers;
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
