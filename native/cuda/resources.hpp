#pragma once
struct SharedBuffer {
  ComPtr<ID3D12Resource> resource;
  cudaExternalMemory_t memory = nullptr;
  void *mapped = nullptr;
  uint64_t allocation_bytes = 0, logical_bytes = 0;
  void create(ID3D12Device *device, uint64_t image_bytes, bool shared = true) {
    logical_bytes = 256 + image_bytes + 256;
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = logical_bytes;
    desc.Height = 1;
    desc.DepthOrArraySize = desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    hr(device->CreateCommittedResource(&heap, shared ? D3D12_HEAP_FLAG_SHARED : D3D12_HEAP_FLAG_NONE,
        &desc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&resource)));
    if (!shared) return;
    allocation_bytes = device->GetResourceAllocationInfo(0, 1, &desc).SizeInBytes;
    require(allocation_bytes != UINT64_MAX && allocation_bytes >= logical_bytes,
        code::backend_error, "bad allocation size");
    Handle handle;
    hr(device->CreateSharedHandle(resource.Get(), nullptr, GENERIC_ALL, nullptr, &handle.value));
    cudaExternalMemoryHandleDesc external{};
    external.type = cudaExternalMemoryHandleTypeD3D12Resource;
    external.handle.win32.handle = handle.value;
    external.size = allocation_bytes;
    external.flags = cudaExternalMemoryDedicated;
    cu(cudaImportExternalMemory(&memory, &external));
    cudaExternalMemoryBufferDesc map{};
    map.size = logical_bytes;
    cu(cudaExternalMemoryGetMappedBuffer(&mapped, memory, &map));
  }
  float4 *pixels() const { return reinterpret_cast<float4 *>(static_cast<char *>(mapped) + 256); }
  ~SharedBuffer() {
    if (mapped) cleanup(cudaFree(mapped));
    if (memory) cleanup(cudaDestroyExternalMemory(memory));
  }
};
struct SharedFence {
  ComPtr<ID3D12Fence> fence;
  cudaExternalSemaphore_t semaphore = nullptr;
  void create(ID3D12Device *device) {
    hr(device->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&fence)));
    Handle handle;
    hr(device->CreateSharedHandle(fence.Get(), nullptr, GENERIC_ALL, nullptr, &handle.value));
    cudaExternalSemaphoreHandleDesc desc{};
    desc.type = cudaExternalSemaphoreHandleTypeD3D12Fence;
    desc.handle.win32.handle = handle.value;
    cu(cudaImportExternalSemaphore(&semaphore, &desc));
  }
  void signal(cudaStream_t stream, uint64_t value) {
    cudaExternalSemaphoreSignalParams params{};
    params.params.fence.value = value;
    cu(cudaSignalExternalSemaphoresAsync(&semaphore, &params, 1, stream));
  }
  void wait(cudaStream_t stream, uint64_t value) {
    cudaExternalSemaphoreWaitParams params{};
    params.params.fence.value = value;
    cu(cudaWaitExternalSemaphoresAsync(&semaphore, &params, 1, stream));
  }
  ~SharedFence() { if (semaphore) cleanup(cudaDestroyExternalSemaphore(semaphore)); }
};
