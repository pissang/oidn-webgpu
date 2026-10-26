#ifndef OIDN_WEBGPU_DX12_H
#define OIDN_WEBGPU_DX12_H
#include "oidn_webgpu_gpu.h"
#if defined(_WIN32)
#include <d3d12.h>
#else
typedef struct ID3D12Device ID3D12Device;
typedef struct ID3D12Resource ID3D12Resource;
typedef struct ID3D12Fence ID3D12Fence;
#endif
#ifdef __cplusplus
extern "C" {
#endif
#define OIDNW_GPU_DX12_FP32 1u
#define OIDNW_GPU_DX12_NO_SUBGROUPS 2u
#define OIDNW_GPU_DX12_DEBUG 4u
#define OIDNW_GPU_DX12_GPU_VALIDATION 8u
typedef struct oidnw_gpu_dx12_device_desc {
  const void *next; /* NULL; non-NULL extensions are rejected. */
  uint32_t flags;
  ID3D12Device *host_device;
  uint32_t agility_sdk_version;
  const char *agility_sdk_path; /* Absolute UTF-8 directory ending with / or \\. */
  uint32_t max_pending_submissions;
} oidnw_gpu_dx12_device_desc;
void oidnw_gpu_dx12_device_desc_default(oidnw_gpu_dx12_device_desc *desc);
/* Create an INDEPENDENT library device on the host device's adapter LUID.
 * This does not import the host ID3D12Device or share a command queue.
 * Both devices must use the same Agility SDK version; requested debug/GPU
 * validation must be enabled on both. A missing/mismatched SDK fails; there is no fallback.
 * The library retains the host device and opens shared resources on its own
 * device. Objects are COM-retained; the host may release its own references.
 * No platform queue callbacks are needed for the dedicated library queue.
 * Returns UNAVAILABLE on non-Windows platforms. */
int32_t oidnw_gpu_device_create_dx12(const oidnw_gpu_dx12_device_desc *desc,
  oidnw_gpu_device **device, char *error, size_t capacity);
/* Import a host-device committed DEFAULT heap buffer with HEAP_FLAG_SHARED,
 * ROW_MAJOR layout and exactly ALLOW_UNORDERED_ACCESS flags. Placed/reserved resources,
 * cross-adapter heaps, textures and overlapping allocations are unsupported.
 * The library creates an unnamed NT handle, opens it on the independent device,
 * then closes that handle on every exit. No caller handle is taken or closed.
 * Opened resources and the original COM identity survive cached/in-flight use;
 * duplicate imports of the same COM identity are rejected while retained.
 * Resources must be COMMON at the ownership handoff. Host DIRECT command lists
 * may use implicit buffer promotion/decay; explicit state transitions must bring
 * resources back to COMMON before signaling a library wait. Host fences order
 * execution and visibility; they do not perform resource-state transitions.
 * Keep each buffer's library input/output role fixed and fence host consumers
 * before overwriting output or reusing an image set. */
int32_t oidnw_gpu_buffer_import_dx12(oidnw_gpu_device *device, ID3D12Resource *resource,
  oidnw_gpu_buffer **buffer, char *error, size_t capacity);
/* Import a host-device fence created with D3D12_FENCE_FLAG_SHARED. Same handle
 * and COM ownership rules as buffer import. UINT64_MAX is reserved for device
 * removal and is rejected as a wait/signal value. Completion queries return
 * DEVICE_LOST when GetCompletedValue()==UINT64_MAX; they never report success
 * or retire uncertain resources on that sentinel. Poll/wait include every
 * external signal, even if wgpu's internal fence completed earlier. */
int32_t oidnw_gpu_sync_import_dx12(oidnw_gpu_device *device, ID3D12Fence *fence,
  oidnw_gpu_sync **sync, char *error, size_t capacity);
#ifdef __cplusplus
}
#endif
#endif
