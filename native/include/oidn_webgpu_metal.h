#ifndef OIDN_WEBGPU_METAL_H
#define OIDN_WEBGPU_METAL_H
#include "oidn_webgpu_gpu.h"
#ifdef __cplusplus
extern "C" {
#endif

#define OIDNW_GPU_METAL_FP32 1u
#define OIDNW_GPU_METAL_NO_SUBGROUPS 2u
typedef struct oidnw_gpu_metal_device_desc {
  uint32_t flags; /* Stable features only; experimental matrices not enabled. */
  void *device; /* id<MTLDevice>, borrowed, use __bridge under ARC. */
  void *queue; /* Optional id<MTLCommandQueue>; NULL creates a dedicated queue. */
  oidnw_gpu_queue_lock_fn lock;
  oidnw_gpu_queue_lock_fn unlock;
  void *user_data;
  uint32_t max_pending_submissions; /* 1..4096, default 64. */
} oidnw_gpu_metal_device_desc;

void oidnw_gpu_metal_device_desc_default(oidnw_gpu_metal_device_desc *desc);
/* Metal objects are retained; the caller may release its own references after
 * import. A shared queue must have capacity >=4096 (as in wgpu-hal 30.0.1) and
 * provide paired lock/unlock callbacks used by both the host and this library.
 * Callbacks must not reenter the library, throw, or unwind; user_data remains
 * valid through successful device destruction. Dedicated queues need none.
 * These functions return UNAVAILABLE on platforms other than macOS. */
int32_t oidnw_gpu_device_import_metal(const oidnw_gpu_metal_device_desc *desc,
  oidnw_gpu_device **device, char *error, size_t capacity);
/* Buffer length must equal the actual MTLBuffer length. Do not access image
 * memory on the CPU while GPU work uses it. Memoryless buffers and overlapping
 * heap allocations are not supported.
 * Import each object only once per device while any library reference survives.
 * Cached/current bindings may retain an allocation after its public handle is
 * released; cache entries are pruned on subsequent calls. */
int32_t oidnw_gpu_buffer_import_metal(oidnw_gpu_device *device, void *metal_buffer,
  uint64_t length, oidnw_gpu_buffer **buffer, char *error, size_t capacity);
int32_t oidnw_gpu_sync_import_metal(oidnw_gpu_device *device, void *shared_event,
  oidnw_gpu_sync **sync, char *error, size_t capacity);

#ifdef __cplusplus
}
#endif
#endif
