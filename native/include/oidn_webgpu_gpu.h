#ifndef OIDN_WEBGPU_GPU_H
#define OIDN_WEBGPU_GPU_H
#include "oidn_webgpu.h"
#ifdef __cplusplus
extern "C" {
#endif

/* GPU images stay in host allocations. Metal, Vulkan and DX12 imports are implemented. Opaque handle pointers must name
 * live objects from this library. Do not destroy a handle concurrently with a
 * call using it. Models may be shared; other same-handle CPU calls are exclusive.
 * Different handles on a device serialize queue operations internally.
 * Descriptions are required and initialized with their *_default function.
 * next must be NULL; see oidn_webgpu.h for version compatibility.
 * prepared_get_info may overlap submit on the same workspace.
 * No background threads are created. Queue lock callbacks run on the caller. */
typedef struct oidnw_gpu_device oidnw_gpu_device;
typedef struct oidnw_gpu_model oidnw_gpu_model;
typedef struct oidnw_gpu_filter oidnw_gpu_filter;
typedef struct oidnw_gpu_prepared oidnw_gpu_prepared;
typedef struct oidnw_gpu_buffer oidnw_gpu_buffer;
typedef struct oidnw_gpu_sync oidnw_gpu_sync;
typedef void (*oidnw_gpu_queue_lock_fn)(void *user_data);

typedef struct oidnw_gpu_filter_desc {
  const void *next; /* NULL; non-NULL extensions are rejected. */
  uint32_t hdr;
  uint32_t transfer;
  uint32_t quality; /* oidnw_quality, default HIGH; FAST is explicit opt-in. */
  uint32_t kernel; /* oidnw_kernel, default AUTO. */
  uint32_t max_tile_size;
  uint32_t flags; /* OIDNW_FILTER_RUNTIME_CHECKS, enabled by default. */
  uint32_t binding_cache_capacity; /* 1..64, default 8 sets per workspace. */
} oidnw_gpu_filter_desc;

typedef struct oidnw_gpu_image {
  oidnw_gpu_buffer *buffer;
  uint64_t offset;
} oidnw_gpu_image;

typedef struct oidnw_gpu_images {
  oidnw_gpu_image color;
  oidnw_gpu_image albedo;
  oidnw_gpu_image normal;
  oidnw_gpu_image output;
} oidnw_gpu_images;

typedef struct oidnw_gpu_timeline_point {
  oidnw_gpu_sync *sync;
  uint64_t value; /* 0..UINT64_MAX-1; UINT64_MAX is reserved. */
} oidnw_gpu_timeline_point;

typedef struct oidnw_gpu_submit_desc {
  const void *next; /* NULL; non-NULL extensions are rejected. */
  float input_scale; /* Positive finite, default 1. */
  oidnw_gpu_images images;
  const oidnw_gpu_timeline_point *waits;
  size_t wait_count; /* 0..64. */
  const oidnw_gpu_timeline_point *signals;
  size_t signal_count; /* 0..64. Values must increase for each sync. */
} oidnw_gpu_submit_desc;

typedef struct oidnw_gpu_limits {
  const void *next; /* Initialize to NULL; non-NULL extensions are rejected. */
  uint32_t image_offset_alignment;
  uint64_t max_buffer_size;
  uint64_t max_image_bytes; /* Min of binding, buffer and planner limits. */
  uint32_t max_pending_submissions;
} oidnw_gpu_limits;

typedef enum oidnw_precision {
  OIDNW_PRECISION_FP32 = 1,
  OIDNW_PRECISION_FP16 = 2
} oidnw_precision;
#define OIDNW_GPU_KERNEL_PATH_DIRECT 1u
#define OIDNW_GPU_KERNEL_PATH_IMPLICIT_GEMM 2u
#define OIDNW_GPU_KERNEL_PATH_SUBGROUP_MATRIX 4u
#define OIDNW_GPU_KERNEL_PATH_SPATIAL 8u
#define OIDNW_GPU_KERNEL_PATH_SUBGROUP 16u

typedef struct oidnw_gpu_prepared_info {
  const void *next; /* Initialize to NULL; non-NULL extensions are rejected. */
  uint32_t precision; /* oidnw_precision; host images remain RGBA32F. */
  uint32_t kernels; /* OIDNW_GPU_KERNEL_PATH_* bits actually used. */
  uint32_t tile_count;
  uint64_t workspace_bytes; /* Excludes host images, weights, transient uploads and driver overhead. */
  uint64_t shared_weight_bytes; /* Count once per Model/device. */
  uint32_t runtime_checks; /* Nonzero when shader runtime checks are enabled. */
} oidnw_gpu_prepared_info;

void oidnw_gpu_filter_desc_default(oidnw_gpu_filter_desc *desc);
void oidnw_gpu_submit_desc_default(oidnw_gpu_submit_desc *desc);
int32_t oidnw_gpu_model_create(const void *weights, size_t weights_size,
  oidnw_gpu_model **model, char *error, size_t capacity);
uint32_t oidnw_gpu_model_input_count(const oidnw_gpu_model *model);
void oidnw_gpu_model_destroy(oidnw_gpu_model *model);
int32_t oidnw_gpu_filter_create(oidnw_gpu_device *device, const oidnw_gpu_model *model,
  const oidnw_gpu_filter_desc *desc, oidnw_gpu_filter **filter, char *error, size_t capacity);
void oidnw_gpu_filter_destroy(oidnw_gpu_filter *filter);
/* Compiles pipelines without image buffers, bindings or workspace allocation.
 * May block for planning and compilation. prepare on this filter with the same
 * size reuses pipelines but still plans and allocates the workspace.
 * Concurrent warm_up/prepare on this filter returns BUSY. */
int32_t oidnw_gpu_filter_warm_up(oidnw_gpu_filter *filter, uint32_t width, uint32_t height,
  char *error, size_t capacity);
/* First bindings are required. Full-sized RGBA32F images remain GPU resident;
 * only the intermediate workspace is tiled. Output uses another buffer.
 * A buffer's input/output role is fixed on first successful preparation/submit.
 * Use {NULL,0} for auxiliary images the model does not read. */
int32_t oidnw_gpu_prepare(oidnw_gpu_filter *filter, uint32_t width, uint32_t height,
  const oidnw_gpu_images *images, oidnw_gpu_prepared **prepared, char *error, size_t capacity);
/* Information is immutable after prepare. This query may run concurrently with
 * submit or other queries on the same workspace; each query needs its own output.
 * The workspace must remain alive throughout every call. Initialize info to {0}.
 * No GPU work is submitted or waited for. */
int32_t oidnw_gpu_prepared_get_info(const oidnw_gpu_prepared *prepared,
  oidnw_gpu_prepared_info *info, char *error, size_t capacity);
void oidnw_gpu_prepared_destroy(oidnw_gpu_prepared *prepared);
/* sequence=0: rejected; no waits/signals accepted. Nonzero: accepted, even on
 * error. An encoding failure submits a wait/signal batch without denoising and quarantines
 * this device (poll/wait/destroy remain available). An uncertain submission
 * returns DEVICE_LOST; its external signals may never fire. No image transfers
 * or GPU idle waits are performed. QUEUE_FULL rejects before synchronization.
 * Multiple executions of one workspace may be in flight. Native submission
 * may block; producers must be able to signal waits independently
 * of submit returning on this CPU thread. */
int32_t oidnw_gpu_submit(oidnw_gpu_prepared *prepared, const oidnw_gpu_submit_desc *desc,
  uint64_t *sequence, char *error, size_t capacity);
/* Completion includes all external signal values, not just wgpu's fence.
 * poll does not wait for GPU completion; serialized CPU queue access may block.
 * On a healthy device, unknown sequences (including zero) return
 * INVALID_ARGUMENT with completed=0.
 * completed=0 is not an error. wait uses nanoseconds;
 * timeout=0 performs one poll, UINT64_MAX waits without a deadline.
 * Completion permits input reuse, but output reuse also needs host-consumer
 * completion. CPU handle release never implies GPU completion. */
int32_t oidnw_gpu_poll(oidnw_gpu_device *device, uint64_t sequence,
  uint32_t *completed, char *error, size_t capacity);
int32_t oidnw_gpu_wait(oidnw_gpu_device *device, uint64_t sequence, uint64_t timeout_ns,
  uint32_t *completed, char *error, size_t capacity);
int32_t oidnw_gpu_device_info(const oidnw_gpu_device *device, char *name, size_t name_capacity,
  uint32_t *backend, uint32_t *features);
/* Initialize limits to {0} before querying. */
int32_t oidnw_gpu_get_limits(const oidnw_gpu_device *device, oidnw_gpu_limits *limits,
  char *error, size_t capacity);
/* Release all device-owned public handles first. BUSY leaves device alive if
 * children or submissions remain; poll/wait before retrying. Success releases
 * queue references synchronously. Native queue teardown can wait for host work
 * on a shared queue. Destroy NULL succeeds. */
int32_t oidnw_gpu_device_destroy(oidnw_gpu_device *device, char *error, size_t capacity);
/* Recovery only after DEVICE_LOST/uncertain submission, with no public children.
 * Host MUST first establish that no GPU work can access any imported allocation
 * or library workspace, including on shared queues. This releases retained
 * uncertain submissions; it does not cancel GPU work or prove quiescence.
 * A wgpu error, timeout or CPU-advanced event alone is NOT such proof. Calling
 * this while work can still run violates the resource lifetime contract. */
int32_t oidnw_gpu_device_destroy_quiescent(oidnw_gpu_device *device, char *error, size_t capacity);
void oidnw_gpu_buffer_destroy(oidnw_gpu_buffer *buffer);
void oidnw_gpu_sync_destroy(oidnw_gpu_sync *sync);

#ifdef __cplusplus
}
#endif
#endif
