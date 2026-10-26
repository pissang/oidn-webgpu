#ifndef OIDN_WEBGPU_H
#define OIDN_WEBGPU_H
/*
 * oidn-webgpu native: an experimental standalone Open Image Denoise `RT` filter
 * running the oidn-webgpu WGSL kernels on the platform GPU API (Metal, Vulkan or
 * Direct3D 12 through wgpu). Library: liboidn_webgpu.dylib / liboidn_webgpu.so /
 * oidn_webgpu.dll. The API and its results may change between versions.
 *
 * The filter takes and returns images in CPU memory, like Open Image
 * Denoise's host buffers. Metal, Vulkan and DX12 hosts can import GPU resources through
 * their platform GPU headers. Other non-wgpu hosts can plan the work with
 * oidn_webgpu_core.h and replay it on their own device.
 *
 * Conventions:
 * - Functions returning int32_t return OIDNW_OK or an oidnw_status. `error`
 *   is optional caller storage of `capacity` bytes that receives a
 *   NUL-terminated UTF-8 message (empty on success).
 * - Handles are not thread-safe: a filter executing on one thread makes a
 *   concurrent execute return OIDNW_ERROR_BUSY. Devices may be shared by
 *   filters on different threads.
 * - Destroy a device after its filters; destroying NULL does nothing.
 */
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum oidnw_status {
  OIDNW_OK = 0,
  OIDNW_ERROR_INVALID_ARGUMENT = 1,
  /* No suitable GPU adapter or device for the requested backend. */
  OIDNW_ERROR_UNAVAILABLE = 2,
  /* The weights are not a supported OIDN TZA model, or planning failed. */
  OIDNW_ERROR_MODEL = 3,
  /* Standalone filter: execution/readback failed; recreate the filter.
   * GPU interop: follow the quarantine/completion contract in its header. */
  OIDNW_ERROR_GPU = 4,
  OIDNW_ERROR_BUSY = 5,
  /* GPU interop device is lost or submission outcome is uncertain. */
  OIDNW_ERROR_DEVICE_LOST = 6,
  OIDNW_ERROR_TIMEOUT = 7,
  /* GPU interop submission was rejected before accepting synchronization. */
  OIDNW_ERROR_QUEUE_FULL = 8
} oidnw_status;

typedef enum oidnw_backend {
  /* Metal on Apple platforms; Vulkan, DX12 or Metal elsewhere (wgpu's order). */
  OIDNW_BACKEND_AUTO = 0,
  OIDNW_BACKEND_METAL = 1,
  /* Required for cooperative matrices (tensor cores) on Windows and Linux. */
  OIDNW_BACKEND_VULKAN = 2,
  OIDNW_BACKEND_DX12 = 3
} oidnw_backend;

/* The model is chosen by the weights: pass the `_small` TZA for OIDN's fast
 * quality. */
typedef enum oidnw_quality {
  /* f32 accumulation. */
  OIDNW_QUALITY_HIGH = 0,
  /* Currently the same as high. */
  OIDNW_QUALITY_BALANCED = 1,
  /* f16 accumulation in the subgroup-matrix kernels: 2-3% faster on an
   * Apple M4 Max, average relative error 3.5e-3 against f32 on a 1024x1024
   * HDR test image (3.3e-4 for high). */
  OIDNW_QUALITY_FAST = 2
} oidnw_quality;

typedef enum oidnw_kernel {
  /* Subgroup-matrix kernels when the device has cooperative matrices,
   * implicit GEMM otherwise. */
  OIDNW_KERNEL_AUTO = 0,
  OIDNW_KERNEL_IMPLICIT_GEMM = 1,
  OIDNW_KERNEL_DIRECT = 2,
  OIDNW_KERNEL_SUBGROUP_MATRIX = 3
} oidnw_kernel;

typedef enum oidnw_transfer {
  OIDNW_TRANSFER_PU = 0,
  /* For RTLightmap HDR models. */
  OIDNW_TRANSFER_LOG = 1
} oidnw_transfer;

/* Compile shaders with bounds checks; performance depends on the GPU/kernel.
 * Without it the generated kernels run unchecked: they are written to stay in
 * bounds, but a kernel bug can then touch arbitrary GPU memory. */
#define OIDNW_FILTER_RUNTIME_CHECKS 1u

#define OIDNW_FEATURE_SHADER_F16 1u
#define OIDNW_FEATURE_SUBGROUPS 2u
#define OIDNW_FEATURE_COOPERATIVE_MATRIX 4u

typedef struct oidnw_filter_desc {
  const void *next; /* NULL; non-NULL extensions are rejected. */
  /* Nonzero for HDR color (default). */
  uint32_t hdr;
  /* oidnw_transfer, used when hdr is nonzero (default PU). */
  uint32_t transfer;
  /* oidnw_quality (default high). */
  uint32_t quality;
  /* oidnw_kernel (default auto). */
  uint32_t kernel;
  /* Maximum output tile edge in pixels, rounded down to a multiple of 16;
   * 0 (default) denoises the whole image as one tile. Tiles overlap by
   * half the model's receptive field and each writes only its own output
   * rectangle, like the browser runtime. */
  uint32_t max_tile_size;
  /* Positive finite scale applied to HDR color before the transfer (default 1). */
  float input_scale;
  /* OIDNW_FILTER_* flags (default 0). */
  uint32_t flags;
} oidnw_filter_desc;

typedef struct oidnw_device oidnw_device;
typedef struct oidnw_filter oidnw_filter;

/* API version = major << 16 | minor << 8 | patch. Use matching headers
 * and library; dynamic loaders must require the exact version. */
#define OIDNW_API_VERSION 0x00030000u
uint32_t oidnw_api_version(void);

void oidnw_filter_desc_default(oidnw_filter_desc *desc);

/* Creates a GPU device (oidnw_backend). Blocks while the adapter is opened. */
int32_t oidnw_device_create(uint32_t backend, oidnw_device **device,
  char *error, size_t capacity);
/* Any output may be NULL. name receives the adapter name; backend the
 * oidnw_backend in use; features OIDNW_FEATURE_* bits enabled on the device. */
int32_t oidnw_device_info(const oidnw_device *device, char *name, size_t name_capacity,
  uint32_t *backend, uint32_t *features);
void oidnw_device_destroy(oidnw_device *device);

/* weights: an OIDN TZA file (copied). desc may be NULL for defaults.
 * Validates the model; GPU resources are created by prepare or the first
 * execute of each image size. */
int32_t oidnw_filter_create(oidnw_device *device, const void *weights, size_t weights_size,
  const oidnw_filter_desc *desc, oidnw_filter **filter, char *error, size_t capacity);
/* Images the model reads: 1 (color), 2 (color, albedo) or 3 (color, albedo,
 * normal); 0 for NULL. */
uint32_t oidnw_filter_input_count(const oidnw_filter *filter);
/* Prepares one size without input images or a GPU execution. May block for
 * shader compilation. Repeating the current size reuses resources; changing
 * size creates a new workspace while reusing weights and matching pipelines.
 * Concurrent prepare/execute on the same filter returns OIDNW_ERROR_BUSY. */
int32_t oidnw_filter_prepare(const oidnw_filter *filter, uint32_t width, uint32_t height,
  char *error, size_t capacity);
/* Denoises one image. All images are width x height RGBA f32 pixels, tightly
 * packed, in the same row order; albedo and normal are required as the model
 * reads them (otherwise ignored, may be NULL). Albedo is linear [0, 1];
 * normals are signed [-1, 1], as in Open Image Denoise. Like Open Image
 * Denoise, NaN becomes 0 and values are clamped to these ranges (HDR color to
 * [0, FLT_MAX]); the output is never NaN or negative. The output keeps the
 * color's alpha and may alias any input. Blocks until the result is written.
 * Changing the size plans and allocates for the new size (cached until the
 * next size change). */
int32_t oidnw_filter_execute(oidnw_filter *filter, uint32_t width, uint32_t height,
  const float *color, const float *albedo, const float *normal, float *output,
  char *error, size_t capacity);
/* Updates the scale for subsequent executions without preparing resources again.
 * scale must be positive and finite, including for LDR where it is ignored.
 * Concurrent prepare/execute/set_input_scale on this filter returns BUSY. */
int32_t oidnw_filter_set_input_scale(oidnw_filter *filter, float scale,
  char *error, size_t capacity);
void oidnw_filter_destroy(oidnw_filter *filter);

#ifdef __cplusplus
}
#endif
#endif
