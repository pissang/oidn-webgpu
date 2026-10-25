#ifndef OIDN_WEBGPU_CORE_H
#define OIDN_WEBGPU_CORE_H
/*
 * oidn-webgpu native planner: the oidn-webgpu TypeScript core compiled by scriptc
 * into a static library (liboidn_webgpu_core.a / oidn_webgpu_core.lib) with no GPU
 * dependency. It plans a denoising for one image size and device and returns
 * it as a program that the host replays with its own GPU API, sharing its
 * device and image buffers without copies. native/src/lib.rs replays programs
 * with wgpu and is the reference implementation.
 *
 * The functions use process-global state and are not thread-safe: serialize
 * all calls. Call oidnw_core_init once before the first recording.
 *
 * The scriptc runtime is localized to this archive, so it links beside other
 * scriptc libraries.
 */
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void oidnw_core_init(void);

/* Validates weights without a GPU. The returned little-endian u32 packet is
 * magic 0x4c444f4d, version 1, input image count (color, albedo, normal).
 * Failures use the same ERRO packet as recording. Reset after copying it. */
void oidnw_core_inspect_model(const uint8_t *weights, size_t weights_size,
  const uint8_t **result, size_t *result_size);

/*
 * Records the whole denoising of one image: tiles, input processing (HDR
 * transfer), the U-Net and output processing (color alpha preserved).
 *   weights: OIDN TZA file.
 *   config:  little-endian u32 words (config_size bytes):
 *     0 version (1), 1 image width, 2 image height,
 *     3 precision (0 auto, 1 fp32, 2 fp16),
 *     4 kernel (0 auto, 1 implicit-gemm, 2 direct, 3 subgroup-matrix),
 *     5 features (1 shader-f16, 2 subgroups,
 *       4 chromium-experimental-subgroup-matrix, 8 wgpu-cooperative-matrix),
 *     6 maxComputeWorkgroupStorageSize, 7 maxComputeInvocationsPerWorkgroup,
 *     8 maxComputeWorkgroupSizeX, 9 maxComputeWorkgroupSizeY,
 *     10 maxComputeWorkgroupsPerDimension, 11 subgroup min size,
 *     12 subgroup max size (the matrix kernels need min == max),
 *     13 matrix configuration count N, then N x 5 words:
 *        component type (0 f32, 1 f16), result type, M, N, K,
 *     then 3 subgroup-matrix options: double buffering (0 default, 1 off,
 *        2 on), accumulation (0 f32, 1 f16), sub-pixel decoders (0 on, 1 off),
 *     then 4 denoising words: hdr (0/1), transfer (0 PU, 1 log),
 *        maximum tile edge (0 = whole image), input scale (f32 bits, 0 = 1).
 *   program/program_size: receives the program, valid until oidnw_core_reset.
 *
 * Program: little-endian u32 words; strings and byte blobs are a u32 length
 * followed by the bytes padded to 4. Magic 0x4e44494f ("OIDN") then:
 *   version (2), input count, input buffer ids, output buffer id,
 *   output external (1), image width, image height, output channels (4),
 *   buffers:    count, then size, usage (WebGPU GPUBufferUsage bits),
 *               initial data blob (empty = zero-filled), label string
 *   shaders:    count, then WGSL source string
 *   pipelines:  count, then shader index, entry point string
 *               (compute pipelines with automatic layouts)
 *   bind groups: count, then pipeline index, group index, entry count,
 *               entries of (binding, buffer id, offset, size; 0 = whole)
 *   operations: count, then kind
 *     0 submit: command count, commands of kind
 *         0 dispatch: pipeline, bind group (group 0), x, y, z
 *         1 copy: source, source offset, destination, destination offset, size
 *     1 write:  buffer, offset, data blob
 * Optional trailing metadata (older replayers may ignore it):
 *   magic 0x4154454d (META), version 1, precision (1 f32, 2 f16), tile count,
 *   convolution kernel bits (1 direct, 2 implicit GEMM, 4 matrix, 8 spatial,
 *   16 subgroup), frame uniform buffer id (0xffffffff for none), immutable
 *   weight/bias buffer count and ids. A frame uniform stores inputScale as
 *   f32 at offset zero, initially the config's scale. Updating it in command
 *   order enables per-execution exposure. Immutable weight buffers can be
 *   shared for identical contents/usage on the same device; workspaces cannot.
 * Create every buffer, pipeline and bind group once, binding the host's
 * images in place of the input buffers (color, albedo, normal: RGBA f32
 * pixels) and the output buffer (RGBA f32). Then run the operations in order
 * for every denoising; dispatches inside one submit need storage barriers
 * between them, as in a WebGPU compute pass.
 * On failure the program is instead magic 0x4f525245 ("ERRO") followed by a
 * message string.
 *
 * WGSL uses `enable f16` with shader-f16 and, for subgroup-matrix kernels,
 * wgpu's `wgpu_cooperative_matrix` (feature bit 8) or Chrome's
 * `chromium_experimental_subgroup_matrix` (bit 4) spelling.
 */
void oidnw_core_record_denoise(const uint8_t *weights, size_t weights_size,
  const uint8_t *config, size_t config_size,
  const uint8_t **program, size_t *program_size);

/* Records only the U-Net for one input size (same config without the 4
 * denoising words): inputs are RGBA f32 tensors in the network's transfer
 * space and the output buffer is internal (output external = 0). */
void oidnw_core_record_network(const uint8_t *weights, size_t weights_size,
  const uint8_t *config, size_t config_size,
  const uint8_t **program, size_t *program_size);

/* Releases the last returned program. */
void oidnw_core_reset(void);
/* Frees unreachable scriptc heap memory. */
void oidnw_core_collect(void);

#ifdef __cplusplus
}
#endif
#endif
