# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Added

- A macOS Metal interoperability validation example that imports host GPU buffers and checks asynchronous execution without uploading or reading back image pixels.
- Windows Vulkan host-device/buffer import and DX12 shared-buffer/fence validation prototypes, plus a reproducible GPU test runner. These source examples precede the GPU C ABI; the C filter API is unchanged.

### Changed

- Reduce the macOS ARM64 native C shared library from 6.49 MB to 4.38 MB before compression, excluding model weights.
- Clarify that Windows native releases support `x86_64-pc-windows-msvc` only; Windows GNU has no bundled core or precompiled package, and its source builds are not validated.

## [Native 0.1.1] - 2026-10-25

### Added

- Native precompiled packages support CMake `find_package(oidn_webgpu CONFIG REQUIRED)` with `oidn_webgpu::oidn_webgpu` for the C filter and `oidn_webgpu::oidn_webgpu_core` for the static planner. Packaged CMake examples build and install the C applications with their shared library.

### Changed

- Native Release package filenames now contain only the crate version and target triple; download URLs no longer include the source commit. The full commit remains in `manifest.json`; CI and local archive names still include it.

## [0.6.0] - 2026-10-25

oidn-webgpu 0.6.0 is the first release under the new package name. It adds experimental native Rust and C APIs, sanitizes inputs and outputs like OIDN, and lowers default FP16 latency in Chrome by up to 19% on the measured GPUs. The package name is the only breaking change.

### Breaking changes

- The package is renamed from `oidn-web` to `oidn-webgpu`. Replace the dependency and imports (`import { initUNetFromURL } from 'oidn-webgpu'`); the API is unchanged.

### Added

- Experimental native denoising through Rust and C APIs on Metal, Vulkan or Direct3D 12. Rust hosts can share GPU buffers; the C API accepts CPU images. See the [native guide](native/README.md).
- Native releases provide precompiled C packages and a Rust crate with SHA-256 verified cores for Windows x86_64 MSVC, Linux x86_64 GNU, macOS x86_64 and macOS ARM64. The bundled crate builds without Node, npm, Zig or build-time downloads from GitHub.
- Native packages include standalone Rust and Windows MSVC applications for output validation and timing. The Rust application's Git dependency is pinned to the package source revision.
- Experimental `kernel: 'subgroup-matrix'` and `subgroupMatrix` tuning options. Browsers need `chromium-experimental-subgroup-matrix`; native hosts need wgpu cooperative-matrix support.

### Changed

- Default FP16 denoising in Chrome has 11–19% lower latency on Apple M4 Max and 0–16% on NVIDIA RTX 4060 Laptop than 0.5.0 ([Performance](#performance)); first initialization takes longer. Output differs slightly from 0.5.0; `gemm: { decoderSubPixel: false }` restores the 0.5.0 decoder calculation. FP32 is unchanged.
- `getLastExecutionProfile()` sums network time over all tiles of an image instead of reporting only the first tile. Preprocessing and output conversion are excluded.

### Fixed

- Browser CPU input preserves normalized `Float32Array` albedo and normal values instead of dividing them by 255. `tileExecute` now accepts floating-point auxiliary images in its TypeScript API; 8-bit auxiliary images still use [0, 255].
- Windows native packages build and link with MSVC. C consumers use the DLL CRT (`/MD`).
- Native cooperative-matrix denoising produces finite, repeatable output with shader runtime checks disabled on the tested NVIDIA Vulkan device.
- Native distributions include the required runtime license notices, and the crate declares the bundled components' licenses alongside MIT.
- Inputs and network outputs are sanitized like OIDN: NaN becomes 0, and color, albedo and normals are clamped to their valid ranges. Previously NaN or out-of-range input could spread across the image, HDR output could be negative and GPU LDR output could exceed 1.
- `examples/aux.*` are renamed to `examples/auxiliary.*` so the repository can be checked out on Windows, where `AUX` is a reserved file name.

### Performance

Measured on 2026-10-24 with the [Mac benchmark suite](docs/development.md#benchmarks-and-gpu-checks) protocol:

- Cells are median milliseconds per image over 21 executions after 5 warmups, with no outliers removed. Parentheses show the latency reduction relative to 0.5.0 in Chrome.
- Each execution denoises one whole HDR image with albedo and normals (`inputScale=1`) in GPU-resident RGBA32F buffers and synchronizes before the next. Timings include preprocessing, network execution and output conversion; they exclude device creation, model preparation, uploads and readback.
- oidn-webgpu uses FP16 with default accumulation; native runs with shader runtime checks disabled, the C API default. Official OIDN loads the same weights: Small `rt_hdr_alb_nrm_small.tza` at fast quality, Base `rt_hdr_alb_nrm.tza` at high quality, Large `rt_hdr_calb_cnrm_large.tza` at high quality with clean auxiliary images.
- 720p = 1280 x 720, 1080p = 1920 x 1080, 1440p = 2560 x 1440. Image quality is not compared.

#### Apple M4 Max

40 GPU cores, macOS 27.0.1, Chrome 155.0.8059.85, OIDN 2.5.1 Metal, revision `fd0d816`. Chrome subgroup matrix needs `--enable-unsafe-webgpu`; native uses Metal cooperative matrices. Large 720p comes from a separate run because every column, OIDN included, was 8–17% slower in the main run.

| Model | Input | 0.5.0 Chrome | Chrome | Chrome subgroup matrix | Native Metal | OIDN Metal |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Small | 720p | 15.60 | 13.60 (↓12.8%) | 11.90 (↓23.7%) | 8.51 (↓45.4%) | 8.78 (↓43.7%) |
| Small | 1080p | 35.00 | 30.40 (↓13.1%) | 26.50 (↓24.3%) | 18.48 (↓47.2%) | 19.19 (↓45.2%) |
| Small | 1440p | 61.60 | 53.50 (↓13.1%) | 46.40 (↓24.7%) | 32.26 (↓47.6%) | 33.48 (↓45.7%) |
| Base | 720p | 28.20 | 22.90 (↓18.8%) | 20.20 (↓28.4%) | 14.55 (↓48.4%) | 15.91 (↓43.6%) |
| Base | 1080p | 62.90 | 51.20 (↓18.6%) | 45.00 (↓28.5%) | 32.09 (↓49.0%) | 35.12 (↓44.2%) |
| Base | 1440p | 123.60 | 103.70 (↓16.1%) | 81.90 (↓33.7%) | 58.59 (↓52.6%) | 65.42 (↓47.1%) |
| Large | 720p | 64.90 | 54.20 (↓16.5%) | 50.50 (↓22.2%) | 36.29 (↓44.1%) | 34.85 (↓46.3%) |
| Large | 1080p | 161.20 | 143.00 (↓11.3%) | 127.10 (↓21.2%) | 93.94 (↓41.7%) | 89.79 (↓44.3%) |
| Large | 1440p | 295.10 | 259.30 (↓12.1%) | 224.70 (↓23.9%) | 171.30 (↓42.0%) | 159.14 (↓46.1%) |

#### NVIDIA RTX 4060 Laptop

8 GiB, Intel Core i9-13900H, Windows 11 build 26200, driver 566.14, Chrome 156.0.8078.48, OIDN 2.5.1 CUDA, on AC power, revision `b30a515`. Native uses Vulkan cooperative matrices; its output was not checked for correctness on this device.

| Model | Input | 0.5.0 Chrome | Chrome | Native Vulkan | OIDN CUDA |
| --- | --- | ---: | ---: | ---: | ---: |
| Small | 720p | 30.10 | 30.10 (0.0%) | 9.44 (↓68.6%) | 10.27 (↓65.9%) |
| Small | 1080p | 66.20 | 62.50 (↓5.6%) | 20.49 (↓69.1%) | 22.92 (↓65.4%) |
| Small | 1440p | 113.70 | 105.40 (↓7.3%) | 32.98 (↓71.0%) | 39.76 (↓65.0%) |
| Base | 720p | 58.90 | 51.00 (↓13.4%) | 15.36 (↓73.9%) | 20.02 (↓66.0%) |
| Base | 1080p | 121.00 | 101.80 (↓15.9%) | 34.05 (↓71.9%) | 45.45 (↓62.4%) |
| Base | 1440p | 195.50 | 166.80 (↓14.7%) | 59.50 (↓69.6%) | 78.51 (↓59.8%) |
| Large | 720p | 125.50 | 108.30 (↓13.7%) | 37.58 (↓70.1%) | 40.41 (↓67.8%) |
| Large | 1080p | 262.70 | 229.50 (↓12.6%) | 75.10 (↓71.4%) | 92.42 (↓64.8%) |
| Large | 1440p | 453.30 | 406.50 (↓10.3%) | 130.59 (↓71.2%) | 163.50 (↓63.9%) |

## [0.5.0] - 2026-10-09

oidn-web 0.5.0 simplifies device setup, makes tiled execution always settle, and switches FP16 convolutions to implicit GEMM. It contains breaking changes for TypeScript callers that pass `adapterInfo`, for code that constructs `UNet` directly, and for code that relies on square tiles or per-frame tile pacing.

### Performance

Measured with `npm run benchmark -- --direct-gemm-only` in headless Chrome, using the Clean Aux Large model with FP16. Times are medians. Performance depends on the browser, GPU, model, and tile size. Devices without `shader-f16` use FP32, which already used implicit GEMM in 0.4.0, so their gains are expected to be smaller.

| GPU | Input | 0.4.0 | 0.5.0 | Speedup |
| --- | --- | ---: | ---: | ---: |
| Apple GPU (macOS) | 512 x 512, one tile | 27.2 ms | 19.1 ms | 1.42x |
| Apple GPU (macOS) | 1920 x 1080, 512 px tiles (12 tiles) | 742 ms | 369 ms | 2.01x |
| NVIDIA RTX 4060 Laptop (Windows 11, Chrome 153) | 512 x 512, one tile | 339 ms | 39.7 ms | 8.55x |
| NVIDIA RTX 4060 Laptop (Windows 11, Chrome 153) | 1920 x 1080, 512 px tiles (12 tiles) | 8844 ms | 555 ms | 15.94x |

- Implicit GEMM is now the default for FP16. The 0.4.0 direct FP16 kernel was especially slow on the tested NVIDIA GPU. Network GPU time for a 512 x 512 tile drops from 26.8 ms to 18.5 ms on the Apple GPU, and from 360 ms to 38.5 ms on the RTX 4060.
- Overlap only on shared tile edges reduces the input each tile computes. With the direct kernel alone, the 1920 x 1080 case is about 1.5x faster than 0.4.0 on both GPUs.

### Breaking changes

- `initUNetFromURL` and `initUNetFromBuffer` take `{ device }` as the second argument. `adapterInfo` was only needed by the removed TensorFlow.js backend and is no longer accepted by the type. Extra properties are ignored at runtime, but TypeScript rejects an object literal that still contains `adapterInfo`.
- `new UNet(tensors, device, options)` takes a `GPUDevice` instead of a `{ device, adapterInfo }` object.
- `tileExecute` now continues on the event loop between tiles by default instead of waiting for `requestAnimationFrame`. Pass `scheduling: 'animation-frame'` to keep pacing tiles to display frames.
- Tiles are balanced rectangles with overlap only on edges shared with another tile, instead of fixed-size squares. Tile count, position, and size reported to `progress` differ from 0.4.0. The default `dynamicTile.initialTileSize` is now 432.
- `UNetExecutionStats` no longer has `tileWidth` and `tileHeight`. Use `tileColumns`, `tileRows`, `tileOverlap`, `inputPixelCount`, and `inputShapeCount` instead.
- FP16 convolutions use implicit GEMM by default. Output differs slightly from 0.4.0 because accumulation order changed. Set `kernel: 'direct'` to reproduce the 0.4.0 FP16 path.

### Added

- `hdrTransfer: 'log'` for RTLightmap HDR models, alongside the default PU transfer.
- `error` callback on `tileExecute` for asynchronous failures, WebGPU device loss, and exceptions thrown by `progress` or `done`. `progress` and `done` may return promises.
- `scheduling: 'event-loop' | 'animation-frame'`, `tileOverlap`, and `wholeImage` options on `tileExecute`.
- `prepareForImage(width, height)` to create per-shape GPU resources before the first denoise.
- `planTileGrid` and its `TilePlan`, `PlannedTile`, and `TileRect` types.
- Experimental `gemm` tuning options. These are intended for benchmarking and are not covered by semver.
- `hdrTransfer` and `activeExecutionCount` in `getRuntimeInfo()`, and the active GEMM configuration under `kernel.gemm`.

### Changed

- Implicit GEMM tiles are selected by output alignment and GPU limits, with optimized addressing, weight layout, register tiles, shared-memory layout, and pooling access.
- The final RGB convolution uses an adaptive shared-memory cache.
- Adaptive tile sizing uses the smoothed P75 tile GPU time, excludes the cold first tile, ignores cancelled and single-tile work, and buckets input shapes to at most two sizes.
- `animation-frame` scheduling falls back to a 100 ms timer so execution still completes in hidden tabs.

### Fixed

- Tiled execution always settles with `done` or `error`, or stops silently after abort, including when `requestAnimationFrame` never fires or the device is lost.
- Completed executions release their device-loss listeners.
- Edge tiles whose size is not a multiple of 16 replicate edge pixels into the padded model input.
- The npm package no longer includes benchmark results, tests, and scripts.

### Migration notes

- Replace `{ device, adapterInfo }` with `{ device }`.
- Replace `new UNet(tensors, { device, adapterInfo }, options)` with `new UNet(tensors, device, options)`.
- Interactive renderers that share the GPU with OIDN should pass `scheduling: 'animation-frame'`.
- Pass an `error` callback to handle failures. Without one, failures are logged to the console.

## [0.4.0] - 2026-08-20

oidn-web 0.4.0 replaces the TensorFlow.js inference stack with a purpose-built, model-driven WebGPU runtime. Existing `initUNetFromURL` and `initUNetFromBuffer` integrations remain supported while gaining native FP16, adaptive scheduling, runtime diagnostics, and stronger model validation.

### Highlights

- Removed TensorFlow.js and its WebGPU backend from the runtime dependencies.
- Added a custom WGSL U-Net executor with automatic FP16 selection and a deterministic FP32 fallback.
- Added built-in topology descriptors for the current OIDN small and large RT models, including clean auxiliary variants.
- Reduced the reference Vite ESM bundle from approximately 168.8 KB to 31.1 KB gzip compared with 0.3.5, a reduction of about 82%.
- Reached a 27.6 ms median for a 512 x 512 Clean Aux Large inference on the tested Apple GPU, versus 35.0 ms for the TensorFlow.js baseline (1.27x faster). Performance depends on the browser, GPU, model, and tile size.

### Added

- Model-driven graph planning and lifetime-aware activation-buffer reuse.
- Versioned `UNetModelSpec` descriptors, topology detection, and validation of tensor names, layouts, data types, byte lengths, kernel shapes, bias shapes, and graph channel flow.
- `modelSpec` initialization option for future or custom OIDN model topologies.
- `npm run model:inspect` for model hashes, tensor signatures, and descriptor compatibility checks.
- Dynamic tile sizing with configurable minimum, initial, and maximum tile sizes and a target GPU time.
- GPU queue backpressure between tiles to keep cancellation and rendering interaction responsive.
- Runtime diagnostics through `getRuntimeInfo()`, including the selected engine, precision, model, kernel capabilities, tile state, and resource statistics.
- Per-layer GPU profiling through `profileNextExecution()` and `getLastExecutionProfile()` when `timestamp-query` is enabled.
- Cross-version browser benchmarks with queue-complete timings, sampled output validation, TFJS baseline comparison, and per-layer GPU profiles.
- Explicit, idempotent resource disposal and resource lifecycle accounting.
- Automated model, graph, scheduler, resource leak, allocation rollback, and late-async-cleanup tests.

### Changed

- U-Net inference now runs directly in WGSL. The stable `auto` engine selects the native WGSL backend and no longer initializes TensorFlow.js.
- FP16-capable devices retain TZA half-float weights and use short FP16 FMA accumulation groups folded into FP32 accumulators. The final output remains FP32.
- Devices without `shader-f16` automatically use native FP32 inference.
- Convolution tensors and activations use a blocked four-channel layout with channel-specialized shaders.
- Decoder `upsample + concat + conv` patterns are fused, and all network passes for a tile are encoded into one command buffer.
- Shape-independent pipelines are compiled asynchronously before initialization resolves to avoid first-denoise shader compilation stalls.
- `maxTileSize` is now a hard upper bound for the adaptive tile controller.
- A shared `GPUDevice` uses FP16 only when `shader-f16` was requested during device creation; WebGPU features cannot be enabled afterward.

### Migration notes

- No changes are required for basic `initUNetFromURL` or `initUNetFromBuffer` usage.
- Call `dispose()` when a U-Net instance is no longer needed.
- When supplying an existing `GPUDevice`, request `shader-f16` before creating the device if FP16 inference is desired.
- Set `dynamicTile: false` to restore fixed-size tiling.

[Unreleased]: https://github.com/pissang/oidn-webgpu/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/pissang/oidn-webgpu/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/pissang/oidn-webgpu/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/pissang/oidn-webgpu/compare/v0.3.5...v0.4.0
