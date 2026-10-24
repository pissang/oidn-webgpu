# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Breaking changes

- The package is renamed from `oidn-web` to `oidn-webgpu`. Replace the dependency and imports (`import { initUNetFromURL } from 'oidn-webgpu'`); the API is unchanged.

### Added

- Experimental native denoising through Rust and C APIs on Metal, Vulkan or Direct3D 12. Rust hosts can share GPU buffers; the C API accepts CPU images. See the [native guide](native/README.md).
- Experimental `kernel: 'subgroup-matrix'` acceleration and `subgroupMatrix` tuning options. Browser use requires `chromium-experimental-subgroup-matrix`; native use requires wgpu cooperative-matrix support. On native hosts the kernel also computes the final RGB convolution with matrices.

### Changed

- Default FP16 GEMM denoising is faster on the tested Apple M4 Max, with increased first-time initialization cost. Outputs differ slightly from the previous FP16 path; `gemm: { decoderSubPixel: false }` restores its decoder calculation. FP32 defaults are unchanged.
- Native denoising uses less temporary GPU memory. Native FP16 GEMM also runs faster with shader runtime checks enabled on the tested Apple M4 Max.
- Native cooperative-matrix denoising has lower execution latency on the tested Apple M4 Max. Large also uses less temporary GPU memory. Shader compilation may take longer.
- GPU profiles now sum network time across all tiles of a completed image, rather than reporting only its first tile. Image preprocessing and output conversion are excluded.

### Fixed

- Inputs and network outputs are sanitized like OIDN: NaN becomes 0, and color, albedo and normals are clamped to their valid ranges. Previously NaN or out-of-range input could spread across the denoised image, and network outputs were not clamped, so HDR results could be negative and GPU LDR results could exceed 1.

### Performance on Apple M4 Max

Measured on 2026-10-24: Apple M4 Max (40 GPU cores), macOS 27.0.1 (26A434), Chrome 155.0.8059.85 and official OIDN 2.5.1 Metal. The release baseline is `v0.5.0` (`6428fc3`); the current revision is `fd0d816`. All cells come from one run of the Mac benchmark suite, except the Large 720p row: in that run every column, including OIDN Metal, was 8–17% slower than in neighbouring measurements, so the row was measured again in a separate run.

Each cell shows **median milliseconds per image**, from 5 warmups and 21 measured executions. Parentheses show the **latency reduction relative to 0.5.0**, calculated as `(0.5.0 time - measured time) / 0.5.0 time x 100%`. Every execution processes one whole image and synchronizes before the next. Inputs and outputs stay in GPU-resident RGBA32F buffers. Timings include HDR preprocessing, network execution, output conversion and host synchronization; they exclude device creation, model preparation, uploads and output readback. These are wall latencies, not GPU timestamp measurements.

All oidn-web paths use FP16 weights and activations with their default accumulation settings. Ordinary Chrome runs without the unsafe WebGPU flag; Chrome subgroup matrix requires `--enable-unsafe-webgpu`. `oidn-web(native wgpu)` uses Metal cooperative matrices with FP32 accumulation and shader runtime checks disabled, matching the C API default. All paths receive the same HDR color/albedo gradients and constant surface normals, use `inputScale=1`, and load the same TZA weights for each model.

| Model | Weights | Official OIDN setting |
| --- | --- | --- |
| Small | `rt_hdr_alb_nrm_small.tza` | Fast quality |
| Base | `rt_hdr_alb_nrm.tza` | High quality |
| Large | `rt_hdr_calb_cnrm_large.tza` | High quality, clean auxiliary images |

| Model | Input | oidn-web 0.5.0 (Chrome) | oidn-web (Chrome) | oidn-web (Chrome subgroup matrix) | oidn-web(native wgpu) | OIDN Metal |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Small | 720p | 15.60 ms | 13.60 ms (↓12.8%) | 11.90 ms (↓23.7%) | 8.51 ms (↓45.4%) | 8.78 ms (↓43.7%) |
| Small | 1080p | 35.00 ms | 30.40 ms (↓13.1%) | 26.50 ms (↓24.3%) | 18.48 ms (↓47.2%) | 19.19 ms (↓45.2%) |
| Small | 1440p | 61.60 ms | 53.50 ms (↓13.1%) | 46.40 ms (↓24.7%) | 32.26 ms (↓47.6%) | 33.48 ms (↓45.7%) |
| Base | 720p | 28.20 ms | 22.90 ms (↓18.8%) | 20.20 ms (↓28.4%) | 14.55 ms (↓48.4%) | 15.91 ms (↓43.6%) |
| Base | 1080p | 62.90 ms | 51.20 ms (↓18.6%) | 45.00 ms (↓28.5%) | 32.09 ms (↓49.0%) | 35.12 ms (↓44.2%) |
| Base | 1440p | 123.60 ms | 103.70 ms (↓16.1%) | 81.90 ms (↓33.7%) | 58.59 ms (↓52.6%) | 65.42 ms (↓47.1%) |
| Large | 720p | 64.90 ms | 54.20 ms (↓16.5%) | 50.50 ms (↓22.2%) | 36.29 ms (↓44.1%) | 34.85 ms (↓46.3%) |
| Large | 1080p | 161.20 ms | 143.00 ms (↓11.3%) | 127.10 ms (↓21.2%) | 93.94 ms (↓41.7%) | 89.79 ms (↓44.3%) |
| Large | 1440p | 295.10 ms | 259.30 ms (↓12.1%) | 224.70 ms (↓23.9%) | 171.30 ms (↓42.0%) | 159.14 ms (↓46.1%) |

Native latency was 3.1–3.7% lower than OIDN Metal for Small and 8.5–10.4% lower for Base. For Large it was 4.1–7.6% higher.

Input sizes: `720p` = 1280 x 720, `1080p` = 1920 x 1080, `1440p` = 2560 x 1440. No outlier samples are removed; the Large 1440p case had substantial variation (for example, 239.40–303.00 ms for ordinary Chrome). These local timings depend on GPU load, power state, browser, model and image size. They do not measure image quality, first-use latency or interactive tiled execution.

The [Mac benchmark suite](docs/development.md#benchmarks-and-gpu-checks) can reproduce these cases and retains every timing sample, P95, model hash, source revision, selected kernel and resource count in its local JSON report.

### Performance on NVIDIA RTX 4060 Laptop

Measured on 2026-10-24: NVIDIA GeForce RTX 4060 Laptop GPU (8 GiB), Intel Core i9-13900H, Windows 11 build 26200, NVIDIA driver 566.14, Chrome 156.0.8078.48 and official OIDN 2.5.1 CUDA, with the laptop plugged in. The current revision is `b30a515`; the locally remeasured 0.5.0 baseline is `6428fc3`.

Uses the same [timing protocol, model weights and OIDN quality settings](#performance-on-apple-m4-max). Parentheses use the same latency-reduction formula relative to the local 0.5.0 Chrome baseline. Ordinary Chrome uses FP16 implicit GEMM without the unsafe WebGPU flag. Native wgpu uses Vulkan cooperative matrices with default accumulation and shader runtime checks disabled.

| Model | Input | oidn-web 0.5.0 (Chrome) | oidn-web (Chrome) | oidn-web(native wgpu) | OIDN CUDA |
| --- | --- | ---: | ---: | ---: | ---: |
| Small | 720p | 30.10 ms | 30.10 ms (0.0%) | 9.44 ms (↓68.6%) | 10.27 ms (↓65.9%) |
| Small | 1080p | 66.20 ms | 62.50 ms (↓5.6%) | 20.49 ms (↓69.1%) | 22.92 ms (↓65.4%) |
| Small | 1440p | 113.70 ms | 105.40 ms (↓7.3%) | 32.98 ms (↓71.0%) | 39.76 ms (↓65.0%) |
| Base | 720p | 58.90 ms | 51.00 ms (↓13.4%) | 15.36 ms (↓73.9%) | 20.02 ms (↓66.0%) |
| Base | 1080p | 121.00 ms | 101.80 ms (↓15.9%) | 34.05 ms (↓71.9%) | 45.45 ms (↓62.4%) |
| Base | 1440p | 195.50 ms | 166.80 ms (↓14.7%) | 59.50 ms (↓69.6%) | 78.51 ms (↓59.8%) |
| Large | 720p | 125.50 ms | 108.30 ms (↓13.7%) | 37.58 ms (↓70.1%) | 40.41 ms (↓67.8%) |
| Large | 1080p | 262.70 ms | 229.50 ms (↓12.6%) | 75.10 ms (↓71.4%) | 92.42 ms (↓64.8%) |
| Large | 1440p | 453.30 ms | 406.50 ms (↓10.3%) | 130.59 ms (↓71.2%) | 163.50 ms (↓63.9%) |

Chrome latency decreased by 0.0–7.3% for Small, 13.4–15.9% for Base and 10.3–13.7% for Large relative to the local 0.5.0 baseline. Native Vulkan latency was 7.0–25.1% lower than OIDN CUDA across these cases.

Every measured path used one whole-image tile. Browser output samples were finite; native and OIDN checked every RGB output value outside the timed region. These measurements do not establish image-quality equivalence or directly compare GPU performance with the Apple M4 Max results. The local report retains all samples, P95, source revisions and model, bundle and binary hashes.

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
- Renamed `examples/aux.*` to `examples/auxiliary.*` so the repository can be checked out on Windows, where `AUX` is a reserved file name.

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

[Unreleased]: https://github.com/pissang/oidn-webgpu/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/pissang/oidn-webgpu/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/pissang/oidn-webgpu/compare/v0.3.5...v0.4.0
