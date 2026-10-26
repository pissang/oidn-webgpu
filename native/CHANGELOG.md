# Native changelog

Changes to the native library: the Rust crate `oidn-webgpu-native` and the C/C++ packages published as `native-v<VERSION>` GitHub Releases. Versions follow `version` in `Cargo.toml`, which is independent of the npm package. Browser package changes are in the [repository changelog](https://github.com/pissang/oidn-webgpu/blob/main/CHANGELOG.md).

## [Unreleased]

## [0.2.0] - 2026-10-26

Native 0.2.0 adds a GPU C API for host-owned Metal, Vulkan and DX12 buffers, C++17 wrappers and an optional Windows CUDA adapter. C ABI 3.0.0 replaces the published 1.0.0 ABI and requires rebuilding C consumers. The Rust crate has breaking changes; see [Migration notes](#migration-notes).

### Breaking changes

- Rust `DenoiseParams`, `DenoiseInfo`, `Kernel`, `KernelPath`, `Transfer`, `Limit` and `Precision` are non-exhaustive. Construct execution parameters with `DenoiseParams::default().with_input_scale(scale)` and include wildcard arms when matching these enums. C ABI implementation modules are hidden from Rust documentation and outside Rust semver guarantees.
- C ABI 3.0.0 replaces 1.0.0. `oidnw_filter_desc` no longer has a `size` field, and older descriptor layouts are not accepted. Descriptors, `oidnw_gpu_limits` and `oidnw_gpu_prepared_info` start with a NULL `next` extension pointer; non-NULL extensions return `INVALID_ARGUMENT`. CPU filter `input_scale` must be positive and finite, and `flags` accepts only defined `OIDNW_FILTER_*` bits; zero scales and unknown flag bits return `INVALID_ARGUMENT`.

### Added

- Rust `DenoiserOptions` exposes experimental `kernel` (default `Kernel::Auto`) and `quality` (default `Quality::High`). `Quality::Balanced` currently matches HIGH; `Quality::Fast` explicitly selects lower-precision f16 matrix accumulation. Explicit `Kernel::SubgroupMatrix` requires the enabled cooperative-matrix device feature or creation returns `Error::UnsupportedDevice`. Kernel/quality selection is outside semver guarantees.
- `Denoiser::warm_up` compiles pipelines before image buffers exist. Preparation on the same denoiser and size reuses them, while still planning and allocating the workspace.
- `oidnw_filter_set_input_scale` changes CPU filter exposure without rebuilding its workspace.
- GPU C API ([`oidn_webgpu_gpu.h`](include/oidn_webgpu_gpu.h)): denoises host GPU buffers without uploads or readback. Submissions wait on and signal host timeline objects on the GPU, with bounded pending work and polling or waiting for completion. GPU errors and device loss quarantine the context; `oidnw_gpu_device_destroy_quiescent` releases it once the host knows the GPU is idle. `oidnw_gpu_filter_warm_up` compiles pipelines before image buffers exist, and `oidnw_gpu_prepared_get_info` reports the prepared workspace. Kernel/quality options and the matrix flags below are experimental and outside semver guarantees. Limits and defaults are in the [GPU C API guide](README.md#gpu-c-api).
  - **Metal (macOS):** imports the host `MTLDevice`, buffers and shared events, with an optional shared queue guarded by host lock callbacks. Supported cooperative matrices are enabled by default; `OIDNW_GPU_METAL_NO_COOPERATIVE_MATRIX` disables them.
  - **Vulkan (Windows; Linux not GPU-tested):** borrows the host instance, physical device, device and queue, and imports storage buffers and timeline semaphores. `oidnw_gpu_vulkan_instance_extensions` and `oidnw_gpu_vulkan_requirements_create/get` report extensions and features before instance/device creation. Preflight includes cooperative-matrix requirements by default when supported unless `OIDNW_GPU_VULKAN_NO_COOPERATIVE_MATRIX` is set; import enables matrices only when the host enabled the extension and feature.
  - **DX12 (Windows):** creates a separate library device on the host adapter's LUID with a matching Agility SDK, and imports shared committed UAV buffers and shared fences. Uses implicit GEMM by default; cooperative matrices are unavailable.
- C++17 wrappers (`oidn_webgpu_gpu.hpp` and one header per platform) with move-only owners, shared context lifetime, status-returning submit/poll/wait, explicit `close(timeout_ns)` and optional host-object retention. Helpers expose device information, model input count and status names; `workspace::info()` exposes immutable workspace information, and `oidnw::gpu::warm_up` compiles pipelines before image buffers exist. CMake target `oidn_webgpu::cpp` enables C++17 and MSVC exception handling.
- Vulkan C++ helpers in `oidn_webgpu_vulkan.hpp`: `merge_requirements` merges the library's device requirements into your `VkDeviceCreateInfo` (deduplicated extensions, OR-ed features, `VkPhysicalDeviceFeatures2` converted, unsupported structures rejected before `vkCreateDevice`); `queue_lock` is the shared queue mutex; and an `import_device(handles, instance_info, device_info, lock)` overload fills the import descriptor and retains the lock.
- Optional Windows CUDA/DX12 C++17 adapter (`oidn_webgpu_cuda.hpp`, CMake target `oidn_webgpu::cuda`) for CUDA renderers: copies from ordinary CUDA device pointers or denoises adapter-allocated shared images without copies, and orders work on CUDA streams. Options select HDR/LDR and PU/log transfer; debug layers default to off, and GPU validation requires explicitly enabling debug. `resize(width, height)` reuses devices and weights after task/lease release and consumer retirement; `info().workspace` reports memory and selected kernels. It uses the DX12 backend and is built only when enabled, so CUDA is not otherwise required.
- The [GPU C integration walkthrough](examples/gpu-integration.md) and complete hosts for each route: `metal_gpu.m`, `metal_gpu_cpp.mm`, `vulkan_gpu.cpp`, `dx12_gpu.cpp` and `cuda_dx12_gpu.cu`. All ship in native packages with their SPIR-V shaders.

### Changed

- Native source builds watch the shared core input list and native source directory, so browser-only edits no longer trigger scriptc.
- The macOS ARM64 C shared library is 4.67 MB instead of 6.49 MB before compression, including the new GPU C API.
- Registry crate builds for `*-windows-gnu` targets fail at build time with a message directing to `x86_64-pc-windows-msvc`. Windows GNU has no bundled core or precompiled package, and its source builds are not validated.

### Migration notes

- Rebuild C consumers against the 3.0.0 headers and remove assignments to `oidnw_filter_desc.size`. Ship a compatible library as described below.
- Initialize descriptors with their default functions and query outputs with `{0}`, keep `next` NULL and replace zero CPU input scales with `1.0`. Use only defined CPU filter flag bits.
- Replace Rust `DenoiseParams { input_scale: scale }` literals with `DenoiseParams::default().with_input_scale(scale)`. Add wildcard arms when matching `Kernel` or the other newly non-exhaustive enums.
- Dynamic loaders must check `OIDNW_API_VERSION_IS_COMPATIBLE(oidnw_api_version())` before calling the API; C++ wrappers, including those used by the CUDA adapter, do this. The header exposes `OIDNW_API_VERSION_MAJOR`, `OIDNW_API_VERSION_MINOR` and `OIDNW_API_VERSION_PATCH`. Compatibility rules are in [Versioning](README.md#versioning).

## [0.1.1] - 2026-10-25

### Added

- Native precompiled packages support CMake `find_package(oidn_webgpu CONFIG REQUIRED)` with `oidn_webgpu::oidn_webgpu` for the C filter and `oidn_webgpu::oidn_webgpu_core` for the static planner. Packaged CMake examples build and install the C applications with their shared library.

### Changed

- Native Release package filenames now contain only the crate version and target triple; download URLs no longer include the source commit. The full commit remains in `manifest.json`; CI and local archive names still include it.

## [0.1.0] - 2026-10-25

First release, published together with oidn-webgpu 0.6.0: experimental native denoising through Rust and C APIs on Metal, Vulkan or Direct3D 12, with precompiled C packages and a Rust crate for Windows x86_64 MSVC, Linux x86_64 GNU, macOS x86_64 and macOS ARM64. The [0.6.0 release notes](https://github.com/pissang/oidn-webgpu/blob/main/CHANGELOG.md#060---2026-10-25) list the native additions, fixes and performance measurements.

[Unreleased]: https://github.com/pissang/oidn-webgpu/compare/native-v0.2.0...HEAD
[0.2.0]: https://github.com/pissang/oidn-webgpu/compare/native-v0.1.1...native-v0.2.0
[0.1.1]: https://github.com/pissang/oidn-webgpu/compare/native-v0.1.0...native-v0.1.1
[0.1.0]: https://github.com/pissang/oidn-webgpu/releases/tag/native-v0.1.0
