# oidn-webgpu native

Native GPU denoising with the Open Image Denoise U-Net. It runs the same WGSL kernels as the browser package through [wgpu](https://wgpu.rs) on Metal, Vulkan or Direct3D 12, and denoises images that are already in your renderer's GPU buffers.

**Status:** experimental; APIs and results may change between versions. See [Tested configurations](#tested-configurations). CI builds and checks packages but does not run them on a GPU.

**Availability:** the latest release, Native 0.1.1 (C ABI 1.0.0; [changelog](https://github.com/pissang/oidn-webgpu/blob/main/native/CHANGELOG.md)), contains the Rust API, the CPU C filter and the core planner. The GPU C API, C++17 wrappers and CUDA adapter (C ABI 3.0.0) are not released yet: [build from source](#build-from-source) or use a [CI package](#precompiled-packages) from a commit that contains them. The interface described here uses C ABI 3.0.0 headers; see [Versioning](#versioning) for compatible library versions.

- [Choose an API](#choose-an-api)
- [Get the library](#get-the-library)
- [Rust API](#rust-api): wgpu renderers
- [GPU C API](#gpu-c-api): Metal, Vulkan and DX12 renderers
- [CUDA on Windows](#cuda-on-windows)
- [CPU C filter](#cpu-c-filter)
- [Image contract](#image-contract)
- [Precompiled packages](#precompiled-packages)
- [Validate a build](#validate-a-build)
- [Build configuration](#build-configuration)
- [Performance](#performance)
- [Licensing](#licensing)

## Choose an API

Use a GPU API whenever your images are already on the GPU. The images stay in your buffers, and denoising is ordered with your render passes by GPU synchronization, so the CPU does not wait.

| Your renderer | Use | Images | Available in | Start with |
| --- | --- | --- | --- | --- |
| Rust with wgpu `=30.0.1` | **Rust `Denoiser`** | Your `wgpu::Buffer`s, encoded into your command encoder | Native 0.1.0 | [Rust API](#rust-api) |
| Metal, Vulkan or DX12, in C, C++ or Objective-C | **GPU C API** with the C++17 wrapper | Your GPU buffers | Unreleased (C ABI 3.0.0) | [GPU C API](#gpu-c-api) |
| CUDA on Windows | **CUDA adapter** (C++17, on top of the DX12 route) | CUDA device pointers | Unreleased (C ABI 3.0.0) | [CUDA on Windows](#cuda-on-windows) |
| Offline tools, tests, any language with C FFI | CPU C filter | CPU arrays; each call uploads, denoises, reads back and blocks | Native 0.1.0 | [CPU C filter](#cpu-c-filter) |
| Your own non-wgpu GPU executor | Core planner | Host-defined | Native 0.1.0 | [Core planner](#core-planner) |

All APIs take linear RGBA32F images with the same [image contract](#image-contract). Weights are not bundled: choose a `.tza` file from the [weights table](https://github.com/pissang/oidn-webgpu#choose-weights) and ship it with your application.

## Get the library

| Language | Get it |
| --- | --- |
| Rust | Add the crate from crates.io; see [Rust API](#rust-api) |
| C, C++, Objective-C | Download a [precompiled package](#precompiled-packages) and use its CMake config, or build one from source |

### CMake

Set `CMAKE_PREFIX_PATH` to the extracted package root:

```cmake
cmake_minimum_required(VERSION 3.20)
project(my_renderer LANGUAGES CXX)

find_package(oidn_webgpu CONFIG REQUIRED)
add_executable(my_renderer main.cpp)
target_link_libraries(my_renderer PRIVATE oidn_webgpu::cpp)
```

| Target | Links | Use for | Available in |
| --- | --- | --- | --- |
| `oidn_webgpu::cpp` | Shared library, headers, C++17 and MSVC exception handling | C++ GPU hosts (recommended) | Unreleased |
| `oidn_webgpu::oidn_webgpu` | Shared library and C headers | C hosts: GPU C API or CPU filter | Native 0.1.1 |
| `oidn_webgpu::oidn_webgpu_core` | Planner-only static library and its system libraries | [Core planner](#core-planner) | Native 0.1.1 |
| `oidn_webgpu::cuda` | Optional CUDA adapter, built separately | [CUDA on Windows](#cuda-on-windows) | Unreleased |

With a Native 0.1.1 package, link `oidn_webgpu::oidn_webgpu` and set `project(... LANGUAGES C)`.

- The package can be moved without regenerating its CMake files. An optional `find_package` version uses the native crate version; patch versions with the same major and minor are compatible.
- Headers and libraries must match the application's OS and CPU architecture and satisfy [version compatibility](#versioning).
- Ship the shared library and your TZA weights with the application. On Windows, use the DLL CRT (`/MD`) and place `oidn_webgpu.dll` beside the executable or on `PATH`:

```cmake
if(WIN32)
  set_property(TARGET my_renderer PROPERTY MSVC_RUNTIME_LIBRARY MultiThreadedDLL)
  add_custom_command(TARGET my_renderer POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
      "$<TARGET_FILE:oidn_webgpu::oidn_webgpu>" "$<TARGET_FILE_DIR:my_renderer>"
    VERBATIM)
endif()
```

### Build from source

Needs Node 22+, npm, CMake 3.20+, stable Rust and a C toolchain; on Windows also Zig 0.16.0 on `PATH`. From the repository root:

```bash
npm run native:build
```

The package is written to `.native-build/package`; pass `-- --output DIR` to choose another directory. The commands in this guide use that path; any extracted download works the same way.

## Rust API

For Rust renderers on wgpu: the denoiser records into your command encoder and reads and writes your buffers.

### Install

The updated examples below use unreleased `warm_up` and `with_input_scale` methods; the `kernel` and `quality` options are also unreleased. Use a local checkout or a Git commit containing them; the published Native 0.1.1 crate does not provide these additions.

```toml
[dependencies]
oidn-webgpu = { package = "oidn-webgpu-native", path = "../oidn-webgpu/native" }
# Published API, without the new methods:
# oidn-webgpu = { package = "oidn-webgpu-native", version = "0.1.1" }
# Git source:
# oidn-webgpu = { package = "oidn-webgpu-native", git = "https://github.com/pissang/oidn-webgpu.git", rev = "<commit>" }
```

- The registry crate bundles cores for the four [supported targets](#precompiled-packages); on Windows that is `x86_64-pc-windows-msvc`. It needs only Rust and a platform linker.
- A Git or path dependency builds the core from source: stable Rust, a C toolchain, Node 22+ and npm, and on Windows Zig 0.16.0 on `PATH`. The first build installs the pinned scriptc compiler from the npm registry. To skip Node, use a [precompiled core](#build-configuration).
- **wgpu version:** the crate pins `wgpu = "=30.0.1"`. Your `Device`, `Buffer` and `CommandEncoder` must come from the same wgpu crate; another release or a fork gives incompatible types. Use the re-export: `use oidn_webgpu::wgpu;`.

### Usage

```rust
use oidn_webgpu::{DenoiseBuffers, DenoiseParams, Denoiser, DenoiserOptions, Model};

// Once.
let model = Model::from_tza(&weights)?;
let mut denoiser = Denoiser::new(&device, &adapter, &model, DenoiserOptions::default())?;

// Optional, before the renderer creates image buffers.
denoiser.warm_up([width, height])?;

// Per image size (allocates the workspace).
let mut prepared = denoiser.prepare([width, height], DenoiseBuffers {
    color: (&color).into(),
    albedo: Some((&albedo).into()),
    normal: Some((&normal).into()),
    output: (&output).into(),
})?;

// Per frame.
let mut encoder = device.create_command_encoder(&Default::default());
// ... your render pass writes color/albedo/normal ...
prepared.encode(&mut encoder, DenoiseParams::default().with_input_scale(exposure))?;
// ... your display pass reads output ...
queue.submit([encoder.finish()]);
```

| Step | Call | Cost and behavior |
| --- | --- | --- |
| Load | `Model::from_tza(&[u8])` | Validates the TZA file on the CPU. Clones share CPU weights and per-device GPU weights |
| Bind | `Denoiser::new(&device, &adapter, &model, options)` | Cheap. `adapter` must be the device's adapter |
| Warm up | `denoiser.warm_up([w, h])` | Optional shader compilation without image buffers or workspace allocation; prepare for the same size reuses pipelines |
| Prepare | `denoiser.prepare([w, h], buffers)` | May block for shader compilation. Allocates a workspace for that size; call again when the size changes |
| Encode | `prepared.encode(&mut encoder, params)` | Appends commands only: no submit, wait or readback |
| Rebind | `prepared.rebind(buffers)` | Swaps image buffers at the same size. Already encoded commands keep their old bindings |
| Inspect | `prepared.info()` | `DenoiseInfo`: precision, `kernels` used, `tile_count`, `workspace_bytes`, `shared_weight_bytes`, `planning_ms`, `prepare_ms`, `runtime_checks` |

[`examples/host_buffers.rs`](examples/host_buffers.rs) is a complete program that renders inputs on the GPU, denoises in the same encoder and writes RGBA32F output:

```bash
cargo run --release --manifest-path native/Cargo.toml --example host_buffers -- weights/rt_hdr_alb_nrm_small.tza output.f32 256 192 128
```

### Device setup

- Before requesting the device, add `Denoiser::required_features()`, `Denoiser::recommended_features(&adapter)` and `Denoiser::required_limits()`. Large images may need higher buffer limits. The library cannot enable features later.
- Precision is FP16 when the device enables `SHADER_F16`, otherwise FP32. On DX12, `SHADER_F16` also needs DXC (Shader Model 6.2+): ship `dxcompiler.dll` and its matching `dxil.dll`.
- With `kernel = Kernel::Auto`, cooperative-matrix kernels are selected when the device enables `EXPERIMENTAL_COOPERATIVE_MATRIX` (wgpu's experimental feature contract applies); on Windows and Linux they need Vulkan. Otherwise AUTO uses implicit GEMM. Selecting `Kernel::SubgroupMatrix` does not enable the feature; `Denoiser::new` returns `Error::UnsupportedDevice` if it is not enabled.

### Options

`DenoiserOptions` is fixed per `Denoiser` and `#[non_exhaustive]`: start from `default()` and assign fields.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `hdr` | `bool` | `true` | `false` for LDR weights |
| `transfer` | `Transfer` | `Transfer::Pu` | `Transfer::Log` for RTLightmap HDR weights |
| `max_tile_size` | `u32` | `0` | `0` = whole image; otherwise 16..=16384, rounded down to a multiple of 16. Reduces workspace memory only |
| `kernel` | `Kernel` | `Kernel::Auto` | AUTO selects matrix or implicit GEMM as above; `ImplicitGemm` and `Direct` select those paths explicitly |
| `quality` | `Quality` | `Quality::High` | `High`/`Balanced` use f32 matrix accumulation; `Quality::Fast` explicitly selects f16 accumulation when supported, with lower precision |

Kernel and quality selection are experimental and outside semver guarantees. Quality does not select the model weights; use the Small TZA for OIDN's fast model.

`DenoiseParams` is non-exhaustive and passed to every `encode`. Use `DenoiseParams::default().with_input_scale(exposure)` to change it every frame:

| Field | Default | Meaning |
| --- | --- | --- |
| `input_scale` | `1.0` | Positive, finite. Multiplies HDR color before the transfer and is undone on output. Ignored for LDR |

### Buffers, memory and errors

- `ImageBuffer { buffer, offset }` binds `width * height * 16` bytes from `offset`; `(&buffer).into()` uses offset 0. The offset must be a multiple of `min_storage_buffer_offset_alignment`. Input ranges may share a buffer if they do not overlap; output must be a different buffer from every input.
- Each prepared instance owns its workspace. Denoisers made from one `Model` (or its clones) share GPU weights per device; count `shared_weight_bytes` once. Loading the same TZA again as a new `Model` creates a separate cache.
- Prepared instances may outlive the `Denoiser`. Do not destroy buffers that pending GPU work uses. Submit one prepared instance's executions in encode order.
- Errors are a typed `Error`: `InvalidArgument`, `Model`, `UnsupportedDevice`, `Gpu`, `BufferTooSmall { .. }`, `LimitExceeded { .. }`. A successful `encode` does not mean the GPU succeeded; execution errors and device loss go through your wgpu error handling.
- `Denoiser::new` compiles shaders with bounds checks and workgroup initialization. `unsafe Denoiser::new_unchecked` disables them and can be faster; the caller upholds its documented safety contract.

## GPU C API

Unreleased: needs C ABI 3.0.0 headers and library; Native 0.1.1 packages do not contain them. See **Availability** at the [top of this guide](#oidn-webgpu-native).

The GPU C API denoises your renderer's GPU buffers. It never uploads or reads back image pixels. Your render pass signals a timeline value when the inputs are ready, the library waits for it on the GPU, and signals another value when the output is written. Applications use only the C ABI (or the header-only C++17 wrapper); no Rust or wgpu types are involved.

| Platform | How the library gets a device | Header | Complete host |
| --- | --- | --- | --- |
| Metal (macOS) | Imports your `MTLDevice`; uses its own queue, or shares yours | [`oidn_webgpu_metal.hpp`](include/oidn_webgpu_metal.hpp) | [`metal_gpu_cpp.mm`](examples/metal_gpu_cpp.mm) (C++), [`metal_gpu.m`](examples/metal_gpu.m) (C) |
| Vulkan (Windows; Linux not GPU-tested) | Borrows your instance, device and queue | [`oidn_webgpu_vulkan.hpp`](include/oidn_webgpu_vulkan.hpp) | [`vulkan_gpu.cpp`](examples/vulkan_gpu.cpp) |
| DX12 (Windows) | Creates its own device on your adapter; shares buffers and fences | [`oidn_webgpu_dx12.hpp`](include/oidn_webgpu_dx12.hpp) | [`dx12_gpu.cpp`](examples/dx12_gpu.cpp) |

Each `.hpp` includes its C header (`oidn_webgpu_metal.h`, `oidn_webgpu_vulkan.h`, `oidn_webgpu_dx12.h`) and the shared [`oidn_webgpu_gpu.h`](include/oidn_webgpu_gpu.h). The [integration walkthrough](examples/gpu-integration.md) shows the same steps in plain C for every platform.

### Example

A C++ Vulkan renderer on Windows or Linux (Linux is not GPU-tested), linked to `oidn_webgpu::cpp`. Metal and DX12 skip steps 1–2 and use the calls in the [import table](#platform-imports) for step 3; [`examples/metal_gpu_cpp.mm`](examples/metal_gpu_cpp.mm) is the complete Metal program.

The library takes part in instance and device creation: it adds instance extensions and device features, and import checks the original create infos. The snippet assumes:

| Name | Type | Requirement |
| --- | --- | --- |
| `app_info` | `VkApplicationInfo` | `apiVersion` is `VK_API_VERSION_1_2` or newer |
| `physical_device`, `family` | `VkPhysicalDevice`, `uint32_t` | Your GPU and a queue family with `GRAPHICS` and `COMPUTE` |
| `my_device_info` | `VkDeviceCreateInfo` | Your declaration: one unprotected queue in `family`, plus your own extensions and features as individual feature structs or `VkPhysicalDeviceFeatures2` |
| `color_buffer`, `albedo_buffer`, `normal_buffer`, `output_buffer` | `VkBuffer` | Created with `buffer_info` and bound to memory |
| `buffer_info` | `VkBufferCreateInfo` | `VK_BUFFER_USAGE_STORAGE_BUFFER_BIT`, size at least `width * height * 16`, exclusive to `family` |
| `ready_semaphore`, `done_semaphore` | `VkSemaphore` | Created with `semaphore_info` |
| `semaphore_info` | `VkSemaphoreCreateInfo` | `pNext` = `VkSemaphoreTypeCreateInfo` with `VK_SEMAPHORE_TYPE_TIMELINE`, initial value 0 |
| `tza` | `std::vector<uint8_t>` | Contents of a `.tza` weights file |
| `width`, `height` | `uint32_t` | Image size, 1..=16384 |
| `frame` | `uint64_t` | Starts at 1 and increases every frame; used as both semaphore values |
| `exposure` | `float` | HDR input scale, positive; `1.0f` if unused |

```cpp
#include <memory>
#include <vector>
#include "oidn_webgpu_vulkan.hpp"
namespace ovk = oidnw::vulkan;

// 1. Instance: enable the library's extensions (append your own to the list).
std::vector<const char *> extensions = ovk::instance_extensions(vkGetInstanceProcAddr, app_info.apiVersion);
VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
instance_info.pApplicationInfo = &app_info;
instance_info.enabledExtensionCount = uint32_t(extensions.size());
instance_info.ppEnabledExtensionNames = extensions.data();
VkInstance instance;
vkCreateInstance(&instance_info, nullptr, &instance);

// 2. Device: merge the library's requirements into your own declaration.
oidnw_gpu_vulkan_requirements_desc query;
oidnw_gpu_vulkan_requirements_desc_default(&query);
query.get_instance_proc_addr = vkGetInstanceProcAddr;
query.instance = instance;
query.physical_device = physical_device;
query.instance_create_info = &instance_info;
ovk::merged_device_info device_info = ovk::merge_requirements(ovk::query_requirements(query), my_device_info);
VkDevice device;
vkCreateDevice(physical_device, &device_info.get(), nullptr, &device);
VkQueue queue;
vkGetDeviceQueue(device, family, 0, &queue);

// 3. Import the device, image buffers and timeline semaphores. Keep instance_info,
//    extensions, device_info and my_device_info's queue array alive until here.
auto queue_lock = std::make_shared<ovk::queue_lock>();  // Retained by the context.
auto context = ovk::import_device({vkGetInstanceProcAddr, instance, physical_device, device, queue, family, 0},
                                 instance_info, device_info.get(), queue_lock);

auto color  = ovk::import_buffer(context, color_buffer,  buffer_info);
auto albedo = ovk::import_buffer(context, albedo_buffer, buffer_info);
auto normal = ovk::import_buffer(context, normal_buffer, buffer_info);
auto output = ovk::import_buffer(context, output_buffer, buffer_info);
auto ready  = ovk::import_sync(context, ready_semaphore, semaphore_info);
auto done   = ovk::import_sync(context, done_semaphore,  semaphore_info);
oidnw_gpu_images images = {{color.get(), 0}, {albedo.get(), 0}, {normal.get(), 0}, {output.get(), 0}};

// 4. Per image size: may compile shaders and block.
auto model     = oidnw::gpu::load_model(tza.data(), tza.size());
auto filter    = oidnw::gpu::make_filter(context, model);
auto workspace = oidnw::gpu::prepare(filter, width, height, images);
uint64_t last_sequence = 0;  // Renderer state: latest accepted submission.

// 5. Per frame. Your render submit signals ready = frame and your display submit
//    waits for done = frame, both at VK_PIPELINE_STAGE_ALL_COMMANDS_BIT
//    (VkTimelineSemaphoreSubmitInfo). Hold the lock around your own submits:
//      { std::lock_guard<ovk::queue_lock> guard(*queue_lock); vkQueueSubmit(queue, ...); }
auto submitted = workspace.submit(images, {ready.get(), frame}, {done.get(), frame}, exposure);
if (submitted.code == OIDNW_ERROR_QUEUE_FULL && last_sequence) {
  // Vulkan admits a submission only after earlier external waits resolve.
  context.wait(last_sequence);
  submitted = workspace.submit(images, {ready.get(), frame}, {done.get(), frame}, exposure);
}
if (submitted.accepted()) last_sequence = submitted.sequence;
if (!submitted.accepted()) {
  // Rejected, nothing queued: show the noisy frame; do not wait for done = frame.
} else if (!submitted) {
  // Accepted but failed: output is invalid; recreate the context.
}

// 6. Shutdown: reset workspace, filter, model and imports (for example by leaving
//    their scope), then close. Destroy your Vulkan objects only after this.
context.close();   // Waits for accepted work.
```

`context.info()` returns the adapter name, backend and enabled feature bits; `model.input_count()` returns the number of input images; `oidnw::gpu::status_name(code)` returns a status name, or `UNKNOWN` for an unrecognized code.

Startup failures throw `oidnw::gpu::error` (with the C status in `code`). Submit, poll, wait and close never throw; check their returned status. `model`, `filter` and `workspace` can be recreated when the image size changes; the imports and context live as long as the renderer. A fixed-size workspace accepts different imported buffers on every submit.

### Platform imports

| Step | Metal | Vulkan | DX12 |
| --- | --- | --- | --- |
| Context | `oidnw::metal::import_device(desc)` | `oidnw::vulkan::import_device(desc)` | `oidnw::dx12::create_device(desc)` |
| Image buffer | `import_buffer(ctx, MTLBuffer, bytes)` | `import_buffer(ctx, VkBuffer, VkBufferCreateInfo)` | `import_buffer(ctx, ID3D12Resource *)` |
| Timeline sync | `import_sync(ctx, MTLSharedEvent)` | `import_sync(ctx, VkSemaphore, VkSemaphoreCreateInfo)` | `import_sync(ctx, ID3D12Fence *)` |
| Before creating your device | Nothing | [Vulkan preflight](#vulkan) | Matching Agility SDK |
| C function prefix | `oidnw_gpu_*_metal` | `oidnw_gpu_*_vulkan` | `oidnw_gpu_*_dx12` |

The import functions live in the context function's namespace. Pass `.get()` from buffer and sync wrappers when filling `oidnw_gpu_images` and `oidnw_gpu_timeline_point`.

### Frame loop

- **Inputs ready:** the producer must signal its ready value on the GPU without waiting for the CPU `submit` call to return.
- **Waits and signals:** each submit takes up to 64 waits and 64 signals. Signal values for one sync must increase; `UINT64_MAX` is reserved.
- **Output ready:** completion covers denoising and every external signal. Your consumer waits on the done value on the GPU; no CPU wait is needed. Before rewriting an input or output, make sure both the denoise and your consumer of the output have completed.
- **Backpressure:** at most `max_pending_submissions` (default 64) submissions are in flight. Beyond that, submit returns `OIDNW_ERROR_QUEUE_FULL` with sequence 0 and touches no synchronization; do not make a consumer wait on that frame's signal.
- **Reclaiming:** call submit, `poll` or `wait` regularly so completed work is released. `poll(sequence)` never waits for the GPU. `wait(sequence, timeout_ns)` waits; `0` polls once and `UINT64_MAX` waits without a deadline.
- **In flight:** one workspace may have several submissions in flight. A buffer's role (input or output) is fixed by its first successful prepare or submit.
- **Native submission** may block on the CPU when queued work accumulates.

### Errors and recovery

| Result | Meaning | What to do |
| --- | --- | --- |
| Sequence 0, `OIDNW_ERROR_QUEUE_FULL` | Rejected; nothing queued | Retry after earlier work completes |
| Sequence 0, other status | Rejected; nothing queued | Fix the argument or state named in the message |
| Nonzero sequence, error status | Accepted, then failed to encode; its signals are still issued | Treat the output as invalid; the context rejects new work |
| `OIDNW_ERROR_GPU` from poll/wait | GPU error; the context is quarantined | Drain, destroy and import a new context |
| `OIDNW_ERROR_DEVICE_LOST` | Device lost, or submission outcome unknown | External signals may never arrive; rebuild the host device. Recovery is not validated |
| `OIDNW_ERROR_BUSY` | Concurrent CPU calls on one handle, or children/submissions still alive at destroy | Serialize calls; release children and drain before destroying |

A quarantined context can hold submissions whose outcome is unknown, so normal destruction keeps returning `BUSY`. Once your own synchronization proves that no GPU work can still access the imported buffers or library workspaces, `oidnw_gpu_device_destroy_quiescent` (C++: `close_quiescent()`) releases it. It does not cancel work; a timeout or an advanced CPU event is not such proof.

### Lifetime and threads

- Startup order: context, then imports, model, filter, workspace. Models can be released after filter creation, and filters after preparation.
- Handles may be released while their submissions are in flight; accepted work retains what it needs. A workspace's binding cache can retain an imported buffer until it is evicted or the workspace is destroyed.
- Shutdown order: release every child handle, wait for accepted sequences, then destroy the context (C++: `close(timeout_ns)`; a busy or timed-out close keeps everything and can be retried). Only then free your native buffers, syncs and device.
- C++ wrappers are move-only and keep the context alive. The last owner drains accepted work without a deadline and may block; a failed cleanup terminates unless you install a handler. Handlers, host-lifetime tokens, `release()` and `track()` are in the [C++17 wrapper walkthrough](examples/gpu-integration.md#c17-wrapper).
- Different handles on one context may be used from different threads; queue access is serialized. Calls on the same workspace must not overlap, and no handle may be destroyed while another call uses it.

### Filter options

`oidnw_gpu_filter_desc`, from `oidnw_gpu_filter_desc_default` (C++: `oidnw::gpu::filter_options()`):

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `next` | pointer | NULL | Reserved for extensions; must be NULL |
| `hdr` | u32 | `1` | `0` for LDR weights |
| `transfer` | enum | `OIDNW_TRANSFER_PU` | `OIDNW_TRANSFER_LOG` for RTLightmap HDR weights |
| `kernel` | enum | `OIDNW_KERNEL_AUTO` | Uses subgroup matrix when enabled on the device, otherwise implicit GEMM; explicit `SUBGROUP_MATRIX` returns `UNAVAILABLE` if not enabled |
| `quality` | enum | `OIDNW_QUALITY_HIGH` | HIGH/BALANCED use f32 matrix accumulation; FAST explicitly enables f16 accumulation |
| `max_tile_size` | u32 | `0` | `0` = whole image; otherwise 16..=16384, rounded down to a multiple of 16. Reduces workspace memory only |
| `flags` | bitmask | `OIDNW_FILTER_RUNTIME_CHECKS` | `0` disables shader runtime checks, which can be faster |
| `binding_cache_capacity` | u32 | `8` | 1..=64 cached image binding sets per workspace |

`oidnw_gpu_filter_warm_up(filter, w, h, ...)` (C++: `oidnw::gpu::warm_up(filter, w, h)`) optionally compiles pipelines before image buffers exist. It can block, but does not allocate image bindings or a workspace. A later prepare on the same filter and size reuses those pipelines; planning, weight upload and workspace allocation still occur during prepare. A different size may require more pipelines. Serialize warm-up and prepare on each filter.

Per-submit `input_scale` (default `1`, finite and positive) multiplies HDR color before the transfer and is undone on output. `get_limits` (C++: `context.limits()`) reports the image offset alignment, maximum buffer and image bytes and the pending-submission limit. In C, initialize `oidnw_gpu_limits` to `{0}` before the query; a non-NULL `next` returns `OIDNW_ERROR_INVALID_ARGUMENT`.

`oidnw_gpu_prepared_get_info` (C++: `workspace.info()`) reports the prepared workspace:

| Field | Type | Meaning |
| --- | --- | --- |
| `next` | pointer | Initialize to NULL; non-NULL extensions are rejected |
| `precision` | enum | `OIDNW_PRECISION_FP16` or `OIDNW_PRECISION_FP32`; image buffers remain RGBA32F |
| `kernels` | bitmask | Actual `OIDNW_GPU_KERNEL_PATH_*` paths; one workspace can use multiple paths |
| `tile_count` | u32 | Number of output tiles |
| `workspace_bytes` | u64 | Per-workspace GPU buffers, excluding host images, weights, transient uploads and driver overhead |
| `shared_weight_bytes` | u64 | Immutable GPU weights; count once per Model/device |
| `runtime_checks` | u32 | Nonzero when shader runtime checks are enabled |

Initialize the output to `{0}`. Information is immutable after prepare, so queries may overlap submit on the same workspace; each query needs separate output storage and the workspace must remain alive.

### Metal

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `next` | pointer | NULL | Reserved for extensions; must be NULL |
| `device` | `void *` | NULL | Required `id<MTLDevice>`; use `__bridge` under ARC |
| `queue` | `void *` | NULL | Optional shared host queue; NULL creates a library queue |
| `flags` | bitmask | `0` | `OIDNW_GPU_METAL_FP32` disables FP16; `OIDNW_GPU_METAL_NO_SUBGROUPS` disables subgroups; `OIDNW_GPU_METAL_NO_COOPERATIVE_MATRIX` disables cooperative matrices |
| `lock`, `unlock`, `user_data` | callbacks, pointer | NULL | Required with a shared queue; your code must use the same lock for that queue |
| `max_pending_submissions` | u32 | `64` | 1..=4096 |

- A shared queue needs a command-buffer capacity of at least 4096. Lock callbacks run on the calling thread, must not reenter the library or unwind, and `user_data` must stay valid until the context is destroyed.
- FP16, subgroups and experimental cooperative matrices are enabled when supported. Metal cooperative matrices have been verified on Apple M4 Max through the Rust API, CPU filter and GPU C API. On NVIDIA RTX 4060 Laptop Vulkan, cooperative matrices have been verified through the Rust API and CPU filter; the GPU C API has only been verified with implicit GEMM, and its Vulkan cooperative-matrix path remains unverified. Other hardware is unverified. For problems, set the platform's `NO_COOPERATIVE_MATRIX` flag or select `kernel = OIDNW_KERNEL_IMPLICIT_GEMM`. Matrix flags and GPU filter kernel/quality options are experimental and outside semver guarantees. FAST quality requires explicit selection.

Build and run both Metal hosts:

```bash
npm run native:build
cmake -S .native-build/package/examples -B .native-build/metal-examples \
  -DCMAKE_PREFIX_PATH="$PWD/.native-build/package"
cmake --build .native-build/metal-examples --target metal_gpu_cpp metal_gpu
.native-build/metal-examples/metal_gpu_cpp weights/rt_hdr_alb_nrm_small.tza
```

The hosts produce inputs with a Metal shader, denoise a 31×31 image and validate the output with another shader; only integer flags reach the CPU.

### Vulkan

Vulkan 1.2 or newer. The library takes part in instance and device creation, as in the [example](#example). The C++ helpers in `oidn_webgpu_vulkan.hpp`:

| Helper | Does |
| --- | --- |
| `instance_extensions(loader, api_version)` | Instance extensions to enable before `vkCreateInstance` |
| `query_requirements(desc)` | Device extensions and features after physical-device selection; includes `VK_KHR_cooperative_matrix` and its feature structure when supported, unless `NO_COOPERATIVE_MATRIX` is set |
| `merge_requirements(requirements, host_info)` | Returns a `merged_device_info` that enables yours and the library's: extensions deduplicated, features OR-ed, `VkPhysicalDeviceFeatures2` converted to `pEnabledFeatures`. Throws `INVALID_ARGUMENT` for declarations import rejects (aggregate `VkPhysicalDeviceVulkan1xFeatures`, other `pNext` structures, duplicates). Owns copies of names and features; borrows your queue and layer arrays. `requirements` may be released afterwards |
| `queue_lock` | Mutex the library and your code share for the queue; lock it with `std::lock_guard` around your queue calls |
| `import_device(handles, instance_info, device_info, lock, flags = 0, options = {})` | Fills `oidnw_gpu_vulkan_device_desc` and retains `lock` until the context is destroyed. `query.flags` must equal `flags` |

The C descriptor below is what these helpers fill; use it directly for `max_pending_submissions` or from C.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `next` | pointer | NULL | Reserved for extensions; must be NULL |
| `get_instance_proc_addr` | `PFN_vkGetInstanceProcAddr` | NULL | Required loader entry point, including your layers |
| `instance`, `physical_device`, `device`, `queue` | Vulkan handles | NULL | Required, from one context |
| `queue_family_index`, `queue_index` | u32 | `0` | An unprotected graphics/compute queue |
| `instance_create_info`, `device_create_info` | pointers | NULL | Original creation declarations, valid during import |
| `flags` | bitmask | `0` | `OIDNW_GPU_VULKAN_FP32` disables FP16; `OIDNW_GPU_VULKAN_NO_SUBGROUPS` disables subgroups; `OIDNW_GPU_VULKAN_NO_COOPERATIVE_MATRIX` disables cooperative matrices |
| `lock`, `unlock`, `user_data` | callbacks, pointer | NULL | Required; all your access to this queue must take the same lock (`queue_lock` provides both). Release it before calling the library |
| `max_pending_submissions` | u32 | `64` | 1..=4096 |

- Requirements queries include supported cooperative matrices by default; use the same flags for query and import. Import enables matrices only if the host enabled both the extension and `cooperativeMatrix`; otherwise AUTO uses implicit GEMM. For hardware coverage and fallback options, see [Metal](#metal).
- Import fully bound, non-sparse storage buffers and timeline semaphores with their original creation info. Buffers must belong to the queue family or use concurrent sharing that includes it; you perform any queue-family ownership transfers. Overlapping memory aliases, binary semaphores and aggregated Vulkan 1.1/1.2/1.3 feature structs are unsupported.
- The library borrows your objects: keep them and their memory alive until the context is destroyed.
- While an earlier submission's external waits are unresolved, the next submit returns `QUEUE_FULL`. Signal those producers independently and poll before retrying.

Build and run from an x64 Visual Studio developer PowerShell:

```powershell
npm run native:build
cmake -S .native-build/package/examples -B .native-build/gpu-examples `
  -DCMAKE_PREFIX_PATH="$PWD/.native-build/package" -DOIDNW_BUILD_VULKAN_GPU_EXAMPLE=ON `
  -DVulkan_INCLUDE_DIR="$env:VULKAN_SDK/Include"
cmake --build .native-build/gpu-examples --config Release --target vulkan_gpu
.native-build/gpu-examples/Release/vulkan_gpu.exe weights/rt_hdr_alb_nrm_small.tza .native-build/package/examples/shaders 1000 16 0 0 0 1
```

### DX12

DX12 cannot import another `ID3D12Device`, so the library creates its own device and queue on your adapter's LUID and shares buffers and fences with it.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `next` | pointer | NULL | Reserved for extensions; must be NULL |
| `host_device` | `ID3D12Device *` | NULL | Your device; selects the adapter LUID and exports resources |
| `agility_sdk_version` | u32 | `0` | Required; exactly the Agility SDK version your device uses |
| `agility_sdk_path` | UTF-8 string | NULL | Required absolute directory containing `D3D12Core.dll`, ending with `/` or `\` |
| `flags` | bitmask | `0` | `OIDNW_GPU_DX12_FP32`, `OIDNW_GPU_DX12_NO_SUBGROUPS`, `OIDNW_GPU_DX12_DEBUG`, `OIDNW_GPU_DX12_GPU_VALIDATION` (requires DEBUG) |
| `max_pending_submissions` | u32 | `64` | 1..=4096 |

- Image buffers: committed, DEFAULT heap, `D3D12_HEAP_FLAG_SHARED`, ROW_MAJOR, `ALLOW_UNORDERED_ACCESS`. Syncs: shared fences from the same device. Placed and reserved resources, textures and cross-adapter heaps are unsupported.
- Hand buffers over in the COMMON state: use DIRECT command-list promotion and decay, or transition back to COMMON before signaling. Fences order access; they do not change resource states.
- FP16 needs DXC with Shader Model 6.2 or later: ship `dxcompiler.dll` and its matching `dxil.dll`.
- The library opens and closes its own NT handles. Device removal, including a fence value of `UINT64_MAX`, quarantines the context.

Build and run from an x64 Visual Studio developer PowerShell with the Agility SDK headers:

```powershell
$env:WGPU_DX12_AGILITY_SDK_PATH = 'C:/SDKs/Agility/build/native/bin/x64/'
$env:WGPU_DX12_AGILITY_SDK_VERSION = '619'
npm run native:build
cmake -S .native-build/package/examples -B .native-build/dx12-examples `
  -DCMAKE_PREFIX_PATH="$PWD/.native-build/package" -DOIDNW_BUILD_DX12_GPU_EXAMPLE=ON `
  -DDX12_AGILITY_INCLUDE_DIR='C:/SDKs/Agility/build/native/include'
cmake --build .native-build/dx12-examples --config Release --target dx12_gpu
.native-build/dx12-examples/Release/dx12_gpu.exe weights/rt_hdr_alb_nrm_small.tza .native-build/package/examples/shaders 1000 16 0 0 0 0
```

The Windows hosts' arguments after the shader directory are frame count, image edge, tile edge, FP32 (`0`/`1`), shared queue (Vulkan; `0` on DX12) and shader runtime checks (`0`/`1`). They select the adapter whose name contains `OIDNW_INTEROP_ADAPTER` (default `RTX 4060`). The Vulkan host additionally needs Vulkan 1.3 and three graphics/compute queues in one family; this is an example requirement, not a library one. Set `WGPU_GPU_BASED_VALIDATION=1` to enable DX12 GPU validation. Known Naga workgroup/ArrayStride diagnostics are counted separately from other validation errors; a nonzero known count is not clean shader validation. These runs include deliberate delays and are not benchmarks.

## CUDA on Windows

Unreleased, like the GPU C API. CUDA renderers use the optional C++17 adapter in [`oidn_webgpu_cuda.hpp`](include/oidn_webgpu_cuda.hpp) (CMake target `oidn_webgpu::cuda`). It builds the DX12 bridge for you and orders denoising on your CUDA streams:

```cpp
#include <oidn_webgpu_cuda.hpp>

cudaSetDevice(renderer_device);
oidnw::cuda::options options;
options.width = width;
options.height = height;
oidnw::cuda::adapter denoiser(options, tza_bytes, tza_size);

// Copy mode: oidnw::cuda::image views of your cudaMalloc buffers; four GPU copies per frame.
auto task = denoiser.enqueue_copy({color_view, albedo_view, normal_view, output_view}, stream, exposure);
if (task.result) consume<<<grid, block, 0, stream>>>(output_pointer);
```

`options.hdr` and `options.transfer` select HDR/LDR and PU/log transfer; debug layers default to off. `resize(width, height)` reuses devices and weights after task/lease release and consumer completion.

The shared-image mode (`acquire` / `enqueue_shared` / `release`) renders straight into adapter-owned buffers with no copies. CUDA is needed only when you build the adapter. The [CUDA guide](examples/gpu-integration.md#windows-cuda-adapter) covers both modes, build, deployment and shutdown.

## CPU C filter

For tools and tests whose images are in CPU memory. Each call uploads the inputs, denoises, reads back the output and blocks, so prefer the [GPU C API](#gpu-c-api) in a renderer.

```c
#include "oidn_webgpu.h"

char error[1024];
oidnw_device *device;
oidnw_device_create(OIDNW_BACKEND_AUTO, &device, error, sizeof error);

oidnw_filter_desc desc;
oidnw_filter_desc_default(&desc);
oidnw_filter *filter;
oidnw_filter_create(device, tza, tza_size, &desc, &filter, error, sizeof error);

oidnw_filter_execute(filter, width, height, color, albedo, normal, output, error, sizeof error);

oidnw_filter_destroy(filter);
oidnw_device_destroy(device);
```

Check every `int32_t` status (`OIDNW_OK` = 0); [`examples/denoise.c`](examples/denoise.c) shows full error handling. Link `oidn_webgpu::oidn_webgpu`.

- Images are tightly packed RGBA32F arrays of one size. Pass `NULL` for albedo or normal when the model does not read them (`oidnw_filter_input_count`). Output may alias an input.
- `oidnw_filter_set_input_scale(filter, scale, ...)` changes exposure for later executions without rebuilding the workspace. The scale must be positive and finite, including for LDR where it is ignored.
- The filter caches a workspace for its last image size. `oidnw_filter_prepare(filter, w, h, ...)` optionally prepares a size ahead of time.
- Handles are not thread-safe: concurrent calls on one filter return `OIDNW_ERROR_BUSY`. One device may serve filters on different threads. Destroy filters before their device.
- `OIDNW_ERROR_GPU` means execution or readback failed: destroy and recreate the filter.

`oidnw_filter_desc`, from `oidnw_filter_desc_default`:

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `next` | pointer | NULL | Reserved for extensions; must be NULL |
| `hdr` | u32 | nonzero | `0` for LDR weights |
| `transfer` | enum | `OIDNW_TRANSFER_PU` | `OIDNW_TRANSFER_LOG` for RTLightmap HDR weights |
| `quality` | enum | `OIDNW_QUALITY_HIGH` | `FAST` uses f16 accumulation in subgroup-matrix kernels |
| `kernel` | enum | `OIDNW_KERNEL_AUTO` | Subgroup-matrix kernels with cooperative matrices, otherwise implicit GEMM |
| `max_tile_size` | u32 | `0` | `0` = whole image; otherwise rounded down to a multiple of 16 |
| `input_scale` | float | `1.0` | Positive finite scale applied to HDR color before the transfer; zero is rejected |
| `flags` | bitmask | `0` | `OIDNW_FILTER_RUNTIME_CHECKS` enables shader bounds checks (off by default); unknown bits return `OIDNW_ERROR_INVALID_ARGUMENT` |

Use `OIDNW_BACKEND_VULKAN` for cooperative-matrix kernels on Windows and Linux. `oidnw_device_info` reports the adapter name, backend and enabled `OIDNW_FEATURE_*` bits. The [header](include/oidn_webgpu.h) documents every option and lifetime rule.

### Versioning

`oidnw_api_version()` returns `major << 16 | minor << 8 | patch`; `OIDNW_API_VERSION` in the header is 3.0.0 (`0x00030000`). It covers the CPU filter and GPU C API.

- Before calling the API, dynamic loaders must check `OIDNW_API_VERSION_IS_COMPATIBLE(oidnw_api_version())`. Compatibility requires the library's major to equal the header's major and its minor to be at least the header's minor; patch is ignored. For example, a 3.0 consumer accepts a 3.1 library, while a 3.1 consumer rejects a 3.0 library. The C++ wrappers perform this check.
- `OIDNW_API_VERSION_MAJOR`, `OIDNW_API_VERSION_MINOR` and `OIDNW_API_VERSION_PATCH` expose the header version components.
- Minor versions may add functions, enum values, flag bits and extension structures linked through `next`. They cannot change existing structure layouts. Removals and changes to existing layouts require a major version bump.
- ABI 3.0.0 replaces 1.0.0: `oidnw_filter_desc.size` is removed. Descriptors, `oidnw_gpu_limits` and `oidnw_gpu_prepared_info` start with `next`, which must be NULL. Rebuild 1.x consumers and use the [CPU filter options](#cpu-c-filter) described above.

### Core planner

Hosts with their own non-wgpu GPU executor can link the planner-only static library (`oidn_webgpu::oidn_webgpu_core`) and replay its programs; see [`include/oidn_webgpu_core.h`](include/oidn_webgpu_core.h) and [`examples/plan.c`](examples/plan.c). Rust and wgpu hosts should use `Denoiser`. On Windows, use the DLL CRT (`/MD`); no MinGW installation or extra runtime DLL is needed.

## Image contract

All APIs use the same image conventions as OIDN.

- **Layout:** tightly packed linear RGBA32F; no textures or row strides. GPU APIs bind exactly `width * height * 16` bytes at an offset aligned to the device's storage-buffer offset alignment. The output must be a different buffer from every input (the CPU filter allows aliasing).
- **Which images:** exactly the images the model reads (`input_count`): 1 = color, 2 = color + albedo, 3 = color + albedo + normal. Unused GPU bindings are `{NULL, 0}`.
- **Dimensions:** 1..=16384 per side, within the device's buffer limits. Tiling reduces only the intermediate workspace.
- **Color:** linear RGB; LDR in [0, 1], HDR nonnegative. Pass the averaged color, not a sample sum. Output keeps the color's alpha.
- **Albedo:** linear RGB in [0, 1]. Alpha is ignored.
- **Normal:** world- or view-space shading normals with signed components in [-1, 1]; unit length is not required. Remap normals stored in [0, 1] first ([OIDN conventions](https://www.openimagedenoise.org/documentation.html#normals)). The browser package takes the [0, 1] form instead. Alpha is ignored.
- **Sanitizing:** NaN becomes 0 and values are clamped to the ranges above (HDR color to [0, FLT_MAX] after `input_scale`). Output is never NaN or negative, LDR output is at most 1, and extreme HDR input (for example +inf) can produce +inf.

If your renderer uses textures, copy them to buffers on the GPU and back; those passes add time and memory.

## Precompiled packages

Published versions are on [GitHub Releases](https://github.com/pissang/oidn-webgpu/releases) under `native-v<VERSION>`, with packages, raw core archives, a `.crate` and `SHA256SUMS`. PR builds are 30-day artifacts of the [Native packages workflow](https://github.com/pissang/oidn-webgpu/actions/workflows/native.yml) (GitHub sign-in required); extract the artifact, then its `.tar.gz`.

| Target | Rust triple | Built on |
| --- | --- | --- |
| macOS ARM64 | `aarch64-apple-darwin` | macOS 15 |
| macOS x86_64 | `x86_64-apple-darwin` | macOS 15 |
| Windows x86_64 | `x86_64-pc-windows-msvc` | Windows Server 2022, MSVC |
| Linux x86_64 | `x86_64-unknown-linux-gnu` | Ubuntu 24.04, glibc |

Windows GNU, Linux musl and other architectures are not provided, and Windows GNU source builds are not validated. `manifest.json` records the build OS and, on Linux, the glibc version; older systems are not guaranteed to work.

| Package path | Contents |
| --- | --- |
| `include/oidn_webgpu_gpu.h`, `oidn_webgpu_{metal,vulkan,dx12}.h` | GPU C API and platform imports |
| `include/oidn_webgpu_gpu.hpp`, `oidn_webgpu_{metal,vulkan,dx12}.hpp` | C++17 GPU wrappers |
| `include/oidn_webgpu.h` | CPU C filter and API version |
| `include/oidn_webgpu_core.h` | Core planner |
| `include/oidn_webgpu_cuda.hpp`, `examples/cuda-adapter/` | Optional Windows CUDA adapter header and source |
| `lib/liboidn_webgpu.dylib`, `.so`, or `oidn_webgpu.dll` + `oidn_webgpu.dll.lib` | Shared library |
| `lib/liboidn_webgpu_core.a` or `oidn_webgpu_core.lib` | Core planner static library; also the Rust core override |
| `lib/cmake/oidn_webgpu/` | CMake config and imported targets |
| `examples/gpu-integration.md` | GPU C API walkthrough and CUDA guide |
| `examples/metal_gpu_cpp.mm`, `metal_gpu.m`, `vulkan_gpu.cpp`, `dx12_gpu.cpp`, `cuda_dx12_gpu.cu`, `gpu_cpp_lifecycle.hpp`, `shaders/` | GPU hosts and their shaders |
| `examples/denoise.c`, `plan.c`, `host_buffers.rs` | CPU filter, planner and Rust examples |
| `examples/validate_windows.c`, `validate.rs`, `rust-app/` | [Validation tools](#validate-a-build) |
| `examples/CMakeLists.txt` | Builds the examples against the packaged targets |
| `bin/oidn-webgpu-bench` | Network benchmark |
| `manifest.json`, `SHA256SUMS` | Source commit, target, compiler versions, file hashes |

Release package names contain the crate version and target triple, for example `oidn-webgpu-native-0.1.1-aarch64-apple-darwin.tar.gz`; `manifest.json` records the full source commit as `source.commit`. CI and local archive names also include the commit, and archives built from modified sources end in `-dirty`.

To use a package's core from Rust, pin the Cargo `rev` to `source.commit` and set `OIDN_WEBGPU_CORE_ARCHIVE` to its core library. Cargo still compiles the Rust layer; precompiled `rlib`s are not provided.

## Validate a build

### Tested configurations

| Platform | GPU | Checked |
| --- | --- | --- |
| macOS | Apple M4 Max | Metal cooperative matrices: Rust API, CPU filter and GPU C API |
| Windows | NVIDIA RTX 4060 Laptop | Vulkan: cooperative matrices in the Rust API and CPU filter; GPU C API verified only with implicit GEMM, cooperative matrices unverified. DX12: implicit GEMM in the Rust API, CPU filter and GPU C API |
| Linux | — | Builds only; GPU execution not verified |

The Windows CI package at source `572efc0` was exercised on an RTX 4060 Laptop with NVIDIA 617.42. Vulkan output with and without runtime checks matched byte-for-byte for Small/Base/Large at 720p, 1080p, 1440p, partial workgroups and 128-pixel tiles, and real 1024x1024 HDR images matched official OIDN 2.5.1 CUDA numerically with the same weights. DX12 implicit GEMM passed with runtime checks disabled; DX12 with runtime checks is not validated.

### Validation tools

Both tools generate deterministic HDR inputs (or load `color.f32`, `albedo.f32`, `normal.f32`), check finite, nonnegative RGB, preserved alpha and byte-identical repeated output, and print JSON with the adapter and timings.

**Rust** ([`examples/rust-app`](examples/rust-app/Cargo.toml), source [`examples/validate.rs`](examples/validate.rs)). Measured executions exclude uploads, preparation and readback. From an x64 Visual Studio developer PowerShell, using a package's core so no Node is needed:

```powershell
rustup toolchain install stable-x86_64-pc-windows-msvc --profile minimal
$package = (Resolve-Path .native-build/package).Path
$env:OIDN_WEBGPU_CORE_ARCHIVE = Join-Path $package 'lib/oidn_webgpu_core.lib'
cargo +stable-x86_64-pc-windows-msvc run --release --target x86_64-pc-windows-msvc --manifest-path "$package/examples/rust-app/Cargo.toml" -- --weights weights/rt_hdr_calb_cnrm_large.tza --output .native-build/rust-output.f32 --width 257 --height 259 --backend vulkan --matrix
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--weights PATH` | required | HDR TZA weights |
| `--output PATH` | no file | Save RGBA32F output |
| `--width N`, `--height N` | `256`, `256` | 1..=16384 |
| `--backend NAME` | `auto` | `auto`, `vulkan`, `dx12` or `metal` |
| `--matrix` | off | Require cooperative matrices; use Vulkan on Windows |
| `--unchecked` | off | Disable shader runtime checks |
| `--tile N` | `0` | Maximum output tile edge |
| `--warmup N`, `--runs N` | `5`, `21` | Warm-ups 0..=10000; measured executions 1..=10000 |
| `--input-dir DIR` | generated | Load raw RGBA32F inputs |
| `--inspect-model PATH` | — | Load the model without a GPU (used by CI) |

The packaged manifest pins a Git dependency to `source.commit`; in a source checkout it uses the local crate. Run the same source as a crate example with `cargo run --release --manifest-path native/Cargo.toml --example validate -- ...`.

**Windows C** ([`examples/validate_windows.c`](examples/validate_windows.c)) needs only MSVC, the package and weights. Timings include uploads and readback.

```powershell
$package = (Resolve-Path .native-build/package).Path
$work = Join-Path $PWD '.native-build/windows-validation'
New-Item -ItemType Directory -Force $work | Out-Null
cl.exe /nologo /std:c11 /O2 /MD "/I$package/include" "$package/examples/validate_windows.c" "/Fo$work/validate_windows.obj" "/Fe$work/validate_windows.exe" /link "$package/lib/oidn_webgpu.dll.lib"
Copy-Item "$package/lib/oidn_webgpu.dll" $work
& "$work/validate_windows.exe" weights/rt_hdr_calb_cnrm_large.tza 257 259 2 3 0 0 5 21 "$work/output.f32"
```

Arguments: weights, width, height, backend (`2` Vulkan, `3` DX12), kernel (`0` auto, `1` implicit GEMM, `2` direct, `3` subgroup matrix), runtime checks (`0`/`1`), maximum tile size (`0` = whole image), warm-ups, timed samples, output path and an optional input directory. Set `OIDNW_EXPECT_ADAPTER` to a name substring such as `RTX 4060` to fail if another GPU is selected.

The packaged `examples/CMakeLists.txt` also builds `denoise`, `plan` and, on Windows, `validate_windows`, and installs them with their shared library:

```bash
cmake -S .native-build/package/examples -B .native-build/cmake-examples \
  -DCMAKE_PREFIX_PATH="$PWD/.native-build/package" -DCMAKE_BUILD_TYPE=Release
cmake --build .native-build/cmake-examples --config Release
cmake --install .native-build/cmake-examples --config Release --prefix "$PWD/.native-build/cmake-app"
.native-build/cmake-app/bin/plan weights/rt_hdr_alb_nrm_small.tza 16 16
```

## Build configuration

Environment variables read by `cargo build`:

| Variable | Effect |
| --- | --- |
| `OIDN_WEBGPU_CORE_ARCHIVE` | Absolute path to a prebuilt `liboidn_webgpu_core.a` / `oidn_webgpu_core.lib`. Skips Node and scriptc. Must match the target OS, architecture, ABI and source revision; a registry crate also checks it against the bundled manifest. Useful for cross compilation |
| `OIDN_WEBGPU_SCRIPTC` | Path to an existing scriptc compiler; skips installing it |
| `OIDN_WEBGPU_NODE` | Node executable used for the core build |

Without `OIDN_WEBGPU_CORE_ARCHIVE`, a registry crate selects and verifies its bundled core for Cargo's target triple, including supported cross-compilation targets. A Git or path checkout builds the core automatically for native builds only (host = target). Installed build tools stay in Cargo's build directory.

## Performance

The [0.6.0 release notes](https://github.com/pissang/oidn-webgpu/blob/main/CHANGELOG.md#060---2026-10-25) compare native denoising with official OIDN 2.5.1 on an Apple M4 Max (Metal) and an NVIDIA RTX 4060 Laptop (Vulkan). Those timings cover one whole image in GPU buffers, including preprocessing and output conversion, with runtime checks disabled; they exclude uploads and readback.

On the M4 Max, the CPU C filter takes about 21 ms per 1024x1024 image with `rt_hdr_alb_nrm.tza`, including three input uploads and the output readback. First preparation includes planning and shader compilation; there is no on-disk pipeline cache.

## Licensing

Project-owned code is licensed under MIT (see `LICENSE`). The published crate
also bundles precompiled cores with separately licensed components:

| Component | License | Distributed text |
| --- | --- | --- |
| scriptc runtime | Apache-2.0 | `licenses/scriptc/runtime-LICENSE.txt` |
| V8-derived radix number conversion | BSD-3-Clause | `licenses/scriptc/v8-number-radix-LICENSE.txt` |
| Ryū decimal conversion | BSL-1.0 | Object code only; this license exempts that form from reproducing its text |

The crate declares `MIT AND Apache-2.0 AND BSD-3-Clause AND BSL-1.0` to describe
these combined obligations. Each license applies to its respective component;
this expression does not relicense project-owned code. `NOTICE` records the
upstream version and component origins.
