# oidn-webgpu native

Experimental native GPU denoising through [wgpu](https://wgpu.rs) on Metal, Vulkan or Direct3D 12. It runs the same WGSL kernels as the browser package.

**Status:** the API and results may change between versions. Denoised output has been checked on macOS with Apple silicon (Metal) and Windows with an NVIDIA RTX 4060 Laptop (Vulkan cooperative matrices and DX12 implicit GEMM). See the [Windows validation example](#validate-a-windows-package) for the tested runtime-check settings. CI builds and checks packages but does not run them on a GPU.

- [Choose an API](#choose-an-api)
- [Rust API](#rust-api)
- [C API](#c-api)
- [Precompiled packages](#precompiled-packages)
- [Build configuration](#build-configuration)
- [Performance](#performance)
- [Licensing](#licensing)

## Choose an API

| API | Use when | GPU device | Images | Header / type |
| --- | --- | --- | --- | --- |
| Rust `Denoiser` | Your Rust host already renders with wgpu `=30.0.1` | The host's | Host `wgpu::Buffer`s, RGBA f32 | `oidn_webgpu::Denoiser` |
| C filter | Any language; images are in CPU memory | Created by the library | CPU arrays, RGBA f32 | [`include/oidn_webgpu.h`](include/oidn_webgpu.h) |
| C core planner | A non-wgpu host that executes GPU programs itself | The host's | Host-defined | [`include/oidn_webgpu_core.h`](include/oidn_webgpu_core.h) |

Weights are not bundled. Choose a `.tza` file from the [weights table](https://github.com/pissang/oidn-webgpu#choose-weights) and ship it with your application.

## Rust API

### Install

```toml
[dependencies]
oidn-webgpu = { package = "oidn-webgpu-native", version = "0.1.1" }
# Git source:
# oidn-webgpu = { package = "oidn-webgpu-native", git = "https://github.com/pissang/oidn-webgpu.git", rev = "<commit>" }
# Local checkout:
# oidn-webgpu = { package = "oidn-webgpu-native", path = "../oidn-webgpu/native" }
```

- The registry crate bundles cores for the four [supported targets](#precompiled-packages) only; on Windows the supported target is `x86_64-pc-windows-msvc`. Run `rustc -vV` to see the toolchain's default target as `host`; an explicit Cargo `--target` overrides it. Building the registry crate needs Rust and a platform linker, with no Node, npm, Zig or build-time GitHub download.
- A source build needs stable Rust, a C toolchain, Node 22+ and npm; on Windows also Zig 0.16.0 on `PATH`, which scriptc uses to link the core. `cargo build` installs the pinned scriptc compiler (the first build needs the npm registry) and builds the core static library. To skip Node, use a [precompiled core](#build-configuration).

### wgpu version

The crate pins `wgpu = "=30.0.1"`. A host that passes `Device`, `Buffer` or `CommandEncoder` must resolve to the same wgpu crate and source; a different release or a git fork produces incompatible types. A precompiled core does not change this. Use the re-export to avoid a mismatch:

```rust
use oidn_webgpu::wgpu;
```

### Usage

```rust
use oidn_webgpu::{DenoiseBuffers, DenoiseParams, Denoiser, DenoiserOptions, Model};

// Once.
let model = Model::from_tza(&weights)?;
let mut denoiser = Denoiser::new(&device, &adapter, &model, DenoiserOptions::default())?;

// Per image size (may block for shader compilation).
let mut prepared = denoiser.prepare([width, height], DenoiseBuffers {
    color: (&color).into(),
    albedo: Some((&albedo).into()),
    normal: Some((&normal).into()),
    output: (&output).into(),
})?;

// Per frame.
let mut encoder = device.create_command_encoder(&Default::default());
// ... the host's render pass writes color/albedo/normal ...
prepared.encode(&mut encoder, DenoiseParams { input_scale: exposure })?;
// ... a display pass reads output ...
queue.submit([encoder.finish()]);
```

| Step | Call | Cost and behavior |
| --- | --- | --- |
| Load | `Model::from_tza(&[u8])` | Validates the TZA file on the CPU. Clones share CPU weights and per-device GPU weights. |
| Bind | `Denoiser::new(&device, &adapter, &model, options)` | Cheap. `adapter` must be the device's adapter. |
| Prepare | `denoiser.prepare([w, h], buffers)` | May block for shader compilation. Allocates a workspace for that size. Call again when the size changes; matching pipelines and weights are reused. |
| Encode | `prepared.encode(&mut encoder, params)` | Appends commands only: no submit, wait or readback. |
| Rebind | `prepared.rebind(buffers)` | Swaps image buffers at the same size, keeping pipelines and workspace. Already encoded commands keep their old bindings. |
| Inspect | `prepared.info()` | `DenoiseInfo`: precision, `kernels` actually used (including fallbacks), `tile_count`, `workspace_bytes`, `shared_weight_bytes`, `planning_ms`, `prepare_ms`, `runtime_checks`. |

[`examples/host_buffers.rs`](examples/host_buffers.rs) is a complete program: it creates a device, renders inputs on the GPU, denoises in the same encoder and writes the result as tightly packed little-endian RGBA f32. From the repository root:

```bash
cargo run --release --manifest-path native/Cargo.toml --example host_buffers -- weights/rt_hdr_alb_nrm_small.tza output.f32 256 192 128
```

Raw Metal, Vulkan and DX12 interoperability is being tested in source-checkout examples before defining a GPU C ABI. Metal and Vulkan import a host device and buffers; DX12 opens shared buffers and fences on a separate library device. These examples use tightly packed RGBA32F buffers with fixed input/output roles and separate output allocations. See the [interop validation instructions and limitations](https://github.com/pissang/oidn-webgpu/blob/main/docs/development.md#native-windows-interoperability-prototypes). The C filter API continues to accept CPU images.

### Standalone Rust application

[`examples/rust-app/Cargo.toml`](examples/rust-app/Cargo.toml) is an independent Cargo project using [`examples/validate.rs`](examples/validate.rs). It creates a wgpu device and image buffers, denoises generated HDR gradients or raw input files, checks the complete output, and optionally saves RGBA32F. JSON output includes the actual backend, kernels, runtime checks, initialization time, individual GPU execution times and median latency. Uploads, preparation, readback and output validation are excluded from the measured executions.

From the repository root in an x64 Visual Studio Native Tools PowerShell, use a downloaded Windows package's core:

```powershell
rustup toolchain install stable-x86_64-pc-windows-msvc --profile minimal
$package = (Resolve-Path .native-build/package).Path
$env:OIDN_WEBGPU_CORE_ARCHIVE = Join-Path $package 'lib/oidn_webgpu_core.lib'
$env:OIDN_WEBGPU_NODE = 'C:/node-must-not-run.exe'
cargo +stable-x86_64-pc-windows-msvc run --release --target x86_64-pc-windows-msvc --manifest-path "$package/examples/rust-app/Cargo.toml" -- --weights weights/rt_hdr_calb_cnrm_large.tza --output .native-build/rust-output.f32 --width 257 --height 259 --backend vulkan --matrix
```

`$package` may point to any extracted download containing this example. Its Cargo manifest pins a Git dependency to `manifest.json`'s `source.commit`. Windows packages use MSVC, so the command explicitly selects the matching Rust toolchain and target. In a source checkout, `native/examples/rust-app/Cargo.toml` instead depends on the local native crate; its source revision must match a supplied core. Cargo fetches and compiles the Rust wrapper and dependencies, while `OIDN_WEBGPU_CORE_ARCHIVE` skips Node, npm, Zig and rebuilding the planner. Without that override, a source build uses the [normal build requirements](#install).

| Option | Default | Meaning |
| --- | --- | --- |
| `--weights PATH` | required | TZA weights; choose HDR models for this example |
| `--output PATH` | no file | Save tightly packed little-endian RGBA32F |
| `--width N`, `--height N` | `256`, `256` | Image dimensions, 1..=16384 and within device limits |
| `--backend NAME` | `auto` | `auto`, `vulkan`, `dx12` or `metal` |
| `--matrix` | off | Enable experimental cooperative matrices; fails if unavailable. Use Vulkan on Windows |
| `--unchecked` | off | Disable shader runtime checks; the default uses `Denoiser::new` |
| `--tile N` | `0` | Maximum output tile edge, as in `DenoiserOptions` |
| `--warmup N`, `--runs N` | `5`, `21` | Warm-ups 0..=10000; measured executions 1..=10000 |
| `--input-dir DIR` | generated gradients | Load `color.f32`, `albedo.f32`, `normal.f32`; same RGBA32F layout and image conventions as the API |

The example checks finite, nonnegative RGB, preserved input alpha, and byte-identical output across three extra executions. Readbacks for these checks occur after timing. `--inspect-model WEIGHTS.tza` loads the model without requesting a GPU; CI uses this mode to compile and link the actual application against each packaged core. `cargo run --release --manifest-path native/Cargo.toml --example validate -- ...` runs the same source as a crate example.

### Device setup

- Before requesting a device, add `Denoiser::required_features()`, `Denoiser::recommended_features(&adapter)` and `Denoiser::required_limits()`. Large images may need higher buffer limits.
- The library uses only the features and limits enabled on an existing device; it cannot enable more later.
- Precision is FP16 when the device enables `SHADER_F16`, otherwise FP32. Image buffers are always f32.
- On DX12, exposing `SHADER_F16` also requires DXC with Shader Model 6.2 or later. Keep `dxcompiler.dll` and its matching `dxil.dll` available to the host; wgpu's FXC fallback uses Shader Model 5.1 and exposes FP32 only.
- Kernels: cooperative-matrix kernels when the host enables `EXPERIMENTAL_COOPERATIVE_MATRIX` (which also requires accepting wgpu's experimental feature contract), otherwise implicit GEMM. On Windows and Linux the cooperative-matrix kernels need Vulkan; DX12 uses implicit GEMM.

### Image contract

- **Layout:** tightly packed RGBA f32, `STORAGE` usage, unmapped, on the denoiser's device. Textures, row strides and in-place output are not supported.
- **Range binding:** `ImageBuffer { buffer, offset }` binds exactly `width * height * 16` bytes from `offset`; `(&buffer).into()` uses offset 0. The offset must be a multiple of `min_storage_buffer_offset_alignment` and the range must fit in the buffer. The buffer itself may be larger than the binding limit as long as the bound range fits.
- **Aliasing:** input ranges may share a buffer if they do not overlap. Output must be a different buffer from every input, because wgpu tracks storage usage per buffer.
- **Which images:** supply exactly the images `Model::input_count()` reports: 1 = color, 2 = color + albedo, 3 = color + albedo + normal.
- **Color:** linear RGB. LDR in [0, 1]; HDR nonnegative. Pass the averaged color, not an unnormalized sample sum. Output keeps the color's alpha.
- **Albedo:** linear RGB in [0, 1]. Alpha is ignored.
- **Normal:** world- or view-space shading normals with signed components in [-1, 1]; unit length is not required. Like OIDN, the library clamps them to [-1, 1] and encodes them as `n * 0.5 + 0.5` for the network. Remap normals stored in [0, 1] first ([OIDN conventions](https://www.openimagedenoise.org/documentation.html#normals)); the browser package takes the [0, 1] form instead. Alpha is ignored.
- **Sanitizing:** like OIDN, NaN becomes 0 and values are clamped to the ranges above (HDR color to [0, FLT_MAX] after `input_scale`). The network output is sanitized before the inverse transfer: output is never NaN or negative, LDR output is at most 1, and extreme HDR input (for example +inf) can produce +inf.

### Options

`DenoiserOptions` is fixed per `Denoiser`. The struct is `#[non_exhaustive]`: start from `default()` and assign fields.

| Field | Default | Meaning |
| --- | --- | --- |
| `hdr` | `true` | HDR color. `false` for LDR weights. |
| `transfer` | `Transfer::Pu` | `Transfer::Log` for RTLightmap HDR weights. |
| `max_tile_size` | `0` | Output tile edge, rounded down to a multiple of 16. `0` runs the whole image; otherwise 16..=16384. Tiling reduces workspace memory, not image buffers. |

`DenoiseParams` is passed to every `encode`. It may change every frame without preparing again, and several executions in one command buffer keep their own values.

| Field | Default | Meaning |
| --- | --- | --- |
| `input_scale` | `1.0` | Positive, finite. Multiplies HDR color before the transfer and is undone on output. Ignored for LDR. Exposure is not estimated automatically. |

Image dimensions must be in 1..=16384 and fit the device's buffer limits.

### Memory and lifetime

- Each prepared instance owns its workspace. Prepare separate instances for independent workloads.
- Denoisers made from the same `Model` (or its clones) share immutable GPU weights per device. Loading the same TZA again as a new `Model` creates a separate cache. Active caches also keep packed weights in CPU memory.
- When estimating memory, count `shared_weight_bytes` once, not per instance. Reported sizes exclude driver overhead and small transient uploads.
- Prepared instances may outlive the `Denoiser` and keep their image buffer handles. Do not destroy buffers that pending GPU work still uses.
- Submit executions of one prepared instance in encode order on the device's queue.

### Errors

- `Result<T>` uses the typed `Error`: `InvalidArgument`, `Model`, `UnsupportedDevice`, `Gpu`, `BufferTooSmall { image, required, available }` and `LimitExceeded { resource, limit, required, available }`.
- `prepare` and `rebind` return errors for invalid images, model failures, insufficient limits and captured GPU validation or allocation failures. `encode` returns invalid `DenoiseParams` before recording any command.
- A successful `encode` does not mean the GPU succeeded. Execution errors and device loss go through the host's wgpu error handling.
- The low-level experimental `Network` API uses `PlanResult<T>` with string errors.

### Runtime checks

`Denoiser::new` compiles shaders with bounds checks and workgroup initialization. `unsafe Denoiser::new_unchecked` disables them and can be faster; the difference depends on the GPU, model and convolution path. The caller must uphold its documented safety contract.

## C API

### Get the library

Download a [precompiled package](#precompiled-packages), or build one from the repository root (needs Node 22+, npm, CMake 3.20+, stable Rust and a C toolchain; on Windows also Zig 0.16.0):

```bash
npm run native:build
```

The package is written to `.native-build/package`; pass `-- --output DIR` to choose another directory.

### Usage

```c
#include "oidn_webgpu.h"

char error[1024];
oidnw_device *device;
oidnw_device_create(OIDNW_BACKEND_AUTO, &device, error, sizeof error);

oidnw_filter_desc desc;
oidnw_filter_desc_default(&desc);
oidnw_filter *filter;
oidnw_filter_create(device, tza, tza_size, &desc, &filter, error, sizeof error);

oidnw_filter_prepare(filter, width, height, error, sizeof error); /* optional warm-up */
oidnw_filter_execute(filter, width, height, color, albedo, normal, output, error, sizeof error);

oidnw_filter_destroy(filter);
oidnw_device_destroy(device);
```

Check every `int32_t` status (`OIDNW_OK` = 0); [`examples/denoise.c`](examples/denoise.c) shows full error handling. Link with `-loidn_webgpu` and ship the shared library and your TZA weights.

### CMake

Precompiled packages provide a CMake 3.20+ config package. Set `CMAKE_PREFIX_PATH` to the extracted package root, then use:

```cmake
cmake_minimum_required(VERSION 3.20)
project(my_app LANGUAGES C)

find_package(oidn_webgpu CONFIG REQUIRED)
add_executable(my_app main.c)
target_link_libraries(my_app PRIVATE oidn_webgpu::oidn_webgpu)
```

| Target | Links |
| --- | --- |
| `oidn_webgpu::oidn_webgpu` | C filter shared library and public headers |
| `oidn_webgpu::oidn_webgpu_core` | Planner-only static library, public headers and required system libraries |

The package can be moved without regenerating its CMake files. An optional version in `find_package` uses the native crate version; compatible patch versions share the same major and minor numbers. The headers and libraries must match the application's target OS and CPU architecture.

Windows applications must ship the DLL beside the executable or make it available on `PATH`. CMake can copy it after building:

```cmake
if(WIN32)
  set_property(TARGET my_app PROPERTY MSVC_RUNTIME_LIBRARY MultiThreadedDLL)
  add_custom_command(TARGET my_app POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
      "$<TARGET_FILE:oidn_webgpu::oidn_webgpu>" "$<TARGET_FILE_DIR:my_app>"
    VERBATIM)
endif()
```

The package's [`examples/CMakeLists.txt`](examples/CMakeLists.txt) builds `denoise`, `plan` and, on Windows, `validate_windows`. Its install step places executables in `bin/` and their shared library in `lib/` (Windows: `bin/`). Move the installed directory as a unit when deploying the examples. From the repository root after building a package:

```bash
cmake -S .native-build/package/examples -B .native-build/cmake-examples \
  -DCMAKE_PREFIX_PATH="$PWD/.native-build/package" -DCMAKE_BUILD_TYPE=Release
cmake --build .native-build/cmake-examples --config Release
cmake --install .native-build/cmake-examples --config Release --prefix "$PWD/.native-build/cmake-app"
.native-build/cmake-app/bin/plan weights/rt_hdr_alb_nrm_small.tza 16 16
```

### Behavior

- Images are tightly packed RGBA f32 in CPU memory, all the same size. Pass `NULL` for albedo or normal when the model does not read them (`oidnw_filter_input_count`). Output keeps the color's alpha and may alias an input.
- Color, albedo and normal follow the [Rust image contract](#image-contract): albedo in [0, 1], normals signed in [-1, 1].
- `oidnw_filter_execute` blocks until the output is written.
- The filter caches a workspace for its last image size. A new size creates a new workspace and reuses weights and matching pipelines.
- `oidnw_filter_prepare` is optional. It allocates and compiles for a size without denoising; `execute` prepares automatically when needed.
- Handles are not thread-safe. Concurrent prepare/execute on one filter returns `OIDNW_ERROR_BUSY`. One device may be shared by filters on different threads. Destroy filters before their device.
- `OIDNW_ERROR_GPU` means execution or readback failed: destroy and recreate the filter.

### `oidnw_filter_desc`

Initialize with `oidnw_filter_desc_default`, then override fields.

| Field | Default | Meaning |
| --- | --- | --- |
| `hdr` | nonzero | HDR color. Set 0 for LDR weights. |
| `transfer` | `OIDNW_TRANSFER_PU` | `OIDNW_TRANSFER_LOG` for RTLightmap HDR weights. |
| `quality` | `OIDNW_QUALITY_HIGH` | Accumulation precision. `FAST` uses f16 accumulation in subgroup-matrix kernels. Model size comes from the weights. |
| `kernel` | `OIDNW_KERNEL_AUTO` | Subgroup-matrix kernels when the device has cooperative matrices, otherwise implicit GEMM. |
| `max_tile_size` | `0` | Output tile edge, rounded down to a multiple of 16; 0 = whole image. |
| `input_scale` | `1.0` | Scale applied to HDR color before the transfer. |
| `flags` | `0` | `OIDNW_FILTER_RUNTIME_CHECKS` enables shader bounds checks. The C API runs unchecked by default. |

`OIDNW_BACKEND_VULKAN` is required for cooperative-matrix kernels on Windows and Linux. `oidnw_device_info` reports the adapter name, backend and enabled `OIDNW_FEATURE_*` bits. The [header](include/oidn_webgpu.h) documents every option and lifetime rule.

### Versioning

`oidnw_api_version()` returns `major << 16 | minor << 8 | patch`; `OIDNW_API_VERSION` in the header is 1.0.0 (`0x00010000`). Dynamic loaders should compare the major version before resolving optional functions. Libraries older than this query lack the symbol, so treat a missing symbol as an unknown version.

### Validate a Windows package

[`examples/validate_windows.c`](examples/validate_windows.c) is a standalone MSVC application that links a downloaded package. It generates deterministic HDR color/albedo/normal images, writes RGBA32F output, and prints JSON with the loaded DLL path, adapter, preparation time and individual execution times. Each execution checks finite, nonnegative RGB, preserved alpha and byte-identical repeated output. Execution timing includes uploads and readback; preparation and validation are outside the timed interval.

In an x64 Visual Studio Native Tools PowerShell, run from the repository root. Set `$package` to the extracted download directory; `.native-build/package` below also works after a local package build. The example is included in Windows packages built after it was added; with an older package, compile `native/examples/validate_windows.c` from this repository instead.

```powershell
$package = (Resolve-Path .native-build/package).Path
$work = Join-Path $PWD '.native-build/windows-validation'
New-Item -ItemType Directory -Force $work | Out-Null
cl.exe /nologo /std:c11 /O2 /MD "/I$package/include" "$package/examples/validate_windows.c" "/Fo$work/validate_windows.obj" "/Fe$work/validate_windows.exe" /link "$package/lib/oidn_webgpu.dll.lib"
Copy-Item "$package/lib/oidn_webgpu.dll" $work
& "$work/validate_windows.exe" weights/rt_hdr_calb_cnrm_large.tza 257 259 2 3 0 0 5 21 "$work/output.f32"
```

| Positional argument | Meaning |
| --- | --- |
| Weights, width, height | A three-input HDR TZA model and image dimensions |
| Backend | `2` = Vulkan, `3` = DX12 |
| Kernel | `0` = auto, `1` = implicit GEMM, `2` = direct, `3` = subgroup matrix; subgroup matrix requires Vulkan and cooperative-matrix support |
| Runtime checks | `0` = disabled, `1` = enabled |
| Maximum tile size | `0` = whole image; for example `128` tests overlapping tiles |
| Warm-ups, samples | Untimed executions after the first, then timed executions; the first execution is reported separately |
| Output | RGBA32F file path |
| Input directory (optional) | `color.f32`, `albedo.f32`, `normal.f32` at the requested size; otherwise generated gradients |

Set `OIDNW_EXPECT_ADAPTER` to a name substring (for example `RTX 4060`) to fail if another GPU is selected. The example needs only MSVC, the package and weights; Node, Rust and Zig are not needed.

The downloaded Windows CI package at source `572efc0` was exercised on an RTX 4060 Laptop with NVIDIA 617.42. Vulkan checked/unchecked output matched byte-for-byte for Small/Base/Large at 720p, 1080p, 1440p, partial workgroups and 128-pixel tiles. Real 1024x1024 HDR images passed numerical comparisons with official OIDN 2.5.1 CUDA using the same weights. DX12 implicit GEMM passed with runtime checks disabled. The additional Small DX12 checked test did not finish within the harness's 240-second limit; checked DX12 is not validated by this run.

### Core planner

Hosts with their own non-wgpu GPU executor can link the planner-only static library and replay its programs; see [`include/oidn_webgpu_core.h`](include/oidn_webgpu_core.h) and [`examples/plan.c`](examples/plan.c). Rust/wgpu hosts should use `Denoiser` instead.

On Windows, use MSVC's DLL CRT (`/MD`). The core archive includes the compatibility objects needed by scriptc's Windows runtime; no MinGW installation or additional runtime DLL is required by consumers.

## Precompiled packages

Published native versions are available from [GitHub Releases](https://github.com/pissang/oidn-webgpu/releases) under `native-v<VERSION>`, with raw core archives, a core manifest, native packages, a `.crate` and `SHA256SUMS`. Release assets have no automatic expiry. PR and validation builds remain available as 30-day artifacts from the [Native packages workflow](https://github.com/pissang/oidn-webgpu/actions/workflows/native.yml); artifact downloads require GitHub sign-in. A native package contains a `.tar.gz` and its `.sha256`: extract the artifact, then the tarball.

| Target | Rust triple | Built on |
| --- | --- | --- |
| macOS ARM64 | `aarch64-apple-darwin` | macOS 15 |
| macOS x86_64 | `x86_64-apple-darwin` | macOS 15 |
| Windows x86_64 | `x86_64-pc-windows-msvc` | Windows Server 2022, MSVC |
| Linux x86_64 | `x86_64-unknown-linux-gnu` | Ubuntu 24.04, glibc |

Windows GNU (`x86_64-pc-windows-gnu`), other architectures and Linux musl are not provided, and Windows GNU source builds are not validated. On Windows, build Rust applications for `x86_64-pc-windows-msvc`.

The build host does not guarantee compatibility with older systems; `manifest.json` records the actual OS and, on Linux, the glibc version.

| Package path | Contents |
| --- | --- |
| `include/oidn_webgpu.h` | C filter API |
| `include/oidn_webgpu_core.h` | Core planner API |
| `lib/liboidn_webgpu.dylib`, `.so`, or `oidn_webgpu.dll` + `oidn_webgpu.dll.lib` | C filter shared library |
| `lib/liboidn_webgpu_core.a` or `oidn_webgpu_core.lib` | Core planner static library; also the Rust core override |
| `lib/cmake/oidn_webgpu/` | CMake config, version checks and imported targets |
| `bin/oidn-webgpu-bench` | Network benchmark |
| `examples/denoise.c`, `examples/plan.c`, `examples/host_buffers.rs` | Complete examples |
| `examples/CMakeLists.txt` | C examples using the packaged CMake targets |
| `examples/validate_windows.c` | Windows DLL output and latency check |
| `examples/validate.rs`, `examples/rust-app/Cargo.toml` | Standalone Rust application with a source-pinned dependency |
| `manifest.json`, `SHA256SUMS` | Source commit, target, compiler versions, file hashes |

Release package names contain the crate version and target triple, for example `oidn-webgpu-native-0.1.1-aarch64-apple-darwin.tar.gz`. The full source commit is recorded in `manifest.json` as `source.commit`. CI and local archives also include the full commit in their names. Local archives built from modified sources have a `-dirty` suffix and `source.dirty = true` in the manifest, so their commit alone does not reproduce the contents.

- **C:** use the shared library directly; Node and Rust are not needed.
- **Rust:** pin the Cargo `rev` to `source.commit` from `manifest.json` and set `OIDN_WEBGPU_CORE_ARCHIVE` to the package's core library. Cargo still compiles the Rust layer, and the [wgpu version](#wgpu-version) rule still applies. Precompiled `rlib` files are not provided.

## Build configuration

Environment variables read by `cargo build`:

| Variable | Effect |
| --- | --- |
| `OIDN_WEBGPU_CORE_ARCHIVE` | Absolute path to a prebuilt `liboidn_webgpu_core.a` / `oidn_webgpu_core.lib`. Skips Node and scriptc. Must match the target OS, architecture, ABI and source revision; a registry crate also checks it against the bundled manifest. Useful for source cross compilation. |
| `OIDN_WEBGPU_SCRIPTC` | Path to an existing scriptc compiler; skips installing it. |
| `OIDN_WEBGPU_NODE` | Node executable used for the core build. |

Without `OIDN_WEBGPU_CORE_ARCHIVE`, a registry crate selects and verifies its bundled core using Cargo's target triple, including supported cross-compilation targets. A Git/path source checkout instead builds the core automatically, for native builds only (host = target). Installed build tools stay in Cargo's build directory.

## Performance

The [0.6.0 release notes](https://github.com/pissang/oidn-webgpu/blob/main/CHANGELOG.md#060---2026-10-25) compare native denoising with official OIDN 2.5.1 on an Apple M4 Max (Metal) and an NVIDIA RTX 4060 Laptop (Vulkan). Those timings cover one whole image in GPU buffers, including preprocessing and output conversion, with runtime checks disabled; they exclude uploads and readback.

On the M4 Max, the C example takes about 21 ms per 1024x1024 image with `rt_hdr_alb_nrm.tza`, including three input uploads and output readback. First preparation includes planning and shader compilation; there is no on-disk pipeline cache.

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
