# oidn-webgpu native

Experimental native GPU denoising through [wgpu](https://wgpu.rs) on Metal, Vulkan or Direct3D 12. It runs the same WGSL kernels as the browser package.

**Status:** the API and results may change between versions. GPU denoising has only been verified on macOS with Apple silicon. CI builds and checks packages for Windows and Linux but does not run them on a GPU.

- [Choose an API](#choose-an-api)
- [Rust API](#rust-api)
- [C API](#c-api)
- [Precompiled packages](#precompiled-packages)
- [Build configuration](#build-configuration)
- [Performance](#performance)

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
oidn-webgpu = { package = "oidn-webgpu-native", git = "https://github.com/pissang/oidn-webgpu.git", rev = "<commit>" }
# Local checkout:
# oidn-webgpu = { package = "oidn-webgpu-native", path = "../oidn-webgpu/native" }
```

- The crate is not on crates.io. Pin `rev` for reproducible builds.
- A source build needs stable Rust, a C toolchain, Node 22+ and npm. `cargo build` installs the pinned scriptc compiler (the first build needs the npm registry) and builds the core static library. To skip Node, use a [precompiled core](#build-configuration).

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

### Device setup

- Before requesting a device, add `Denoiser::required_features()`, `Denoiser::recommended_features(&adapter)` and `Denoiser::required_limits()`. Large images may need higher buffer limits.
- The library uses only the features and limits enabled on an existing device; it cannot enable more later.
- Precision is FP16 when the device enables `SHADER_F16`, otherwise FP32. Image buffers are always f32.
- Kernels: cooperative-matrix kernels when the host enables `EXPERIMENTAL_COOPERATIVE_MATRIX` (which also requires accepting wgpu's experimental feature contract), otherwise implicit GEMM. On Windows and Linux the cooperative-matrix kernels need Vulkan; DX12 uses implicit GEMM.

### Image contract

- **Layout:** tightly packed RGBA f32, `STORAGE` usage, unmapped, on the denoiser's device. Textures, row strides and in-place output are not supported.
- **Range binding:** `ImageBuffer { buffer, offset }` binds exactly `width * height * 16` bytes from `offset`; `(&buffer).into()` uses offset 0. The offset must be a multiple of `min_storage_buffer_offset_alignment` and the range must fit in the buffer. The buffer itself may be larger than the binding limit as long as the bound range fits.
- **Aliasing:** input ranges may share a buffer if they do not overlap. Output must be a different buffer from every input, because wgpu tracks storage usage per buffer.
- **Which images:** supply exactly the images `Model::input_count()` reports: 1 = color, 2 = color + albedo, 3 = color + albedo + normal.
- **Color:** linear RGB. LDR in [0, 1]; HDR nonnegative. Pass the averaged color, not an unnormalized sample sum. Output keeps the color's alpha.
- **Albedo:** linear RGB in [0, 1]. Alpha is ignored.
- **Normal:** world- or view-space shading normals with signed components in [-1, 1]; unit length is not required. Like OIDN, the library clamps them to [-1, 1] and encodes them as `n * 0.5 + 0.5` for the network. Remap normals stored in [0, 1] first ([OIDN conventions](https://www.openimagedenoise.org/documentation.html#normals)); the browser package takes the [0, 1] form instead. Alpha is ignored.

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

Download a [precompiled package](#precompiled-packages), or build one from the repository root (needs Node 22+, npm, stable Rust and a C toolchain):

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

### Core planner

Hosts with their own non-wgpu GPU executor can link the planner-only static library and replay its programs; see [`include/oidn_webgpu_core.h`](include/oidn_webgpu_core.h) and [`examples/plan.c`](examples/plan.c). Rust/wgpu hosts should use `Denoiser` instead.

## Precompiled packages

Download an artifact from a successful [Native packages workflow run](https://github.com/pissang/oidn-webgpu/actions/workflows/native.yml). GitHub sign-in is required, artifacts are kept for 30 days, and they are not GitHub Release assets. Each artifact contains a `.tar.gz` and its `.sha256`: extract the artifact, then the tarball.

| Target | Rust triple | Built on |
| --- | --- | --- |
| macOS ARM64 | `aarch64-apple-darwin` | macOS 15 |
| macOS x86_64 | `x86_64-apple-darwin` | macOS 15 |
| Windows x86_64 | `x86_64-pc-windows-msvc` | Windows Server 2022, MSVC |
| Linux x86_64 | `x86_64-unknown-linux-gnu` | Ubuntu 24.04, glibc |

Other architectures and Linux musl are not provided. The build host does not guarantee compatibility with older systems; `manifest.json` records the actual OS and, on Linux, the glibc version.

| Package path | Contents |
| --- | --- |
| `include/oidn_webgpu.h` | C filter API |
| `include/oidn_webgpu_core.h` | Core planner API |
| `lib/liboidn_webgpu.dylib`, `.so`, or `oidn_webgpu.dll` + `oidn_webgpu.dll.lib` | C filter shared library |
| `lib/liboidn_webgpu_core.a` or `oidn_webgpu_core.lib` | Core planner static library; also the Rust core override |
| `bin/oidn-webgpu-bench` | Network benchmark |
| `examples/denoise.c`, `examples/plan.c`, `examples/host_buffers.rs` | Complete examples |
| `manifest.json`, `SHA256SUMS` | Source commit, target, compiler versions, file hashes |

Package names contain the crate version, target triple and full source commit, for example `oidn-webgpu-native-0.1.0-aarch64-apple-darwin-<commit>.tar.gz`. Packages built from modified sources have a `-dirty` suffix and `source.dirty = true` in the manifest, so their commit alone does not reproduce the contents.

- **C:** use the shared library directly; Node and Rust are not needed.
- **Rust:** pin the Cargo `rev` to `source.commit` from `manifest.json` and set `OIDN_WEBGPU_CORE_ARCHIVE` to the package's core library. Cargo still compiles the Rust layer, and the [wgpu version](#wgpu-version) rule still applies. Precompiled `rlib` files are not provided.

## Build configuration

Environment variables read by `cargo build`:

| Variable | Effect |
| --- | --- |
| `OIDN_WEBGPU_CORE_ARCHIVE` | Absolute path to a prebuilt `liboidn_webgpu_core.a` / `oidn_webgpu_core.lib`. Skips Node and scriptc. Must match the target OS, architecture, ABI and source revision. Required for cross compilation. |
| `OIDN_WEBGPU_SCRIPTC` | Path to an existing scriptc compiler; skips installing it. |
| `OIDN_WEBGPU_NODE` | Node executable used for the core build. |

Without `OIDN_WEBGPU_CORE_ARCHIVE`, the core is built automatically, for native builds only (host = target). Installed build tools stay in Cargo's build directory.

## Performance

Network-only GPU time on an Apple M4 Max, the only measured device. oidn-webgpu numbers exclude input/output processing, which the OIDN numbers include.

| Model | Size | OIDN 2.5.1 Metal | oidn-webgpu native |
| --- | --- | ---: | ---: |
| `rt_hdr_alb_nrm` | 512x512 | 4.64 ms | 4.56 ms |
| `rt_hdr_alb_nrm` | 1920x1080 | 37.1 ms | 33.3 ms |
| `rt_hdr_alb_nrm_small` | 512x512 | 2.53 ms | 2.67 ms |
| `rt_hdr_alb_nrm_small` | 1920x1080 | 18.8 ms | 19.6 ms |

These are unchecked network measurements, not end-to-end timings of the safe Rust API. The C example took about 23 ms per 1024x1024 image, including three input uploads and output readback. First preparation includes planning and shader compilation; there is no on-disk pipeline cache.
