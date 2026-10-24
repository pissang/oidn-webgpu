# oidn-webgpu native

Experimental native GPU denoising through wgpu on Metal, Vulkan or Direct3D 12. Rust hosts can share their GPU device and image buffers; the C API creates a device and accepts CPU images. The API and results may change between versions. GPU denoising has only been verified on macOS with Apple silicon; package checks in CI do not validate GPU execution on Windows or Linux.

## Rust: install

Add a git dependency, or point to a local checkout's `native` directory:

```toml
[dependencies]
oidn-webgpu = { package = "oidn-webgpu-native", git = "https://github.com/pissang/oidn-webgpu.git" }
# Local alternative:
# oidn-webgpu = { package = "oidn-webgpu-native", path = "../oidn-webgpu/native" }
```

Pin a git revision containing this API for reproducible builds. The crate is not published on crates.io.

Source builds require stable Rust, a system C toolchain, Node 22+ and npm. `cargo build` automatically installs the pinned scriptc compiler and builds the core static library; no separate npm command or prebuilt `.native-build` directory is required. The first build needs access to the npm registry. Installed build tools stay in Cargo's build directory. Set `OIDN_WEBGPU_SCRIPTC` to an existing compiler to skip its installation, or `OIDN_WEBGPU_NODE` to select the Node executable.

To use a precompiled core instead, set `OIDN_WEBGPU_CORE_ARCHIVE` to the absolute path of `liboidn_webgpu_core.a` (or `oidn_webgpu_core.lib` on MSVC) before running Cargo. This path requires neither Node nor npm. The archive must match the target OS, architecture, ABI and this source revision. Cross compilation requires a core archive built for the target; automatic core builds support native host/target builds only.

### wgpu compatibility

This crate currently pins `wgpu = "=30.0.1"`. A host sharing `Device`, `Buffer` or `CommandEncoder` must resolve to the same wgpu crate, including its source. An independently loaded C library with CPU images has no such requirement. A precompiled core archive does not remove this Rust type requirement.

You can use the crate's re-export instead of adding a separate wgpu dependency:

```rust
use oidn_webgpu::wgpu;
```

If your host already depends on wgpu, keep its requirements compatible with `=30.0.1`. A different wgpu release or a separate git fork produces incompatible GPU types.

## Rust: share host GPU buffers

Load the model once, create a denoiser on the host's device, and prepare the image size and bindings before the first interactive frame:

```rust
use oidn_webgpu::{DenoiseBuffers, DenoiseParams, Denoiser, DenoiserOptions, Model};

let model = Model::from_tza(weights)?;
let mut denoiser = Denoiser::new(device, adapter, &model, DenoiserOptions::default())?;
let mut prepared = denoiser.prepare([width, height], DenoiseBuffers {
    color: color.into(),
    albedo: Some(albedo.into()),
    normal: Some(normal.into()),
    output: output.into(),
})?;

let mut encoder = device.create_command_encoder(&Default::default());
// Encode the host's render/normalization pass here, ending the pass first.
prepared.encode(&mut encoder, DenoiseParams { input_scale: exposure })?;
// Encode a display or other consumer pass here.
queue.submit([encoder.finish()]);
```

`encode` only appends commands: it does not submit, wait for the GPU or read images back to CPU. Input/output buffers are shared with the host; intermediate GPU processing still uses its own workspace. Submit executions of one prepared instance in encode order on the device's queue. Prepare separate instances for independent workspaces.

`prepare` may block for shader compilation and allocates a workspace for that size. Call it again when dimensions change; the denoiser reuses matching pipelines and immutable GPU weights. Denoisers created from the same `Model` or its clones share compatible weights on the same device; every prepared instance has its own workspace. Loading the TZA again as a separate `Model` creates a separate cache. Active caches also retain packed weight data in CPU memory.

Use `prepared.rebind(buffers)` when only image buffers change; this preserves the pipelines and workspace. Already encoded commands retain their previous bindings. Prepared instances may outlive the denoiser and retain their image buffer handles; do not explicitly destroy buffers while pending GPU work uses them.

Use `prepared.info()` to query the actual activation/weight precision, convolution paths (including fallbacks), tile count, preparation/planning time, per-instance workspace bytes and shared weight bytes. Image buffers and output remain f32. Reported buffer sizes exclude driver overhead and small transient parameter uploads; shared weight bytes must not be added once per instance when estimating total memory.

`Result<T>` uses the typed `Error`. `BufferTooSmall` and `LimitExceeded` include `required` and `available` bytes, and a device-limit error identifies the limit. Invalid per-frame parameters are reported before commands are encoded. Encoding success does not mean GPU completion; execution and device-loss errors follow the host's wgpu handling. The experimental low-level `Network` API uses the separate `PlanResult<T>` with string errors.

The [complete host-buffer example](examples/host_buffers.rs) creates a device, generates noisy color and AOVs on the GPU, denoises in the same command encoder, then exports the final image. From the repository root:

```bash
cargo run --release --manifest-path native/Cargo.toml --example host_buffers -- \
  weights/rt_hdr_alb_nrm_small.tza output.f32 256 192 128
```

The output is tightly packed little-endian RGBA f32. The example reads back only to save the final file; a renderer can consume the output directly on the GPU.

### Image and device requirements

- Every image is tightly packed RGBA f32 with `STORAGE` usage, unmapped and on the denoiser's device. `ImageBuffer { buffer, offset }` binds exactly `width * height * 16` bytes starting at the byte offset; `(&buffer).into()` selects offset zero. The offset must satisfy the device's `min_storage_buffer_offset_alignment` and the range must fit the allocation. A host allocation may be larger than the image's binding limit when the bound image range fits.
- Input ranges may share an allocation if they do not overlap. Output must use a different allocation from all inputs, even for disjoint ranges, because wgpu tracks conflicting storage usages for the whole buffer. Textures, strides and in-place GPU output are not supported.
- Color and albedo are linear RGB. LDR color and albedo use [0, 1]; HDR color is nonnegative. Pass averaged color rather than an unnormalized sample sum. Output keeps the color's alpha; auxiliary alpha is ignored.
- Normals are shading normals in world or view space, with signed components in [-1, 1]. Accumulated normals need not have unit length. Normals encoded in [0, 1] must be remapped before use, consistent with [OIDN's input conventions](https://www.openimagedenoise.org/documentation.html#normals).
- `Model::input_count()` reports 1 (color), 2 (color/albedo) or 3 (color/albedo/normal). Supply exactly those images. Choose weights trained for your HDR/LDR and auxiliary inputs; use `_small` weights for the smaller, faster network. Weights are separate from the crate and C package; see the [repository weights](https://github.com/pissang/oidn-webgpu/tree/main/weights).
- Preparation options default to HDR, PU transfer and whole-image execution. Select `Transfer::Log` for RTLightmap HDR weights. Supply `DenoiseParams` to every `encode`; its default `input_scale` is 1. Scale must be positive and finite, is applied before HDR transfer and undone on output, and is ignored for LDR. You can change it every frame without preparing again. Multiple executions in one command buffer retain their own parameter values. Exposure is not estimated automatically.
- `max_tile_size` limits the output tile edge, rounded down to a multiple of 16. Zero disables tiling; positive values must be in 16..=16384. Tiling reduces intermediate memory but not full-size image buffers. Dimensions must be in 1..=16384 and fit the enabled device buffer limits.
- Before requesting a device, use `Denoiser::required_features()`, `recommended_features(&adapter)` and `required_limits()` to select baseline settings. Higher buffer limits may be needed for your image sizes. Existing devices use only their enabled features and limits; the library cannot enable more afterward.
- Auto uses cooperative-matrix kernels when the host enables `EXPERIMENTAL_COOPERATIVE_MATRIX`, otherwise implicit GEMM. Cooperative matrices also require accepting wgpu's experimental feature contract when requesting the device. Windows/Linux require Vulkan for these kernels; DX12 uses implicit GEMM.

The safe Rust constructor enables shader bounds checks and workgroup initialization. These checks have made convolutions about 4x slower on Metal in existing measurements. `unsafe Denoiser::new_unchecked` disables them and requires the caller to uphold its documented kernel safety contract. Preparation and rebinding return typed errors for invalid images, model failures, insufficient buffer limits and captured GPU validation/allocation failures. GPU execution and device-loss reporting follow the host's wgpu error handling.

## C: build and use

With Node 22+, npm, stable Rust and a C toolchain installed:

```bash
npm run native:build
```

The compiler is installed automatically. The package is written to `.native-build/package`; pass `-- --output DIR` to choose another destination.

| Path | Use |
| --- | --- |
| `include/oidn_webgpu.h` | Standalone CPU-image API |
| `lib/liboidn_webgpu.dylib`, `.so` or `oidn_webgpu.dll` (+ `.dll.lib`) | Standalone denoiser |
| `include/oidn_webgpu_core.h` | Advanced GPU planner API |
| `lib/liboidn_webgpu_core.a` or `oidn_webgpu_core.lib` | Planner, also usable with the Rust archive override |
| `bin/oidn-webgpu-bench` | Network benchmark |
| `examples/denoise.c`, `examples/plan.c` | Complete C examples |
| `examples/host_buffers.rs` | Complete Rust GPU-buffer host example |
| `manifest.json`, `SHA256SUMS` | Source/platform identity and file checksums |

```c
oidnw_device *device;
oidnw_device_create(OIDNW_BACKEND_AUTO, &device, error, sizeof error);
oidnw_filter_desc desc;
oidnw_filter_desc_default(&desc);
oidnw_filter *filter;
oidnw_filter_create(device, tza, tza_size, &desc, &filter, error, sizeof error);
oidnw_filter_prepare(filter, width, height, error, sizeof error);
oidnw_filter_execute(filter, width, height, color, albedo, normal, output, error, sizeof error);
oidnw_filter_destroy(filter);
oidnw_device_destroy(device);
```

Check status codes after each call; [denoise.c](examples/denoise.c) shows error handling. Link with `-loidn_webgpu` and distribute the platform shared library and chosen TZA weights with your application. Images are tightly packed RGBA f32. Execution blocks until CPU output is available, and output may alias an input. The filter caches a workspace for its last image size; changing size creates a new workspace and reuses weights and matching pipelines. Tiling is off by default. `oidnw_filter_prepare` is optional: it allocates and compiles for a size before the first image, without processing an image. Repeating that size reuses resources; `execute` still prepares automatically if needed. Concurrent prepare/execute on one filter returns `OIDNW_ERROR_BUSY`.

`oidnw_api_version()` returns `major << 16 | minor << 8 | patch`; the header's `OIDNW_API_VERSION` is currently 1.0.0 (`0x00010000`). Dynamic loaders can compare the major version for ABI compatibility before resolving optional functions. Libraries predating this query have no version symbol, so feature detection must handle a missing symbol.

The C API defaults to unchecked shaders; set `OIDNW_FILTER_RUNTIME_CHECKS` to enable checks. `quality` controls accumulation precision, while the weights select the model size. See the [header](include/oidn_webgpu.h) for option and lifetime contracts.

Non-wgpu hosts that need their own GPU device may use the [core planner](include/oidn_webgpu_core.h) and implement program execution themselves. Rust/wgpu hosts should use `Denoiser` to share their buffers.

## Download precompiled packages

Download an artifact from a successful [Native packages workflow run](https://github.com/pissang/oidn-webgpu/actions/workflows/native.yml), signing in to GitHub to access the download. Artifacts are retained for 30 days and contain a `.tar.gz` package plus its `.sha256` checksum. Extract the downloaded artifact, then extract the tar archive. Each platform package includes the C shared library, Windows import library where applicable, core static library, headers and examples. Weights are not included. These are Actions artifacts, not GitHub Release assets.

| Platform / architecture | Rust target / ABI | Build environment |
| --- | --- | --- |
| macOS ARM64 (Apple silicon) | `aarch64-apple-darwin` | macOS 15 |
| macOS x86_64 (Intel) | `x86_64-apple-darwin` | macOS 15 |
| Windows x86_64 | `x86_64-pc-windows-msvc` | Windows Server 2022 / MSVC |
| Linux x86_64 | `x86_64-unknown-linux-gnu` | Ubuntu 24.04 / glibc |

Other architectures and Linux musl are not supplied. The build environment is not a guarantee of compatibility with older operating systems; `manifest.json` records the actual OS and, on Linux, glibc version used for each package.

Package names include the native crate version, complete Rust target triple and full source commit SHA. For example: `oidn-webgpu-native-0.1.0-aarch64-apple-darwin-<40-character-commit>.tar.gz`. `manifest.json` records the same identity, compiler versions and file hashes; `SHA256SUMS` also covers the manifest. Local packages made from modified sources carry a `-dirty` suffix and `source.dirty = true`, so their commit alone does not reproduce the contents.

C users can consume the shared library without installing Node or Rust. Rust users should pin the Cargo git dependency's `rev` to `source.commit` from the manifest, then set `OIDN_WEBGPU_CORE_ARCHIVE` to the extracted package's absolute `lib/` archive path. This skips Node/scriptc and still compiles the Rust layer with Cargo. Match the package target to the application's target; the shared-wgpu version requirement still applies. Precompiled Rust `rlib` files are not provided.

## Measured performance

Only an Apple M4 Max has been measured. The following network-only timings exclude oidn-webgpu's input/output processing, which OIDN includes:

| Model | Size | OIDN 2.5.1 Metal | oidn-webgpu native |
| --- | --- | ---: | ---: |
| `rt_hdr_alb_nrm` | 512x512 | 4.64 ms | 4.56 ms |
| `rt_hdr_alb_nrm` | 1920x1080 | 37.1 ms | 33.3 ms |
| `rt_hdr_alb_nrm_small` | 512x512 | 2.53 ms | 2.67 ms |
| `rt_hdr_alb_nrm_small` | 1920x1080 | 18.8 ms | 19.6 ms |

These are existing unchecked network measurements, not end-to-end timings for the new safe Rust API. The C example previously took about 23 ms per 1024x1024 image including three input uploads and output readback. First preparation includes planning and shader compilation; there is no persistent on-disk pipeline cache.
