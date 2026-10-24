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
use oidn_webgpu::{DenoiseBuffers, Denoiser, DenoiserOptions, Model};

let model = Model::from_tza(weights)?;
let mut denoiser = Denoiser::new(device, adapter, &model, DenoiserOptions::default())?;
let mut prepared = denoiser.prepare(device, [width, height], DenoiseBuffers {
    color,
    albedo: Some(albedo),
    normal: Some(normal),
    output,
})?;

let mut encoder = device.create_command_encoder(&Default::default());
// Encode the host's render/normalization pass here, ending the pass first.
prepared.encode(&mut encoder)?;
// Encode a display or other consumer pass here.
queue.submit([encoder.finish()]);
```

`encode` only appends commands: it does not submit, wait for the GPU or read images back to CPU. Input/output buffers are shared with the host; intermediate GPU processing still uses its own workspace. Submit executions of one prepared instance in encode order on the device's queue. Prepare separate instances for independent workspaces.

`prepare` may block for shader compilation and allocates a workspace for that size. Call it again when dimensions change; the denoiser reuses matching pipelines. Use `prepared.rebind(buffers)` when only image buffers change; this preserves the pipelines and workspace. Already encoded commands retain their previous bindings. Prepared instances may outlive the denoiser and retain their image buffer handles; do not explicitly destroy buffers while pending GPU work uses them.

The [complete host-buffer example](examples/host_buffers.rs) creates a device, generates noisy color and AOVs on the GPU, denoises in the same command encoder, then exports the final image. From the repository root:

```bash
cargo run --release --manifest-path native/Cargo.toml --example host_buffers -- \
  weights/rt_hdr_alb_nrm_small.tza output.f32 256 192 128
```

The output is tightly packed little-endian RGBA f32. The example reads back only to save the final file; a renderer can consume the output directly on the GPU.

### Image and device requirements

- Every image is a tightly packed RGBA f32 buffer starting at offset zero, with at least `width * height * 16` bytes and `STORAGE` usage. All supplied buffers must be distinct, unmapped and belong to the denoiser's device. Textures, strides, buffer offsets and in-place output are not supported in this API.
- Color and albedo are linear RGB. LDR color and albedo use [0, 1]; HDR color is nonnegative. Pass averaged color rather than an unnormalized sample sum. Output keeps the color's alpha; auxiliary alpha is ignored.
- Normals are shading normals in world or view space, with signed components in [-1, 1]. Accumulated normals need not have unit length. Normals encoded in [0, 1] must be remapped before use, consistent with [OIDN's input conventions](https://www.openimagedenoise.org/documentation.html#normals).
- `Model::input_count()` reports 1 (color), 2 (color/albedo) or 3 (color/albedo/normal). Supply exactly those images. Choose weights trained for your HDR/LDR and auxiliary inputs; use `_small` weights for the smaller, faster network. Weights are separate from the crate and C package; see the [repository weights](https://github.com/pissang/oidn-webgpu/tree/main/weights).
- Options default to HDR, PU transfer, `input_scale = 1` and whole-image execution. Select `Transfer::Log` for RTLightmap HDR weights. Input scale must be positive and finite; it is applied before the HDR transfer and undone on output. Exposure is not estimated automatically.
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
oidnw_filter_execute(filter, width, height, color, albedo, normal, output, error, sizeof error);
oidnw_filter_destroy(filter);
oidnw_device_destroy(device);
```

Check status codes after each call; [denoise.c](examples/denoise.c) shows error handling. Link with `-loidn_webgpu` and distribute the platform shared library and chosen TZA weights with your application. Images are tightly packed RGBA f32. Execution blocks until CPU output is available, and output may alias an input. The filter caches resources for its last image size; changing size prepares new resources. Tiling is off by default.

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
