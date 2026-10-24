# oidn-web native

An experimental native build of the oidn-web denoiser with a C API. It runs the same model-driven WGSL kernels as the browser runtime on Metal, Vulkan or Direct3D 12 through [wgpu](https://wgpu.rs), with no JavaScript engine at run time. The planner, the kernel generator and the tiling are oidn-web's TypeScript compiled to native code by [scriptc](https://scriptc.dev). The API and the results may change between versions.

## Build

Requirements: Node 22+, stable Rust, and scriptc 0.3.2:

```bash
npm ci --prefix native/toolchain --include=optional
npm run native:build
```

`OIDN_WEB_SCRIPTC=/path/to/scriptc` replaces the toolchain package. The package is written to `.native-build/package` (`--output DIR` changes it):

| Path | Contents |
| --- | --- |
| `include/oidn_web.h` | Standalone denoiser C API |
| `include/oidn_web_core.h` | Planner C API (no GPU dependency) |
| `lib/liboidn_web.dylib`, `.so` or `oidn_web.dll` (+ `.dll.lib`) | Standalone denoiser (wgpu linked in) |
| `lib/liboidn_web_core.a` or `oidn_web_core.lib` | Planner only: scriptc-compiled core, runtime symbols localized |
| `bin/oidn-web-bench` | Network benchmark and checks |
| `examples/denoise.c` | Denoises raw RGBA f32 images |
| `examples/plan.c` | Plans a denoising with the planner library only |

Weights are not included: pass the OIDN TZA files from `weights/`.

## C APIs

The two libraries serve two kinds of hosts.

**Standalone denoiser (`oidn_web.h`).** Open Image Denoise-style, with images in CPU memory:

```c
oidnw_device *device;
oidnw_device_create(OIDNW_BACKEND_AUTO, &device, error, sizeof error);
oidnw_filter_desc desc;
oidnw_filter_desc_default(&desc);            /* HDR, PU transfer, high quality */
oidnw_filter *filter;
oidnw_filter_create(device, tza, tza_size, &desc, &filter, error, sizeof error);
oidnw_filter_execute(filter, width, height, color, albedo, normal, output, error, sizeof error);
oidnw_filter_destroy(filter);
oidnw_device_destroy(device);
```

All images are tightly packed RGBA f32. The output keeps the color's alpha, and only the RGB of albedo and normal is read. The filter caches the plan and the GPU resources of the last image size. Tiling is off by default; `max_tile_size` turns it on. `OIDNW_KERNEL_AUTO` uses the subgroup-matrix (cooperative-matrix) kernels when the device supports them, and implicit GEMM otherwise.

**Planner (`oidn_web_core.h`).** For hosts that render with their own GPU device. `oidn_core_record_denoise` plans one image size for the host's device capabilities. It returns a program: buffers with initial data, WGSL compute shaders, bind groups and the ordered dispatches of one denoising. The host creates the resources once, binding its own color, albedo, normal and output buffers, and replays the operations for each frame. This shares the host's device and buffers without any copy or extra GPU API instance. `native/src/lib.rs` (`Network`) is a ~250-line reference replayer for wgpu. The header documents the config and program formats.

Shaders are compiled without runtime bounds checks unless `OIDNW_FILTER_RUNTIME_CHECKS` is set. Naga's checks make the convolutions about 4x slower on Metal. The kernels are written to stay in bounds and are validated in browsers, but browsers clamp out-of-bounds accesses and would hide a bug that unchecked native execution would not.

## Platforms and kernels

| Platform | Backend | Matrix kernels |
| --- | --- | --- |
| macOS (Apple silicon) | Metal | 8x8 simdgroup matrices, f16 inputs with f32 accumulation |
| Windows, Linux | Vulkan (`OIDNW_BACKEND_VULKAN`) | `VK_KHR_cooperative_matrix` through wgpu, configurations as reported by the driver (unverified) |
| Windows | DX12 | None (implicit GEMM); wgpu has no DX12 cooperative matrices |

Only macOS on an Apple M4 Max has been measured.

Network GPU time per image (`oidn-web-bench --unchecked`) against Open Image Denoise 2.5.1's Metal device (`benchmarks/oidn-sdk/oidn_bench.c`), both with FP16 storage:

| Model | Size | OIDN 2.5.1 Metal | oidn-web native |
| --- | --- | ---: | ---: |
| `rt_hdr_alb_nrm` | 512x512 | 4.64 ms | 4.56 ms |
| `rt_hdr_alb_nrm` | 1920x1080 | 37.1 ms | 33.3 ms |
| `rt_hdr_alb_nrm_small` | 512x512 | 2.53 ms | 2.67 ms |
| `rt_hdr_alb_nrm_small` | 1920x1080 | 18.8 ms | 19.6 ms |

The oidn-web times exclude the input and output processing, which OIDN includes. `examples/denoise.c` on a 1024x1024 image takes about 23 ms per call, including uploading three inputs and reading back the output.

Correctness (`benchmarks/native-denoise-compare.mjs`, `benchmarks/native-compare.mjs`):

- With implicit GEMM, the whole C API pipeline is bit-identical to the browser runtime's `tileExecute` with GPU buffers, both whole-image and tiled.
- The default subgroup-matrix kernels have an average relative error of 3.3e-4 against browser FP32 on the 1024x1024 test image.

## Integrating with MuTracer native

MuTracer's native host currently reads the AOVs back to the CPU and runs OIDN 2.5.1 through `libloading` (`native/host/src/denoise.rs`). It then uploads the result again. Two integrations are possible:

1. **Same device, no copies (recommended).** Link `liboidn_web_core.a` beside MuTracer's own scriptc archive; both localize their runtimes, and only `oidn_core_*` symbols are exported. Add a built-in denoiser to `native/host` that:
   - passes the render device's capabilities (`capability_words` in `native/src/lib.rs`);
   - records the program for the render size;
   - replays it on the render's wgpu device, binding the accumulated color, albedo and normal buffers and a denoised output buffer. Each must be RGBA f32 per pixel with `STORAGE` usage; MuTracer's raw per-sample color needs its normalization pass first.

   Replanning is needed only when the size changes; it takes about 6 ms plus pipeline creation. This removes the OIDN runtime files, the CPU round trip and the `CustomDenoiser` readback. It works with Progressive and EverySample denoising on the GPU timeline.
2. **Drop-in shared library.** Load `liboidn_web` in place of OIDN behind `mt_denoiser_*`. The data flow (CPU arrays) and the API shape stay the same. MuTracer's Rust host should not link the Rust `staticlib`, because two Rust standard libraries in one binary conflict; use the shared library.

On Windows, matrix kernels require MuTracer's device to use Vulkan. With DX12, the planner falls back to implicit GEMM, about 1.4x slower on the M4 Max.
