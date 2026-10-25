# Development guide

How the code is organized, how to build and test it, and how to change it safely. For working rules aimed at coding agents, see [AGENTS.md](../AGENTS.md).

- [Architecture](#architecture)
- [Commands](#commands)
- [Updating to a new OIDN model](#updating-to-a-new-oidn-model)
- [Benchmarks and GPU checks](#benchmarks-and-gpu-checks)
- [CI and native packages](#ci-and-native-packages)
- [Documentation map](#documentation-map)

## Architecture

```
TZA file
 └─ parseTZA                     src/tza.ts
 └─ detect/validateUNetModel     src/modelSpec.ts       graph descriptor + tensor validation
 └─ optimizeModelGraph           src/graphOptimizer.ts  topology-only fusions, execution plan
 └─ NativeUNetCore               src/nativeUNetCore.ts  WGSL generation, buffers, dispatches
      │                                                 (against OIDNGPUDevice, src/gpuTypes.ts)
      ├─ Browser ─ NativeUNetExecutor  src/nativeUNet.ts   async pipelines on a GPUDevice
      │            UNet                src/UNet.ts         tiling, scheduling, callbacks
      │            GPUDataProcess      src/process.ts      input/output conversion passes
      │
      └─ Native ── program.ts          src/native/         records a run on RecordingDevice
                   │  scriptc (src/native/core-profile.json)
                   ▼
                   liboidn_webgpu_core.a   C ABI oidnw_core_*   native/include/oidn_webgpu_core.h
                   ▼
                   native/src/lib.rs       replays the recorded program on wgpu (Network)
                   native/src/denoiser.rs  Rust Denoiser API
                   native/src/capi.rs      C filter API          native/include/oidn_webgpu.h
                   native/src/gpu_capi/    GPU C API, submission ledger, Metal imports
```

The browser and native runtimes share one TypeScript core. The native build compiles it to a static library with scriptc (version pinned in `native/toolchain/package.json`); `RecordingDevice` captures buffers, shaders and dispatches as a program, and Rust replays that program on a wgpu device.

### Shared core vs browser-only code

| Compiled into the native core (shared) | Browser-only |
| --- | --- |
| `src/tza.ts`, `src/modelSpec.ts`, `src/graphOptimizer.ts`, `src/nativeUNetCore.ts`, `src/finalRgbShader.ts`, `src/gpuTypes.ts`, `src/hdrTransfer.ts`, `src/sanitize.ts`, `src/tileScheduler.ts`, `src/resourceTracker.ts`, `src/native/*` | `src/main.ts`, `src/UNet.ts`, `src/nativeUNet.ts`, `src/process.ts`, `src/backend.ts`, `src/webnnUNet.ts`, `src/WGPUComputePass.ts`, `src/WGPUFullQuadPass.ts`, `src/hdr.ts` |

Consequences:

- A change to a shared file changes native output too. Run the native checks below.
- Shared code must compile with scriptc: no DOM or browser globals, and GPU access only through the `OIDNGPU*` types in `src/gpuTypes.ts`. Async GPU work belongs in browser-only code.
- `native/build.rs` watches `src/`, so `cargo build` rebuilds the core automatically after a change.
- The program format between `src/native/program.ts` and `native/src/lib.rs` is versioned (documented at the top of `program.ts`). Change both sides together.

### Key files

| Path | Responsibility |
| --- | --- |
| `src/main.ts` | Public npm exports, `UNetOptions`, `initUNetFromURL` / `initUNetFromBuffer` |
| `src/UNet.ts` | `tileExecute`, `prepareForImage`, runtime info, profiling, device-loss handling |
| `src/tileScheduler.ts` | Tile planning (`planTileGrid`) and adaptive tile size |
| `src/nativeUNetCore.ts` | Kernel selection, WGSL generation, precision; experimental `gemm` / `subgroupMatrix` options |
| `src/modelSpec.ts` | `OIDN_UNET_SMALL_SPEC`, `OIDN_UNET_LARGE_SPEC`, detection and validation |
| `src/native/program.ts` | Native core entry; config and program wire formats |
| `native/src/denoiser.rs` | `Model`, `Denoiser`, `PreparedDenoise`, `Error` |
| `native/src/capi.rs` | `oidnw_*` C functions; must match `native/include/oidn_webgpu.h` |
| `native/src/gpu_capi/` | GPU C handles, bounded completion ledger, binding cache and platform adapters; matches `oidn_webgpu_gpu.h` and `oidn_webgpu_metal.h` |
| `native/src/bin/oidn-webgpu-bench.rs` | Native network benchmark |
| `scripts/native-*.mjs` | Core build, C package build, package identity, package checks |

## Commands

Run from the repository root. Requirements: Node 22+; for native work also stable Rust and a C toolchain (on Windows also Zig 0.16.0 on `PATH`). Building or checking C packages also needs CMake 3.20+.

| Task | Command | Needs a GPU |
| --- | --- | --- |
| Install | `npm ci` | no |
| Build library (`lib/`, type declarations) | `npm run build:lib` | no |
| Build bundle (`dist/`) | `npm run build` | no |
| Unit tests (builds both first) | `npm test` | no |
| Examples dev server (HTTPS) | `npm run dev`, then open `/examples/basic.html` | browser |
| Inspect a TZA file | `npm run model:inspect -- weights/rt_hdr_alb_nrm.tza` | no |
| Resource leak check | `npm run test:leak -- [--cycles 6] [--engine wgsl\|webnn]` | Chrome |
| Native crate build | `cargo build --release --manifest-path native/Cargo.toml` | no |
| Native unit tests | `cargo test --manifest-path native/Cargo.toml` | no |
| Native GPU tests | `cargo test --manifest-path native/Cargo.toml -- --ignored` | yes |
| C package | `npm run native:build` (output in `.native-build/package`) | no |

Tests import compiled output from `lib/` and `dist/`, so always run them through `npm test`. Generated directories (`lib/`, `dist/`, `.native-build/`, `.zig-cache/`, `native/target/`, `benchmarks/results/`) are git-ignored. The Windows CI Zig setup action puts its compiler cache in `.zig-cache/`; it does not affect the package's clean-source check.

## Updating to a new OIDN model

1. Inspect the new weights. The command prints a SHA-256, the full tensor signature and descriptor compatibility:

   ```shell
   npm run model:inspect -- path/to/new.tza
   ```

2. If a built-in descriptor matches (same topology and tensor names, possibly different channel widths), no code change is needed.
3. Otherwise add a descriptor in `src/modelSpec.ts` and a validation fixture in `tests/modelSpec.test.mjs`. Existing graph fusions apply without WGSL changes.
4. Run `npm test`, then the native checks, since `modelSpec.ts` is shared.

## Benchmarks and GPU checks

Browser scripts drive headless Chrome through `playwright-core`. All of them honor `CHROME_PATH`; most also accept `--chrome <path>` and search common install locations. Run `npm run build` first.

| Script | Purpose |
| --- | --- |
| `npm run benchmark -- --width 512 --height 512 --runs 5` | Current WGSL runtime (FP32/FP16) vs. the nearest ancestor commit that still used TensorFlow.js, built in a temporary worktree. Writes `benchmarks/results/latest.{json,md}`. `--help` lists options (`--baseline`, `--kernels`, `--weights`, ...). |
| `node benchmarks/native-compare.mjs` | Native network vs. browser on the same input; expects bit-identical output for the same kernel and precision. Needs the native crate built. |
| `node benchmarks/native-denoise-compare.mjs --denoise <denoise binary>` | Native C API end to end vs. browser on `examples/test`. `--denoise` is the compiled `native/examples/denoise.c`; needs `npm run build:lib` and `npm run native:build`. |
| `node benchmarks/subgroup-matrix.mjs` | Probes `chromium-experimental-subgroup-matrix` configurations and peak throughput. |
| `benchmarks/oidn-sdk/oidn_bench.c` | Official OIDN GPU timing for comparison; usage in the file header. |
| `oidn-webgpu-bench --weights W --width N --height N` | Native network timing and output dumps (`cargo run --release --manifest-path native/Cargo.toml --bin oidn-webgpu-bench -- ...`). |
| `node benchmarks/mac-performance.mjs --oidn-library /path/to/libOpenImageDenoise.dylib` | Pinned 0.5.0/current Chrome, Chrome subgroup matrix, `oidn-webgpu (native wgpu)` and official OIDN Metal; Small/Base/Large at 720p, 1080p and 2560x1440. Builds both revisions and native benchmark automatically. |
| `cargo run --release --manifest-path native/Cargo.toml --bin oidn-webgpu-denoise-bench -- --weights W --width N --height N` | Native full-image GPU-buffer latency, including preprocessing and output conversion. Runtime checks off by default; `--runtime-checks` enables them. `--output PATH` saves packed RGBA32F output after timing. |
| `node scripts/*GpuCheck.mjs` | Correctness checks in Chrome: kernels against CPU references, and input/output sanitizing (`sanitizeGpuCheck.mjs`). |

Each script's header comment lists its options.

The Mac suite requires Node 22+, Chrome, Rust, a C compiler and an OIDN SDK with Metal. `CHROME_PATH` and `OIDN_LIBRARY` may supply the executable/library paths. Run `node benchmarks/mac-performance.mjs --help` for model, size, repetition and revision overrides. It builds detached temporary worktrees, so uncommitted runtime edits are excluded; temporary worktrees are removed after the run. GPU cases run sequentially.

Default inputs are `720p` (1280x720), `1080p` (1920x1080) and `1440p` (2560x1440). Use `--sizes 720p` to measure only 720p, or `--sizes 512` for the 512x512 regression case.

The suite writes raw samples, model and binary hashes, source revisions, selected kernels, resource sizes and a Markdown report to `benchmarks/results/mac-performance.{json,md}`. Its metric is synchronized latency for one GPU-resident RGBA32F HDR image with albedo and normal. It includes preprocessing, network execution and output conversion, and excludes uploads, readback, device creation and preparation. Default protocol: 3 warmups, 7 measured images, whole-image processing, `inputScale=1`. Native runtime checks are disabled to match the C API default. Official OIDN loads the same TZA weights; Small uses fast quality, Base uses high, and Large uses high with clean auxiliary images. This is a timing comparison, not an image-quality comparison.

Repeat the changelog's Apple M4 Max comparison with OIDN 2.5.1 and:

```bash
OIDN_LIBRARY=/absolute/path/to/libOpenImageDenoise.dylib node benchmarks/mac-performance.mjs \
  --current-ref fd0d816 --warmup 5 --runs 21
```

When one case is slower in every column, OIDN Metal included, remeasure only that case, for example `--models large --sizes 720p`.

### Native Metal interoperability prototype

On macOS, `native/examples/metal_interop.rs` validates importing a host Metal device and private image buffers through the Rust and GPU C APIs. Native Metal kernels produce inputs and consume denoised outputs on separate queues; image pixels stay on the GPU. The CPU inspects only per-pixel validation flags after execution.

The test rotates three image sets through one prepared workspace, with 256-byte offsets and three exposure scales. It compares every output pixel byte-for-byte with serialized executions of the existing Rust API. The producer remains gated until three executions have been submitted (or all executions, for shorter runs); denoising must not signal completion before that gate opens. The example is a correctness test, not a latency benchmark.

```bash
MTL_DEBUG_LAYER=1 cargo run --release --locked --manifest-path native/Cargo.toml \
  --example metal_interop -- --frames 1000
MTL_DEBUG_LAYER=1 native/target/release/examples/metal_interop --frames 1000 --shared-queue
MTL_DEBUG_LAYER=1 native/target/release/examples/metal_interop --frames 1000 --size 31 --tile 16 --fp32
MTL_DEBUG_LAYER=1 native/target/release/examples/metal_interop --frames 1000 --c-api --shared-queue
```

| Option | Type | Default | Meaning |
| --- | --- | --- | --- |
| `--frames` | integer | `1000` | 1..=100000 executions; validation flags must fit in 256 MiB |
| `--size` | integer | `16` | Square image edge, 1..=1024 pixels |
| `--weights` | path | Small HDR color/albedo/normal model | TZA model with three inputs |
| `--tile` | integer | `0` | Maximum output tile edge; follows `DenoiserOptions` |
| `--shared-queue` | flag | off | Share the native producer queue; CPU queue access remains serialized by the single submitting thread |
| `--fp32` | flag | off | Disable FP16 on the imported device |
| `--c-api` | flag | off | Exercise exported GPU C functions against the serialized Rust oracle, with 8 pending submissions, 3 cached binding sets, queue-lock callbacks and in-flight C handle release |
| `--staged-waits` | flag | off | Diagnostic mode using HAL's best-effort staged waits instead of strict gating; not an integration contract |
| `--trace` | flag | off | Print CPU submission timestamps to stderr |

The queues use a command-buffer capacity of 4096, matching wgpu-hal 30.0.1. Metal's default queue capacity can block submission when gated tiled executions accumulate. A host supplying a shared queue must account for this capacity and serialize its own submissions with library queue access and destruction. Strict event synchronization is enabled before any library command encoding. It currently uses HAL's relay-event mechanism; do not mix `CommandEncoder::as_hal_mut` with wgpu command encoding, which wgpu 30 rejects.

The C API implementation has one shared submission ledger and a `Backend`/`Timeline` boundary for platform imports and external completion. Binding snapshots share a workspace and are capped per prepared instance; public buffer tokens invalidate cached entries without changing already recorded commands. GPU errors with uncertain submission quarantine the context rather than freeing resources whose use is unknown. Explicit quiescent teardown requires host proof that those resources are no longer in use; a library error alone is not that proof.

For a controlled encoding failure after acceptance, the runtime submits no denoising and clears a private 4-byte marker. Metal HAL's empty-submit path is not relay-gated, so a truly empty batch could publish a signal before its external wait; the marker forces a normal gated command buffer without accessing host images. The ignored GPU test below checks admission limits, timeout, concurrent workspace submissions, balanced shared-queue callbacks, failure signaling, in-flight release, quarantine and quiescent teardown without inducing a driver fault:

```bash
MTL_DEBUG_LAYER=1 cargo test --locked --manifest-path native/Cargo.toml \
  metal_backpressure_and_failed_batch_preserve_external_completion -- --ignored --nocapture
```

The standalone [Objective-C consumer](../native/examples/metal_gpu.m) uses the shipped headers and shared library; its build and usage are in the [GPU C API guide](../native/README.md#gpu-c-api-on-metal). CI compiles it on macOS through the packaged CMake examples but does not run GPU work.

### Native Windows interoperability prototypes

`native/examples/vulkan_interop.rs` and `native/examples/dx12_interop.rs` exercise native GPU production, the existing Rust `PreparedDenoise`, and native GPU consumption. Image allocations are never CPU mapped, uploaded or downloaded. A GPU consumer compares every output channel's bits with three serial references from the same device, model, kernel, precision and runtime-check setting. The CPU reads integer validation flags after execution. These correctness tests include artificial delays and provide no performance measurements.

Both examples reuse one prepared workspace, rotate three image sets, bind at byte offset 256 and vary input patterns and exposure on each reuse of a slot. Producer, denoiser and consumer use GPU dependencies, including a consumer-completion dependency before a slot can be reused. The last producer stays gated while the application drops the prepared workspace and imported wgpu handles. Native command and descriptor objects are retired by completion values, with a maximum of 64 pending pairs. At that limit the test waits for the oldest consumer, rather than waiting for GPU idle or waiting after every frame. Allocator samples are included where the backend exposes them.

| Route | Device creation and ownership | Current scope |
| --- | --- | --- |
| Vulkan A: host device import | Host creates `VkInstance`, `VkDevice`, three queues in one graphics/compute family, and device-local buffers. HAL uses the host loader table. Non-owning instance/device callbacks and `from_raw_externally_owned` retain host owners. | Executed on Windows. A dedicated denoise queue is the default; `--shared-queue` shares the producer queue. A single host mutex covers device import, queue submissions, polling, cleanup and final wgpu destruction. This is an example integration, not a public queue-lock callback API. |
| Vulkan B: external memory | A separate library device would import Win32 memory/semaphore handles. | Only physical-device capability queries are implemented. The RTX 4060 reports `OPAQUE_WIN32` buffer and timeline semaphore import/export support. Cross-device binding, memory-type compatibility, dedicated allocations and `VK_QUEUE_FAMILY_EXTERNAL` release/acquire execution remain unverified. |
| DX12: external memory | Host and library create distinct devices on the same adapter LUID using Agility independent devices. wgpu normally creates the library device. Host exports shared UAV-capable committed buffers and shared fences; the library uses `OpenSharedHandle` and HAL buffer wrapping. | Image roles stay fixed. Native DIRECT command lists use buffer promotion from COMMON and decay to COMMON at `ExecuteCommandLists` completion, plus UAV barriers. Fences order accesses across devices; fences do not establish resource states. Exporter-owned NT handles close immediately after opening; opened resources/fences retain COM references. |

Vulkan A is appropriate for hosts that can satisfy HAL's device-creation contract. Before `vkCreateDevice`, the example queries `required_device_extensions` and `physical_device_features`, snapshots the complete feature chain, and checks the original host `VkDeviceCreateInfo` before import. This includes HAL's implicit robustness, timeline, shader-int8 and other requirements, not just denoise shader requirements. The prototype accepts HAL's canonical individual feature structs; aggregate Vulkan 1.1/1.2/1.3 feature structs are rejected until normalization is implemented. `--no-recommended` disables optional wgpu acceleration features; it does not waive HAL requirements. No experimental matrix features are requested. A host whose existing device does not satisfy that contract needs another route. Route B would avoid that enabled-feature constraint but adds allocation compatibility, external ownership transfers and handle management. It is not offered as a working fallback on the evidence of capability queries alone.

Vulkan has two additional constraints with the pinned wgpu 30.0.1:

- HAL chains submissions with binary relay semaphores. Unresolved future timeline signals in that dependency chain can trigger `VUID-vkQueueSubmit-pWaitSemaphores-03238`, observed in shared-queue testing. The delayed-producer test therefore submits one gated execution, checks that it cannot complete, and resolves its CPU gates before chaining further HAL submissions. Normal frames use GPU producer/consumer dependencies. Deep chains of unresolved future CPU signals are not validated as an integration contract.
- Naga emits workgroup array layout decorations that trigger `VUID-StandaloneSpirv-None-10684` ([upstream issue](https://github.com/gfx-rs/wgpu/issues/7696)). HAL's own messenger suppresses that VUID. This example preserves the full diagnostic and counts the observed workgroup/ArrayStride case separately in `known_naga_workgroup_layout_diagnostics`; other validation errors fail the test. A run with a nonzero count is not clean shader validation.

Run from an x64 Visual Studio developer PowerShell with Rust's `x86_64-pc-windows-msvc` toolchain, Node 22+ and Zig 0.16.0 on `PATH`. A source-matched `OIDN_WEBGPU_CORE_ARCHIVE` can replace the core build. Install the Khronos validation layer and point `VK_LAYER_PATH` at its manifest directory if it is not registered globally:

```powershell
$env:VK_LAYER_PATH = 'C:/VulkanSDK/<version>/Bin'
pwsh -File scripts/native-interop-check.ps1 -Backend vulkan
pwsh -File scripts/native-interop-check.ps1 -Backend vulkan -Frames 12 -GpuValidation
```

DX12 validation requires Windows Graphics Tools, DXC on the DLL search path for FP16, and a local [Agility SDK](https://devblogs.microsoft.com/directx/directx12agility/) with matching x64 `D3D12Core.dll` and `d3d12SDKLayers.dll`. The x64 VS developer environment used locally supplies DXC from the Windows SDK. The prototypes use the independent-device API without changing wgpu or its version. Set an absolute SDK binary directory ending in a slash:

```powershell
$env:WGPU_DX12_AGILITY_SDK_PATH = 'C:/SDKs/Agility/build/native/bin/x64/'
$env:WGPU_DX12_AGILITY_SDK_VERSION = '619'
$env:WGPU_DX12_AGILITY_SDK_REQUIRE = '1'
pwsh -File scripts/native-interop-check.ps1 -Backend dx12 -Unchecked
pwsh -File scripts/native-interop-check.ps1 -Backend dx12 -Frames 12 -Unchecked -GpuValidation
```

The runner builds the explicit MSVC target and runs Small/Base at 16×16 and Large at 31×31 with tile edge 16, each in FP16 and FP32. Vulkan Base uses the shared queue; other cases use a dedicated denoise queue. `-NoDebug` is an explicit DX12 diagnostic mode if the debug layer cannot load; its reports contain `debug_layer=false` and `validation_errors=null`. It does not establish debug-layer validation. `-Unchecked` disables shader bounds checks and workgroup initialization for both the reference and asynchronous execution; the debug layer is a separate option. Checked DX12 Small did not finish a 1000-frame diagnostic run within 300 seconds on the local machine; that run has no correctness conclusion.

Measured correctness on 2026-10-25: Windows 11 build 26200, RTX 4060 Laptop, NVIDIA 617.42, wgpu 30.0.1, Khronos layer API 1.4.366 and Agility SDK 1.619.6. Each row below covers the runner's six model/precision cases. The Intel Iris Xe in the same machine was not exercised.

| Backend | Frames per case | Shader runtime checks | Native validation | Result |
| --- | ---: | --- | --- | --- |
| Vulkan host import | 1000 | enabled | Khronos layer | Byte equal; 8 or 10 known Naga layout diagnostics per case; zero other validation errors, including teardown |
| Vulkan host import | 12 | enabled | GPU-assisted and synchronization validation | Byte equal; same Naga limitation; zero other validation errors |
| DX12 shared memory | 1000 | disabled | D3D12 debug layer on both devices | Byte equal; zero validation errors |
| DX12 shared memory | 12 | disabled | Debug layer and GPU validation on both devices | Byte equal; zero validation errors |

All runs checked delayed production, delayed consumption, per-slot content changes, in-flight wrapper release, rejected parameters/imports and an empty failure batch retaining its wait/signal. Pending native command pairs never exceeded 64 and drained to zero. Allocator samples stayed within the prepared resources plus the bounded staging pool over the 1000-frame runs; this does not establish an unlimited-duration or device-loss reclamation bound. Vulkan also passed a 12-frame FP32 run with recommended wgpu features omitted. The Metal prototype was already present in this checkout; it was not rerun on Windows.

| Example option | Type | Default | Meaning |
| --- | --- | --- | --- |
| `--frames` | integer | `1000` | 1..=100000 executions, with at most 256 MiB of validation flags |
| `--size` | integer | `16` | Square image edge, 1..=1024 |
| `--tile` | integer | `0` | Output tile edge; 0 or 16..=16384 |
| `--weights` | path | Small three-input HDR model | Model must read color/albedo/normal |
| `--adapter` | string | `RTX 4060` | Host adapter name substring; no silent adapter fallback |
| `--fp32` | flag | off | Omit `SHADER_F16` |
| `--no-recommended` | flag | off | Request only required wgpu features |
| `--shared-queue` | flag | off | Vulkan only; host producer and denoiser share one queue |
| `--unchecked` | flag | off | Disable shader runtime checks for both paths |
| `--no-debug` | flag | off | DX12 only; explicitly skip debug-layer validation |
| `--trace` | flag | off | Print preparation and submission progress to stderr |

Raw stdout, stderr, arguments, executable/weight hashes, drivers and validation settings go under `.native-build/interop/`; the runner writes a backend summary after each case. Invalid offsets and parameters are rejected before staging synchronization, duplicate native imports are rejected, and device/LUID mismatches are checked before wrapping. A controlled post-acceptance encoding failure uses an empty GPU batch with the original wait/signal; its output is not consumed. A zero-timeout wait exercises timeout reporting without inducing a GPU fault.

Wait for the external completion value when testing host notification. DX12 HAL signals its internal wgpu fence before enqueuing external fence signals; GPU validation exposed a case where `Device::poll(Wait)` returned while the external counter still held its previous value. The example now waits for the external value before asserting it. A common C completion-query contract must specify whether it includes that external signal operation; an internal wgpu submission index alone does not establish that it has occurred.

These examples do not define the GPU C ABI, bind-group range cache, production queue-lock callbacks, or a production submission/error state machine. The test executables terminate on device loss or an uncaptured wgpu error instead of reusing uncertain in-flight state; this process-exit policy is not a reusable library recovery contract. Device loss, uncertain submission, recovery across independently lost devices, hidden aliases and long-running reclamation beyond the recorded runs remain unverified. An empty batch after a controlled encoding failure does not establish that signaling is recoverable after device loss. A host CPU can advance a shared signal only while its own device and synchronization object remain usable. GPU denoise completion also does not permit output reuse until the host consumer has finished.

### Native Metal profiling

On macOS with Xcode installed, Instruments' Metal System Trace records CPU/GPU scheduling and Metal activity. The `profiling` profile retains symbols and line tables for CPU stack traces. Run from the repository root:

```bash
cargo build --profile profiling --manifest-path native/Cargo.toml --bins
mkdir -p benchmarks/results
xcrun xctrace record --template 'Metal System Trace' \
  --output benchmarks/results/large-native.trace --time-limit 12s --launch -- \
  "$PWD/native/target/profiling/oidn-webgpu-denoise-bench" \
  --weights "$PWD/weights/rt_hdr_calb_cnrm_large.tza" \
  --width 1280 --height 720 --warmup 5 --runs 100
```

Xcode 27's `gpucapture` and `gpudebug` also capture and replay compute workloads for per-dispatch costs and shader statistics such as register use and spills. Start a longer run in one terminal:

```bash
MTL_CAPTURE_ENABLED=1 native/target/profiling/oidn-webgpu-denoise-bench \
  --weights weights/rt_hdr_calb_cnrm_large.tza \
  --width 1280 --height 720 --warmup 5 --runs 400
```

In another terminal, list the process, then replace `PID` with its ID:

```bash
xcrun gpucapture list
xcrun gpucapture start --pid PID --count 64 \
  --output "$PWD/benchmarks/results/large-native.gputrace" \
  --disable-backtrace-recording
xcrun gpudebug --gputrace benchmarks/results/large-native.gputrace
```

At the `gpudebug>` prompt:

```text
profile run --exec serial
profile embed
go /performance/shaders
list --all
info comp0
exit --terminate
```

Capture enough command buffers to include complete network executions; a single buffer can contain only initialization, preprocessing or output conversion. Shader indices depend on the trace, and one pipeline can serve multiple layers. Replay costs help locate hotspots; they do not replace the synchronized full-image latency benchmark. Capture/replay adds overhead and can save hundreds of MiB, so run it separately from timing comparisons. Metal timestamp queries returned zero or invalid samples in local tests; validate them before using them as GPU timings.

## CI and native packages

When `native/Cargo.toml` is the root manifest, release builds use one codegen unit for the native crate and its dependencies and request symbol stripping. Thin LTO applies to benchmark and test executables; Cargo skips it for the mixed `rlib`/`cdylib`/`staticlib` library target and downgrades symbol stripping there to debug-info stripping. The default optimization level and panic unwinding are retained: the C API relies on `catch_unwind` to convert Rust panics to status codes. Downstream Rust applications select their own workspace's profiles.

[`.github/workflows/native.yml`](../.github/workflows/native.yml) runs on pull requests and pushes to `main` that touch `native/`, `src/`, `weights/` or the native scripts, on `v*` / `native-v*` tags, and on manual dispatch. For each target it:

1. builds the core archive and C package: `node scripts/native-build.mjs --target <triple> --archive --require-clean`;
2. verifies the archive without a GPU: `node scripts/native-package-check.mjs <archive>` (checksum, C ABI test `native/tests/abi.c`, CMake consumers using the relocated package, planner run, installed applications with the SDK hidden, the packaged standalone Rust application linked to the packaged core and run with `--inspect-model`). Its Cargo manifest pins the package source commit; CI substitutes that same local checkout to stay offline. On Windows it also compiles the packaged `examples/validate_windows.c`; run GPU validation separately as described in the [native guide](../native/README.md);
3. uploads the `.tar.gz` and `.sha256` as an artifact.

The workflow also runs native unit/core-integrity tests and runs `npm test` on Linux. It assembles a crate containing all four verified cores, then runs isolated `cargo publish --dry-run` and Rust application checks on every target. No GPU execution runs in CI. The [native release guide](native-release.md) describes permanent assets and the manually triggered publication workflow.

Windows builds use PowerShell to keep Git Bash's `link.exe` out of the MSVC linker search. The pinned scriptc 0.3.2 Windows runtime uses MinGW CRT spellings and GNU `.ctors` initialization. `scripts/native-core-msvc.mjs` registers those constructors with the MSVC CRT and adds compatibility objects compiled from `native/toolchain/msvc-compat.c` and Zig's public-domain MinGW directory implementation. The manifest records the Zig version. Package checks cover the Windows 128-bit division ABI against JavaScript BigInt, CRT aliases, directory access and planner execution; unresolved runtime symbols or skipped constructors must fail those checks.

## Documentation map

| File | Audience | Keep in sync with |
| --- | --- | --- |
| `README.md` | Everyone; npm package page | Runtimes, weights table, doc index |
| `docs/web.md` | Browser API users | `src/main.ts`, `src/UNet.ts`, `src/tileScheduler.ts` |
| `native/README.md` | Rust and C users; also shipped as the package README | `native/src/denoiser.rs`, `native/src/gpu_capi/`, `native/include/`, `scripts/native-build.mjs` |
| `docs/native-release.md` | Maintainers releasing native packages and the crate | `.github/workflows/native*.yml`, `scripts/native-release.mjs`, `native/build.rs` |
| `docs/development.md` | Contributors | Scripts, CI, architecture |
| `AGENTS.md` | Coding agents | Repository rules |
| `llms.txt` | LLM tools ([llmstxt.org](https://llmstxt.org)) | Doc index; absolute `main` branch URLs. Update when a doc is added, renamed or re-scoped. |
| `CHANGELOG.md` | Users upgrading | Every user-visible change, under `[Unreleased]` |
| `plans/` | Maintainers | Local working notes; ignored by Git |
