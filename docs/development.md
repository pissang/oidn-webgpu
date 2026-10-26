# Development guide

How the code is organized, how to build and test it, and how to change it safely. For working rules aimed at coding agents, see [AGENTS.md](../AGENTS.md).

- [Architecture](#architecture)
- [Commands](#commands)
- [Validation scope](#validation-scope)
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
                   native/src/gpu_capi/    GPU C API, shared ledger, platform imports
```

The browser and native runtimes share one TypeScript core. The native build compiles it to a static library with scriptc (version pinned in `native/toolchain/package.json`); `RecordingDevice` captures buffers, shaders and dispatches as a program, and Rust replays that program on a wgpu device.

### Shared core vs browser-only code

| Compiled into the native core (shared) | Browser-only |
| --- | --- |
| `src/tza.ts`, `src/modelSpec.ts`, `src/graphOptimizer.ts`, `src/nativeUNetCore.ts`, `src/finalRgbShader.ts`, `src/gpuTypes.ts`, `src/hdrTransfer.ts`, `src/sanitize.ts`, `src/tileScheduler.ts`, `src/resourceTracker.ts`, `src/native/*` | `src/main.ts`, `src/UNet.ts`, `src/nativeUNet.ts`, `src/process.ts`, `src/backend.ts`, `src/webnnUNet.ts`, `src/WGPUComputePass.ts`, `src/WGPUFullQuadPass.ts`, `src/hdr.ts` |

Consequences:

- A change to a shared file changes native output too. Run the native checks below.
- Shared code must compile with scriptc: no DOM or browser globals, and GPU access only through the `OIDNGPU*` types in `src/gpuTypes.ts`. Async GPU work belongs in browser-only code.
- `native/build.rs` watches `src/native/`, the shared source files listed in `src/native/core-inputs.json`, and the native build/toolchain scripts. Browser-only changes do not rebuild the core. When shared imports change, update the list; `tests/buildInputs.test.mjs` checks it against the transitive imports of the profile's entry, including type-only imports.
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
| `native/src/gpu_capi/` | GPU C handles, bounded completion ledger, binding cache and platform adapters; matches the common GPU and platform C headers, including Vulkan creation requirements |
| `native/cuda/`, `native/include/oidn_webgpu_cuda.hpp` | Optional Windows CUDA/DX12 adapter, built separately from the core DLL; CUDA stream ordering, shared-image leases and bounded resource pool |
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
| Local unit test loop (checks build cache) | `npm run test:fast` | no |
| Examples dev server (HTTPS) | `npm run dev`, then open `/examples/basic.html` | browser |
| Inspect a TZA file | `npm run model:inspect -- weights/rt_hdr_alb_nrm.tza` | no |
| Resource leak check | `npm run test:leak -- [--cycles 6] [--engine wgsl\|webnn]` | Chrome |
| Native crate build | `cargo build --release --manifest-path native/Cargo.toml` | no |
| Native unit tests | `cargo test --manifest-path native/Cargo.toml` | no |
| Native GPU tests | `cargo test --manifest-path native/Cargo.toml -- --ignored` | yes |
| C package | `npm run native:build` (output in `.native-build/package`) | no |

Tests import compiled output from `lib/` and `dist/`. Use `npm test` for the complete JS check with unconditional builds, or `npm run test:fast` during local iteration. The local runner follows imports through the selected tests and local helpers, including code fixtures referenced by literal `new URL(...)` expressions. It builds only the imported artifacts: library tests need `lib/`, bundle tests need `dist/`, and tests of standalone scripts need neither. Use the complete check for fixtures loaded through computed paths that static dependency discovery cannot follow.

The runner hashes source files, build configuration, dependency lockfile, tool versions and generated output before reusing a needed build. Missing or modified output triggers a rebuild. Build fingerprints are stored in `.native-build/test-fast.json`; deleting that file forces needed builds on the next run. Test results are never cached.

Pass test filenames to narrow the local loop:

```shell
npm run test:fast -- tests/tileExecute.test.mjs
npm run test:fast -- tests/modelSpec.test.mjs tests/nativeGemm.test.mjs
```

For native unit-test iteration, select the library or a specific integration test to avoid building unrelated examples and benchmark targets:

```shell
cargo test --manifest-path native/Cargo.toml --lib
cargo test --manifest-path native/Cargo.toml --test core_distribution
```

Keep the toolchain, target, profile, features and `OIDN_WEBGPU_CORE_ARCHIVE` setting consistent across iterations so Cargo can reuse its outputs. Changing the archive override switches the build-script inputs; use it only with a source-matched archive. Add `--timings` to a Cargo command to inspect compilation costs in `native/target/cargo-timings/`.

Generated directories (`lib/`, `dist/`, `.native-build/`, `.zig-cache/`, `native/target/`, `benchmarks/results/`) are git-ignored. The Windows CI Zig setup action puts its compiler cache in `.zig-cache/`; it does not affect the package's clean-source check.

## Validation scope

During iteration, run the tests for the affected behavior. Once the relevant changes settle, run each complete suite required by this table once; repeat only after another relevant edit or a failure. A completed suite already builds its required outputs, so no separate preceding build is needed. Run `npm ci` when installing or changing locked dependencies, rather than before each test run.

| Changed scope | Final checks |
| --- | --- |
| Documentation and prose only | Review content and links; no builds or tests |
| Browser-only TypeScript implementation | `npm test`; no native build |
| Shared TypeScript implementation | `npm test` and `cargo test --manifest-path native/Cargo.toml` |
| Native Rust implementation | `cargo test --manifest-path native/Cargo.toml`; no JS build |
| Native build scripts, core profile or input list | `node --test tests/buildInputs.test.mjs` and complete native unit tests; include JS tests when JS build/test tooling also changes |
| JS tests or build/test tooling | `npm test`; native checks only if native dependencies also change |
| Native examples, C/C++ headers or C ABI | Compile affected consumers and run the corresponding ABI/example checks; include native unit tests for implementation changes |
| Package/release scripts, shipped files or toolchain | Corresponding package/crate checks; avoid release packaging for unrelated runtime edits |

For kernel, precision or tiling changes, also run the affected real-GPU correctness checks. For GPU interop changes, start with the affected backend and a short frame count, then run the relevant stress matrix before finishing. Do not disable checks or reuse stale binaries to shorten a run. `-SkipBuild` is appropriate only when source and build inputs are unchanged since that binary was built. GPU benchmarks are needed for performance claims, not for documentation or build-dependency changes.

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

The suite writes raw samples, model and binary hashes, source revisions, selected kernels, resource sizes and a Markdown report to `benchmarks/results/mac-performance.{json,md}`. Its metric is synchronized latency for one GPU-resident RGBA32F HDR image with albedo and normal. It includes preprocessing, network execution and output conversion, and excludes uploads, readback, device creation and preparation. Default protocol: 3 warmups, 7 measured images, whole-image processing, `inputScale=1`. Native benchmark runtime checks are disabled to match the standalone CPU C filter default. Official OIDN loads the same TZA weights; Small uses fast quality, Base uses high, and Large uses high with clean auxiliary images. This is a timing comparison, not an image-quality comparison.

Repeat the changelog's Apple M4 Max comparison with OIDN 2.5.1 and:

```bash
OIDN_LIBRARY=/absolute/path/to/libOpenImageDenoise.dylib node benchmarks/mac-performance.mjs \
  --current-ref fd0d816 --warmup 5 --runs 21
```

When one case is slower in every column, OIDN Metal included, remeasure only that case, for example `--models large --sizes 720p`.

### Native Metal interoperability checks

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

The standalone [Objective-C consumer](../native/examples/metal_gpu.m) uses the shipped headers and shared library; its build and usage are in the [GPU C API guide](../native/README.md#gpu-c-api). CI compiles it on macOS through the packaged CMake examples but does not run GPU work.

### Native Windows GPU interoperability

The platform C contracts, creation options and native ownership rules are in the [GPU C API guide](../native/README.md#gpu-c-api). `native/examples/vulkan_gpu.cpp` and `dx12_gpu.cpp` are independent DLL consumers. Native shaders produce color/albedo/normal, and a native consumer compares all output bits with serial executions. Only integer validation flags reach the CPU. Shader assets are built by `interop_shaders`; the package includes SPIR-V and the host HLSL source so consumers need no Rust or Node runtime.

`native/examples/vulkan_interop.rs` and `dx12_interop.rs` additionally compare formal C submissions with serial `PreparedDenoise` references on the same device, precision, kernel and runtime-check configuration. They rotate three buffer sets at offset 256, change contents and input scale on slot reuse, and cover partial 31×31 images with tile edge 16. Native command/descriptor retirement has a 64-pair bound. The C pending queue and binding cache have their own bounds. These correctness protocols use artificial delays and establish no performance result.

#### Backend boundaries

`native/src/gpu_capi/mod.rs` owns handles, image roles, binding caches, submission numbering, admission, pending-resource retention, completion and quarantine. `ledger.rs` first checks an eligible prefix without dropping resources; any native timeline-query error retains the prefix. Retirement holds the health lock so loss/error callbacks cannot race reclamation. The platform modules implement `Backend` and fallible `Timeline`; they do not copy the runtime. Queue locks cover initialization, uploads, submissions, polling and final queue destruction.

| Route | Implementation | Scope |
| --- | --- | --- |
| Vulkan A | Non-owning HAL views use the host loader, instance, physical device, device, queue, buffers and timeline semaphores. Host queue access shares paired C callbacks. | Executed on Windows. Dedicated and shared producer/denoiser queues are tested. Linux execution is unverified. |
| Vulkan B | The Rust probe queries external buffer/semaphore capabilities only. | No external-memory binding, ownership transfer or separate-device execution; no fallback. |
| DX12 | Matching Agility SDK; a distinct library device and queue on the host LUID. Temporary shared NT handles are opened and closed, and both original/opened COM identities are retained. | Committed DEFAULT UAV buffers and shared fences; no host device or queue import. Debug maintenance checks both devices. |

Vulkan import validates the original `VkDeviceCreateInfo` against the complete pinned HAL extension/feature chain, including implicitly required robustness, timeline and shader-int8 features. `vulkan_features.rs` is shared with the prototype and rejects each missing required boolean in its negative checks. Individual canonical feature structs are required; Vulkan 1.1/1.2/1.3 aggregate structs fail closed. The supplied declarations must still be live during import. Physical-device support alone is insufficient.

Pinned HAL 30.0.1 chains Vulkan submissions with binary relay semaphores. Deep chains behind not-yet-submitted CPU timeline signals can trigger `VUID-vkQueueSubmit-pWaitSemaphores-03238`. The C backend conservatively rejects another submission with `QUEUE_FULL` while a pending external wait remains unresolved. It does not guess whether a future producer signal was already submitted. The host must release producer gates independently of the CPU submit call returning.

DX12 HAL may complete its internal fence before external fence signals. The common ledger checks all external signals too. `GetCompletedValue()==UINT64_MAX` is a device-removal error, never completion. Real removal is not induced by the tests; the sentinel conversion has a GPU-independent unit test.

#### Reproduce

Run from an x64 VS developer PowerShell with Rust's MSVC toolchain, Node 22+ and Zig 0.16.0 on PATH. A source-matched `OIDN_WEBGPU_CORE_ARCHIVE` can replace the planner build. Install the Khronos validation layer and use the matching Agility SDK binaries and headers:

```powershell
$env:VK_LAYER_PATH = 'C:/VulkanSDK/<version>/Bin'
$env:WGPU_DX12_AGILITY_SDK_PATH = 'C:/SDKs/Agility/build/native/bin/x64/'
$env:WGPU_DX12_AGILITY_SDK_VERSION = '619'
$env:WGPU_DX12_AGILITY_SDK_REQUIRE = '1'
pwsh -File scripts/native-interop-check.ps1 -Backend vulkan -VulkanIncludeDirectory "$env:VULKAN_SDK/Include"
pwsh -File scripts/native-interop-check.ps1 -Backend vulkan -RustOracle
pwsh -File scripts/native-interop-check.ps1 -Backend vulkan -Frames 12 -GpuValidation
pwsh -File scripts/native-interop-check.ps1 -Backend dx12 -Dx12IncludeDirectory 'C:/SDKs/Agility/build/native/include'
pwsh -File scripts/native-interop-check.ps1 -Backend dx12 -RustOracle
pwsh -File scripts/native-interop-check.ps1 -Backend dx12 -Unchecked
pwsh -File scripts/native-interop-check.ps1 -Backend dx12 -RustOracle -Unchecked
pwsh -File scripts/native-interop-check.ps1 -Backend dx12 -Frames 12 -GpuValidation
pwsh -File scripts/native-interop-check.ps1 -Backend dx12 -Frames 12 -GpuValidation -Unchecked
cargo test --manifest-path native/Cargo.toml backpressure_and_failed_batch -- --ignored --nocapture
```

The runner defaults to the actual DLL C++ consumer. `-RustOracle` builds the source-only `interop-validation` feature for DX12 so its serial Rust reference uses the exact C device. That feature adds no C exports and does not change shader checks. `-Prototype` selects the original Rust path. `-SkipBuild` reuses existing artifacts; do not use it after source changes. Different protocols/settings should use distinct `-OutputDirectory` values; each directory contains its own `consumer/` executable, DLL and shader assets.

The six cases are Small/Base at 16×16 and Large at 31×31 with tile edge 16, in FP16 and FP32. Vulkan Base shares the producer queue; other Vulkan cases use a dedicated queue. DX12 always uses a dedicated library queue. `-Unchecked` explicitly disables bounds checks and workgroup initialization; defaults remain checked. `-NoDebug` is a DX12 diagnostic mode and supplies no debug validation evidence. The runner saves arguments, source/header/executable/DLL/model hashes, drivers, SDK environment, stdout, stderr and timeout/exit status as JSON under `.native-build/interop`. Timed-out or interrupted processes are not passes.

#### Recorded correctness and diagnostics

On 2026-10-25: Windows 11 build 26200, RTX 4060 Laptop, NVIDIA 617.42, wgpu 30.0.1, Khronos layer API 1.4.366, Agility 1.619.6. Each row covers all six model/precision cases:

| Path | Frames per case | Shader checks | Native validation | Result |
| --- | ---: | --- | --- | --- |
| Vulkan DLL / serial C | 1000 | enabled | Khronos layer | Byte equal |
| Vulkan C / serial Rust, same host device | 1000 | enabled | Khronos layer | Byte equal |
| Vulkan DLL | 12 | enabled | GPU-assisted and synchronization validation, serialized diagnostic protocol | Byte equal |
| DX12 DLL / serial C | 1000 | enabled and disabled separately | Debug layer on both devices | Byte equal |
| DX12 C / serial Rust, same library device | 1000 | enabled and disabled separately | Debug layer on both devices | Byte equal |
| DX12 DLL | 12 | enabled and disabled separately | Debug layer and GPU validation on both devices | Byte equal |

Vulkan is **not clean shader validation**: Naga workgroup/ArrayStride decorations still trigger `VUID-StandaloneSpirv-None-10684`. The native messenger counts these separately in `known_naga_workgroup_layout_diagnostics`; all final cases report zero other validation errors. GPU-assisted validation uses a serialized protocol; asynchronous GPU-assisted validation with deeply gated future signals is not validated.

The GPU state-machine tests cover argument, range, alignment, role and duplicate-sync rejection, same-workspace `BUSY`, multi-workspace contention, queue backpressure, zero-timeout waits, in-flight handle release, controlled post-acceptance encoding failure, quarantine and quiescent teardown. DX12's state-machine fixture uses unchecked shaders; checked image correctness comes from the matrices above.

Not verified: real device removal and recovery, independently lost DX12 devices, hidden memory aliases, long-duration memory behavior, other GPUs and drivers, and performance.

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
2. verifies the archive without a GPU: `node scripts/native-package-check.mjs <archive>` (checksum, C ABI test `native/tests/abi.c`, CMake consumers using the relocated package, planner run, installed applications with the SDK hidden, the packaged standalone Rust application linked to the packaged core and run with `--inspect-model`). Its Cargo manifest pins the package source commit; CI substitutes that same local checkout to stay offline. On Windows it also compiles the packaged `examples/validate_windows.c`; run GPU validation separately as described in the [native guide](../native/README.md). The C++ wrapper test `native/tests/gpu_cpp.cpp` always runs; set `OIDNW_GPU_INTEROP_VULKAN_INCLUDE` to a Vulkan 1.3+ header directory to also run the Vulkan C ABI checks (Windows) and the Vulkan C++ helper test `native/tests/gpu_cpp_vulkan.cpp` (all platforms);
3. uploads the `.tar.gz` and `.sha256` as an artifact.

The workflow also runs native unit/core-integrity tests and runs `npm test` on Linux. It assembles a crate containing all four verified cores, then runs isolated `cargo publish --dry-run` and Rust application checks on every target. No GPU execution runs in CI. The [native release guide](native-release.md) describes permanent assets and the manually triggered publication workflow.

Windows builds use PowerShell to keep Git Bash's `link.exe` out of the MSVC linker search. The pinned scriptc 0.3.2 Windows runtime uses MinGW CRT spellings and GNU `.ctors` initialization. `scripts/native-core-msvc.mjs` registers those constructors with the MSVC CRT and adds compatibility objects compiled from `native/toolchain/msvc-compat.c` and Zig's public-domain MinGW directory implementation. The manifest records the Zig version. Package checks cover the Windows 128-bit division ABI against JavaScript BigInt, CRT aliases, directory access and planner execution; unresolved runtime symbols or skipped constructors must fail those checks.

## Documentation map

| File | Audience | Keep in sync with |
| --- | --- | --- |
| `README.md` | Everyone; npm package page | Runtimes, weights table, doc index |
| `docs/web.md` | Browser API users | `src/main.ts`, `src/UNet.ts`, `src/tileScheduler.ts` |
| `native/README.md` | Rust, C and C++ users; also shipped as the package README | `native/src/denoiser.rs`, `native/src/gpu_capi/`, `native/include/`, `scripts/native-build.mjs` |
| `native/examples/gpu-integration.md` | C API hosts adapting an existing renderer; shipped with native examples | `native/include/oidn_webgpu_gpu.h`, platform headers and native host examples |
| `native/examples/gpu-integration.md#windows-cuda-adapter` | CUDA renderer developers | `native/include/oidn_webgpu_cuda.hpp`, `native/cuda/`, `native/examples/cuda_dx12_gpu.cu` |
| `native/include/oidn_webgpu_*.hpp` | C++ GPU hosts | Common handle ownership and platform import helpers; mirrored by native examples |
| `docs/native-release.md` | Maintainers releasing native packages and the crate | `.github/workflows/native*.yml`, `scripts/native-release.mjs`, `native/build.rs` |
| `docs/development.md` | Contributors | Scripts, CI, architecture |
| `AGENTS.md` | Coding agents | Repository rules |
| `llms.txt` | LLM tools ([llmstxt.org](https://llmstxt.org)) | Doc index; absolute `main` branch URLs. Update when a doc is added, renamed or re-scoped. |
| `CHANGELOG.md` | Browser package users upgrading | Every user-visible browser package and repository tooling change, under `[Unreleased]` |
| `native/CHANGELOG.md` | Native library users upgrading | Every user-visible Rust crate, C/C++ header, package and CUDA adapter change, under `[Unreleased]`; versions follow `native/Cargo.toml` |
| `plans/` | Maintainers | Local working notes; ignored by Git |
