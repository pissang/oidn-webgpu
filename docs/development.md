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
| `native/src/bin/oidn-webgpu-bench.rs` | Native network benchmark |
| `scripts/native-*.mjs` | Core build, C package build, package identity, package checks |

## Commands

Run from the repository root. Requirements: Node 22+; for native work also stable Rust and a C toolchain.

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

Tests import compiled output from `lib/` and `dist/`, so always run them through `npm test`. Generated directories (`lib/`, `dist/`, `.native-build/`, `native/target/`, `benchmarks/results/`) are git-ignored.

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
| `node benchmarks/mac-performance.mjs --oidn-library /path/to/libOpenImageDenoise.dylib` | Pinned 0.5.0/current Chrome, Chrome subgroup matrix, `oidn-web(native wgpu)` and official OIDN Metal; Small/Base/Large at 720p, 1080p and 2560x1440. Builds both revisions and native benchmark automatically. |
| `cargo run --release --manifest-path native/Cargo.toml --bin oidn-webgpu-denoise-bench -- --weights W --width N --height N` | Native full-image GPU-buffer latency, including preprocessing and output conversion. Runtime checks off by default; `--runtime-checks` enables them. `--output PATH` saves packed RGBA32F output after timing. |
| `node scripts/*GpuCheck.mjs` | Correctness checks in Chrome: kernels against CPU references, and input/output sanitizing (`sanitizeGpuCheck.mjs`). |

Each script's header comment lists its options.

The Mac suite requires Node 22+, Chrome, Rust, a C compiler and an OIDN SDK with Metal. `CHROME_PATH` and `OIDN_LIBRARY` may supply the executable/library paths. Run `node benchmarks/mac-performance.mjs --help` for model, size, repetition and revision overrides. It builds detached temporary worktrees, so uncommitted runtime edits are excluded; temporary worktrees are removed after the run. GPU cases run sequentially.

Default inputs are `720p` (1280x720), `1080p` (1920x1080) and `2k` (2560x1440). Use `--sizes 720p` to measure only 720p, or `--sizes 512` for the 512x512 regression case.

The suite writes raw samples, model and binary hashes, source revisions, selected kernels, resource sizes and a Markdown report to `benchmarks/results/mac-performance.{json,md}`. Its metric is synchronized latency for one GPU-resident RGBA32F HDR image with albedo and normal. It includes preprocessing, network execution and output conversion, and excludes uploads, readback, device creation and preparation. Default protocol: 3 warmups, 7 measured images, whole-image processing, `inputScale=1`. Native runtime checks are disabled to match the C API default. Official OIDN loads the same TZA weights; Small uses fast quality, Base uses high, and Large uses high with clean auxiliary images. This is a timing comparison, not an image-quality comparison.

The changelog combines separate runs. Repeat the earlier Chrome and 1080p/2k comparison with OIDN 2.5.1 and:

```bash
OIDN_LIBRARY=/absolute/path/to/libOpenImageDenoise.dylib node benchmarks/mac-performance.mjs \
  --current-ref cd1ff2f --warmup 5 --runs 21
```

For a new 720p comparison at the native optimization revision (including fresh Chrome measurements), use:

```bash
OIDN_LIBRARY=/absolute/path/to/libOpenImageDenoise.dylib node benchmarks/mac-performance.mjs \
  --current-ref 921236e --models small,base,large --sizes 720p --warmup 5 --runs 21
```

### Native Metal profiling

On macOS with Xcode installed, Instruments' Metal System Trace records CPU/GPU scheduling and Metal activity. Run from the repository root:

```bash
cargo build --release --manifest-path native/Cargo.toml --bins
mkdir -p benchmarks/results
xcrun xctrace record --template 'Metal System Trace' \
  --output benchmarks/results/large-native.trace --time-limit 12s --launch -- \
  "$PWD/native/target/release/oidn-webgpu-denoise-bench" \
  --weights "$PWD/weights/rt_hdr_calb_cnrm_large.tza" \
  --width 1280 --height 720 --warmup 5 --runs 100
```

Xcode 27's `gpucapture` and `gpudebug` also capture and replay compute workloads for per-dispatch costs and shader statistics such as register use and spills. Start a longer run in one terminal:

```bash
MTL_CAPTURE_ENABLED=1 native/target/release/oidn-webgpu-denoise-bench \
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

[`.github/workflows/native.yml`](../.github/workflows/native.yml) runs on pull requests and pushes to `main` that touch `native/`, `src/`, `weights/` or the native scripts, on `v*` / `native-v*` tags, and on manual dispatch. For each target it:

1. builds the core archive and C package: `node scripts/native-build.mjs --target <triple> --archive --require-clean`;
2. verifies the archive without a GPU: `node scripts/native-package-check.mjs <archive>` (checksum, C ABI test `native/tests/abi.c`, C examples, planner run, a standalone Rust consumer linked to the packaged core);
3. uploads the `.tar.gz` and `.sha256` as an artifact.

No GPU execution runs in CI. The npm unit tests are not run by this workflow; run `npm test` locally.

## Documentation map

| File | Audience | Keep in sync with |
| --- | --- | --- |
| `README.md` | Everyone; npm package page | Runtimes, weights table, doc index |
| `docs/web.md` | Browser API users | `src/main.ts`, `src/UNet.ts`, `src/tileScheduler.ts` |
| `native/README.md` | Rust and C users; also shipped as the package README | `native/src/denoiser.rs`, `native/include/oidn_webgpu.h`, `scripts/native-build.mjs` |
| `docs/development.md` | Contributors | Scripts, CI, architecture |
| `AGENTS.md` | Coding agents | Repository rules |
| `llms.txt` | LLM tools ([llmstxt.org](https://llmstxt.org)) | Doc index; absolute `main` branch URLs. Update when a doc is added, renamed or re-scoped. |
| `CHANGELOG.md` | Users upgrading | Every user-visible change, under `[Unreleased]` |
| `plans/` | Maintainers | Working notes; not user documentation |
