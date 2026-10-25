# AGENTS.md

Guidance for coding agents (and humans) changing this repository.

oidn-webgpu runs the Open Image Denoise U-Net on WebGPU. One TypeScript core in `src/` powers two runtimes: the browser package (npm `oidn-webgpu`) and a native library, where scriptc compiles the core into a static library that the Rust crate in `native/` replays on wgpu (Rust and C APIs).

## Repository map

```
src/              TypeScript library: shared core + browser runtime
src/native/       Native core entry for scriptc (program recording, wire formats)
native/           Rust crate oidn-webgpu-native: wgpu replay, Rust Denoiser, C API
native/include/   Public C headers (ABI)
tests/            node:test suites; they import built output from lib/ and dist/
examples/         Browser demo pages (served by `npm run dev`)
benchmarks/       Browser, native and OIDN SDK comparisons (need Chrome and/or a GPU)
scripts/          Model inspection, native build/package/check, GPU kernel checks
weights/          Upstream OIDN TZA weights (binary, Apache 2.0); used by tests and examples
docs/             User guide (web.md) and contributor guide (development.md)
plans/            Maintainer working notes, in Chinese; not user documentation
```

Read [docs/development.md](docs/development.md) for the architecture diagram and the full list of shared vs. browser-only files.

## Commands

Run from the repository root.

```bash
npm ci
npm test                                                    # builds lib/ + dist/, runs unit tests; no GPU
cargo test --manifest-path native/Cargo.toml                # native unit tests; no GPU
cargo test --manifest-path native/Cargo.toml -- --ignored   # native GPU tests
npm run native:build                                        # C package in .native-build/package
```

Benchmarks and GPU checks: [docs/development.md#benchmarks-and-gpu-checks](docs/development.md#benchmarks-and-gpu-checks).

## Rules

1. **Shared core affects native.** `src/tza.ts`, `modelSpec.ts`, `graphOptimizer.ts`, `nativeUNetCore.ts`, `finalRgbShader.ts`, `gpuTypes.ts`, `hdrTransfer.ts`, `sanitize.ts`, `tileScheduler.ts`, `resourceTracker.ts` and `src/native/*` are compiled into the native library. After editing them, run both `npm test` and the native unit tests.
2. **Keep shared code scriptc-compatible.** No DOM or browser globals; GPU access only through the `OIDNGPU*` types in `src/gpuTypes.ts`. Async GPU code goes in browser-only files (`src/nativeUNet.ts`, `src/UNet.ts`).
3. **Wire formats change in pairs.** The config and program formats documented at the top of `src/native/program.ts` are parsed by `native/src/lib.rs`. Change both sides and bump the format version.
4. **The C header is an ABI.** `native/include/oidn_webgpu.h` must match `native/src/capi.rs`: function signatures, struct layout and enum values. Follow the header's versioning (`OIDNW_API_VERSION` and `oidnw_api_version()`): a minor bump may only add; removals and layout changes need a major bump. `native/tests/abi.c` checks defaults and error reporting.
5. **wgpu is pinned to `=30.0.1`.** Rust hosts must share the exact version, so changing it is a breaking change.
6. **CI has no GPU.** Unit tests use mock devices. For kernel, precision or tiling changes, run the GPU checks on real hardware; if you cannot, say in your summary that GPU execution was not verified.
7. **Changelog.** User-visible changes go under `[Unreleased]` in `CHANGELOG.md`, with migration notes for breaking changes. Experimental options (`engine: 'webnn'`, `kernel`, `gemm`, `subgroupMatrix`) are not covered by semver.
8. **Generated files stay out of git:** `lib/`, `dist/`, `.native-build/`, `native/target/`, `benchmarks/results/`. Do not modify files in `weights/`.
9. **Plans stay local.** Do not add or commit files under `plans/`.

## Documentation rules

- Update the document that owns the changed API; the ownership table is in [docs/development.md#documentation-map](docs/development.md#documentation-map). When you add, rename or re-scope a doc, update the index in [llms.txt](llms.txt) too.
- `native/README.md` is also copied to the native package root as `README.md`. Its relative links may only point inside `native/` (`include/`, `examples/`), so they resolve in both places. Link anything else with an absolute `https://github.com/pissang/oidn-webgpu/...` URL.
- Style: English; facts, defaults, units and ranges instead of adjectives. Options go in tables (name, type, default, meaning), contracts in bullet lists, commands in copy-pasteable blocks run from the repository root. State each fact once and link to it elsewhere.

## Before you finish

- [ ] `npm test` passes.
- [ ] Native unit tests pass if you touched the shared core or `native/`.
- [ ] Docs and `CHANGELOG.md` reflect user-visible changes.
- [ ] Your summary says what was and was not verified on a GPU.
