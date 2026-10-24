# Browser guide

API guide for the npm package `oidn-webgpu`. Sources of truth: [`src/main.ts`](../src/main.ts) (exports, `UNetOptions`) and [`src/UNet.ts`](../src/UNet.ts) (`UNet`, `tileExecute`).

- [Requirements](#requirements)
- [Load a model](#load-a-model)
- [Input formats](#input-formats)
- [Run: `tileExecute`](#run-tileexecute)
- [Recipes](#recipes)
- [Performance](#performance)
- [Inspect and profile](#inspect-and-profile)
- [Experimental options](#experimental-options)
- [New or custom OIDN models](#new-or-custom-oidn-models)

## Requirements

- A browser with WebGPU.
- Optional device features, used when enabled on the `GPUDevice`:

  | Feature | Effect |
  | --- | --- |
  | `shader-f16` | FP16 activations and weights (default when present) |
  | `timestamp-query` | Per-layer GPU profiling |
  | `subgroups`, `chromium-experimental-subgroup-matrix` | Experimental `kernel: 'subgroup-matrix'` |

Without a `device`, the library requests a high-performance adapter, every feature above that the adapter supports, and the adapter's buffer and workgroup limits. With your own `device`, request these features and limits when you create it; WebGPU cannot enable them later.

## Load a model

```ts
initUNetFromURL(url: string, backend?: { device: GPUDevice }, options?: UNetOptions): Promise<UNet>
initUNetFromBuffer(tza: ArrayBuffer, backend?: { device: GPUDevice }, options?: UNetOptions): Promise<UNet>
```

The promise resolves after the TZA file is validated and shape-independent shaders are compiled. Pick the weights file with the [table in the README](../README.md#choose-weights).

| `UNetOptions` | Type | Default | Meaning |
| --- | --- | --- | --- |
| `aux` | `boolean` | `false` | The model reads albedo and normal. Both are then required by `tileExecute`. |
| `hdr` | `boolean` | `false` | Color is linear HDR. Must match the weights. |
| `hdrTransfer` | `'pu' \| 'log'` | `'pu'` | `'log'` for RTLightmap weights. Ignored for LDR. |
| `precision` | `'auto' \| 'fp16' \| 'fp32'` | `'auto'` | `auto` uses FP16 when the device has `shader-f16`. `fp16` throws without it. FP16 output differs slightly from FP32. |
| `maxTileSize` | `number` | `512` | Hard upper bound for an output tile edge, in pixels. |
| `dynamicTile` | `boolean \| DynamicTileOptions` | `true` | Adaptive tile size; see [Tiling](#tiling-and-scheduling). |
| `modelSpec` | `UNetModelSpec` | detected | Graph for weights that no built-in descriptor matches; see [New or custom OIDN models](#new-or-custom-oidn-models). |
| `engine`, `kernel`, `gemm`, `subgroupMatrix` | | | [Experimental](#experimental-options). |

Loading throws if `aux` does not match the number of inputs in the weights.

Call `unet.dispose()` to release GPU resources. The shared `GPUDevice` is not destroyed.

## Input formats

All images are RGBA, row-major, top row first, with the same width and height.

| Input | CPU, LDR | CPU, HDR | GPU (`hdr: true`, `aux: true`) |
| --- | --- | --- | --- |
| `color` | `ImageData` | `{ data: Float32Array, width, height }`, linear, nonnegative | `{ data: GPUBuffer \| GPUTexture, width, height }` |
| `albedo` | `ImageData` | `ImageData` | GPU buffer or texture, linear RGB in [0, 1] |
| `normal` | `ImageData` | `ImageData` | GPU buffer or texture, encoded as `n * 0.5 + 0.5` in [0, 1] |
| Output | `ImageData`; alpha set to 255 | `{ data: Float32Array, width, height }`; alpha set to 1 | Library-owned `GPUBuffer`, RGBA f32; alpha copied from `color` |

- CPU 8-bit values are divided by 255.
- Normals are always [0, 1]-encoded (`n * 0.5 + 0.5`), on both CPU and GPU. That is the standard 8-bit normal-map encoding and exactly what the network reads, so no remap is applied. OIDN and the [native API](../native/README.md#image-contract) instead take signed [-1, 1] normals and encode them internally.
- CPU HDR color is scaled per tile by its average log luminance before the HDR transfer, and the scale is undone on output.
- GPU buffers hold tightly packed `vec4<f32>` pixels (16 bytes per pixel) with `STORAGE` usage. Textures are read with `textureLoad` from a float `texture_2d`.
- GPU HDR color is used at scale 1. Pre-scale the whole image yourself (for example by exposure) if its values are far from 1.
- The GPU output buffer is reused by the next execution and freed by `dispose()`. Copy it if you need it longer.
- GPU inputs are supported with `hdr: true` and `aux: true` models.
- Inputs are sanitized like OIDN: NaN becomes 0, then color is clamped to [0, 1] (LDR) or [0, FLT_MAX] after scaling (HDR), and albedo and normals to [0, 1]. The network output is sanitized the same way before the inverse transfer, so output is never NaN or negative and LDR output is at most 1. Extreme HDR input (for example +inf) can still produce +inf output.

## Run: `tileExecute`

```ts
const abort: () => void = unet.tileExecute({ color, albedo, normal, done, progress, error, ...options });
```

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `color`, `albedo`, `normal` | see [Input formats](#input-formats) | | `albedo` and `normal` are required exactly when the model was loaded with `aux: true`. |
| `done(output)` | callback, may return a promise | required | Called once with the full denoised image. |
| `progress(output, tileData, tile, index, count)` | callback, may return a promise | | Called after each tile. `tile` is the output rectangle `{ x, y, width, height }`. `tileData` holds that rectangle for CPU input and is `undefined` for GPU input. |
| `error(reason)` | callback | logs to console | Asynchronous failures: GPU errors, device loss, and exceptions thrown by `progress` or `done`. |
| `scheduling` | `'event-loop' \| 'animation-frame'` | `'event-loop'` | How JavaScript yields between tiles; see [Tiling](#tiling-and-scheduling). |
| `wholeImage` | `boolean` | `false` | Run the image as one tile, ignoring `maxTileSize`. The image must fit the device's buffer and dispatch limits. |
| `tileOverlap` | `number` | half the receptive field, rounded up to 16 | Context added on each edge shared with another tile. Smaller is faster, with a risk of visible seams. |
| `denoiseAlpha` | `boolean` | `false` | GPU input only: denoise the alpha channel and pass RGB through. |

Contract:

- Input validation errors throw synchronously: missing or unexpected `albedo`/`normal`, and mismatched sizes of CPU images.
- Unless `abort()` is called first, every execution ends with exactly one call to `done` or `error`. After `abort()`, no further callbacks run.
- At most one tile is in flight on the GPU, so `abort()` takes effect after the current tile.
- To restart with new input, call the previous execution's `abort()` first, as the examples do.

## Recipes

### HDR color with auxiliary images

```ts
const unet = await initUNetFromURL('/weights/rt_hdr_alb_nrm.tza', undefined, {
  hdr: true,
  aux: true
});

unet.tileExecute({
  color: { data: colorF32, width, height }, // Float32Array RGBA, linear
  albedo: new ImageData(albedoU8, width, height), // 8-bit RGBA
  normal: new ImageData(normalU8, width, height), // 8-bit RGBA
  done(denoised) {
    // denoised.data is a Float32Array RGBA
  }
});
```

### GPU pipeline with a shared device

Keep data on the GPU when your renderer is already WebGPU. Request optional features on your device first.

```ts
const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
const requiredFeatures = (['shader-f16', 'timestamp-query'] as const).filter((f) =>
  adapter!.features.has(f)
);
const device = await adapter!.requestDevice({ requiredFeatures });

const unet = await initUNetFromURL('/weights/rt_hdr_calb_cnrm.tza', { device }, {
  hdr: true,
  aux: true
});

unet.tileExecute({
  color: { data: colorBuffer, width, height },
  albedo: { data: albedoBuffer, width, height },
  normal: { data: normalBuffer, width, height },
  scheduling: 'animation-frame',
  done({ data: outputBuffer }) {
    // outputBuffer: GPUBuffer, RGBA f32, owned by unet
  },
  error: console.error
});
```

See [`examples/gpu.ts`](../examples/gpu.ts) for a complete page.

### RTLightmap weights

```ts
const lightmap = await initUNetFromURL('/weights/rtlightmap_hdr.tza', undefined, {
  hdr: true,
  hdrTransfer: 'log' // y -> log(1 + y) / log(65505), inverted on output
});
```

## Performance

### Warm up

Shape-specific GPU buffers and pipelines are created the first time a tile shape is seen. Prepare them while a loading state is visible:

```ts
await unet.prepareForImage(width, height); // same tiling as tileExecute
await unet.prepareForImage(width, height, { wholeImage: true }); // when you pass wholeImage
```

`prepareForImage` also accepts `tileOverlap`; pass the same values you will pass to `tileExecute`.

### Tiling and scheduling

Large images are split into balanced rectangular tiles. Each tile adds `tileOverlap` context only on edges shared with another tile, and at most two input shapes are used so the shape cache stays warm.

With `dynamicTile` enabled, the smoothed P75 GPU time of completed tiled executions adjusts the tile size used by the next execution. The cold first tile, cancelled executions and single-tile executions are ignored, and each layout is kept for at least two executions.

| `DynamicTileOptions` | Default | Meaning |
| --- | --- | --- |
| `initialTileSize` | `432` | Starting output tile edge |
| `minTileSize` | `256` | Lower bound |
| `targetTileTimeMs` | `16` | GPU time per tile to aim for |
| `adjustmentStep` | `16` | Pixels added or removed per adjustment |

`maxTileSize` is the upper bound. Pass `dynamicTile: false` to always use `maxTileSize` for deterministic tiling.

`scheduling` controls the pause between tiles:

- `'event-loop'`: continue on the next macrotask. Lowest latency; also completes in background tabs.
- `'animation-frame'`: at most one tile per display frame. Use it when an interactive renderer shares the GPU. If no frame arrives within 100 ms (for example in a hidden tab), the next tile runs anyway.

### Precision

FP16 halves activation and weight storage and is the default where `shader-f16` exists; its speedup depends on the GPU. Use `precision: 'fp32'` when results must not depend on `shader-f16` availability.

## Inspect and profile

`unet.getRuntimeInfo()` returns the resolved configuration and last execution statistics:

```ts
{
  gpuEngine: 'wgsl', precision: 'fp16', model: 'oidn-unet-large-v1',
  inputChannels: 9, hdrTransfer: 'pu',
  kernel: { configured, gemm, subgroupMatrix, ... },
  dynamicTile: { enabled, currentTileSize, minTileSize, maxTileSize, targetTileTimeMs },
  lastExecution: { tileCount, tileColumns, tileRows, durationMs, tileTimeMs: { min, median, mean, max }, ... },
  resources: { ... }, activeExecutionCount: 0
}
```

Per-layer GPU timings need `timestamp-query` on the device. Request a profile right before an execution:

```ts
if (unet.profileNextExecution()) {
  unet.tileExecute({
    color,
    async done() {
      const profile = await unet.getLastExecutionProfile();
      console.table(profile?.layers); // { id, durationMs }[], summed over all tiles
    }
  });
}
```

Timings cover the network only: they exclude image preprocessing, output conversion and CPU waits.

## Experimental options

These options exist for benchmarking and hardware experiments. They may change or disappear in any release and are not covered by semver.

| Option | Values | Meaning |
| --- | --- | --- |
| `engine` | `'auto'`, `'wgsl'`, `'webnn'` | `auto` and `wgsl` use the WGSL runtime. `webnn` uses WebNN where the browser provides it. |
| `kernel` | `'auto'`, `'implicit-gemm'`, `'direct'`, `'spatial'`, `'subgroup'`, `'subgroup-matrix'` | Convolution kernel. `auto` uses implicit GEMM, with a direct kernel for the final layer. `direct` reproduces the 0.4.0 FP16 path. |
| `gemm` | `NativeUNetGemmOptions` | Implicit-GEMM tuning. `{ decoderSubPixel: false }` restores the 0.5.0 FP16 decoder calculation. |
| `subgroupMatrix` | `NativeSubgroupMatrixOptions` | Tuning for `kernel: 'subgroup-matrix'`, which needs `subgroups` and `chromium-experimental-subgroup-matrix` on the device (Chrome with `--enable-unsafe-webgpu`). |

Field-level documentation is in the type definitions in [`src/nativeUNetCore.ts`](../src/nativeUNetCore.ts).

## New or custom OIDN models

TZA files store tensors but not the graph. The graph lives in a versioned `UNetModelSpec`. Built-in descriptors (`OIDN_UNET_SMALL_SPEC`, `OIDN_UNET_LARGE_SPEC`) cover the current OIDN RT U-Nets and are detected from the tensor names. Tensor layout, dtype, sizes and channel flow are validated before GPU resources are created.

- Same topology and tensor names, different channel widths: works without changes.
- Added or renamed nodes: pass `modelSpec`, or add a built-in descriptor (see [development guide](development.md#updating-to-a-new-oidn-model)).

```ts
const unet = await initUNetFromURL(url, undefined, { hdr: true, aux: true, modelSpec: myModelSpec });
```

Lower-level exports for tools: `parseTZA`, `detectUNetModelSpec`, `validateUNetModel`, `optimizeModelGraph`, `planModelExecution`, `planTileGrid`.
