# oidn-webgpu

[Open Image Denoise](https://github.com/RenderKit/oidn) (OIDN) on WebGPU. It denoises path-traced images on the GPU, in browsers and natively through [wgpu](https://wgpu.rs).

> Formerly published as `oidn-web`; versions up to 0.5.0 remain available under that name.

|                                            2000 samples                                             |                                       3 samples                                       |                                    3 samples + denoised                                    |
| :-------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------: |
| ![](https://github.com/pissang/oidn-webgpu/blob/main/examples/test/ground-truth.png 'Ground Truth') | ![](https://github.com/pissang/oidn-webgpu/blob/main/examples/test/noisy.png 'Noisy') | ![](https://github.com/pissang/oidn-webgpu/blob/main/examples/test/denoised.png 'Denoised') |

## Runtimes

| Runtime | Status | Distribution | Guide |
| --- | --- | --- | --- |
| Browser (WebGPU) | Released (0.x) | npm `oidn-webgpu` | [docs/web.md](docs/web.md) |
| Rust, sharing the host's wgpu device and buffers | Experimental | Cargo git dependency | [native/README.md](native/README.md#rust-api) |
| C, CPU images in and out | Experimental | Build from source or CI artifacts | [native/README.md](native/README.md#c-api) |

Every runtime executes the same WGSL kernels, planned by the same TypeScript core. The [native guide](native/README.md) records tested GPU configurations and output validation.

## Quick start (browser)

```shell
npm i oidn-webgpu
```

Weights are not in the npm package. Copy a `.tza` file from [`weights/`](weights/) or [oidn-weights](https://github.com/RenderKit/oidn-weights) and serve it with your app.

```ts
import { initUNetFromURL } from 'oidn-webgpu';

const unet = await initUNetFromURL('/weights/rt_ldr.tza');
const noisy = ctx.getImageData(0, 0, width, height);

const abort = unet.tileExecute({
  color: noisy,
  progress(_image, tileData, tile) {
    ctx.putImageData(tileData!, tile.x, tile.y);
  },
  done(denoised) {
    ctx.putImageData(denoised, 0, 0);
  },
  error(reason) {
    console.error(reason);
  }
});
```

HDR input, auxiliary images, GPU buffers, tiling and performance options are in the [browser guide](docs/web.md).

## Choose weights

Pick the weights from what your renderer produces. The same files work in every runtime.

| File | Color | Auxiliary inputs | Browser options |
| --- | --- | --- | --- |
| `rt_ldr.tza` | LDR | none | `{}` |
| `rt_ldr_alb_nrm.tza` | LDR | albedo + normal | `{ aux: true }` |
| `rt_hdr.tza` | HDR | none | `{ hdr: true }` |
| `rt_hdr_alb_nrm.tza` | HDR | albedo + normal | `{ hdr: true, aux: true }` |
| `rt_hdr_alb_nrm_small.tza` | HDR | albedo + normal; faster, lower quality | `{ hdr: true, aux: true }` |
| `rt_hdr_calb_cnrm.tza` | HDR | noise-free albedo + normal | `{ hdr: true, aux: true }` |
| `rt_hdr_calb_cnrm_large.tza` | HDR | noise-free albedo + normal; slower, higher quality | `{ hdr: true, aux: true }` |
| `rtlightmap_hdr.tza` ([oidn-weights](https://github.com/RenderKit/oidn-weights)) | HDR lightmap | none | `{ hdr: true, hdrTransfer: 'log' }` |

The model size (small, default, large) is chosen by the file alone; no option changes. Loading throws if `aux` does not match the file's inputs. An HDR/LDR mismatch cannot be detected from the file, so set `hdr` to match the weights.

## Documentation

| Document | Covers |
| --- | --- |
| [docs/web.md](docs/web.md) | Browser API: loading, inputs, `tileExecute`, GPU pipelines, tiling, precision, profiling |
| [native/README.md](native/README.md) | Rust and C APIs, image contracts, builds and precompiled packages |
| [docs/development.md](docs/development.md) | Architecture, build/test/benchmark commands, updating OIDN models, CI |
| [AGENTS.md](AGENTS.md) | Repository map and working rules for coding agents and contributors |
| [llms.txt](llms.txt) | Doc index for LLM tools |
| [CHANGELOG.md](CHANGELOG.md) | Release notes and migration steps |

## Examples and users

- [Basic example](https://oidn-web-example.vercel.app/) ([code](https://github.com/pissang/oidn-web-example/blob/main/src/main.js)), built on the earlier `oidn-web` 0.3 package
- Local examples in [`examples/`](examples/): run `npm run dev` and open `/examples/basic.html`, `hdr.html`, `auxiliary.html` or `gpu.html`.
- [three-gpu-pathtracer `OIDNDenoiser`](https://github.com/gkjohnson/three-gpu-pathtracer/blob/main/src/webgpu/API.md#oidndenoiser)
- [Vector to 3D](https://www.figma.com/community/plugin/1264600219316901594/) Figma plugin

## Credits and license

Inspired by Max Liani's series [DNND: a deep neural network dive](https://maxliani.wordpress.com/2023/03/17/dnnd-1-a-deep-neural-network-dive/).

Code is MIT licensed ([LICENSE](LICENSE)). The OIDN weights in [`weights/`](weights/) are Apache 2.0 ([weights/LICENSE.txt](weights/LICENSE.txt)).
