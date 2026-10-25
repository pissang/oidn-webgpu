/// <reference types="@webgpu/types" />
// Compiled by types.test.mjs against the built declarations in lib/.
import {
  initUNetFromURL,
  NativeUNetExecutor,
  resolveNativeUNetPrecision,
  selectSubgroupMatrixConfig,
  type NativeSubgroupMatrixConfig,
  type NativeUNetPrecision,
  type ValidatedUNetModel
} from '../../lib/main.js';

declare const device: GPUDevice;
declare const model: ValidatedUNetModel;
declare const color: GPUBuffer;

const executor = new NativeUNetExecutor(device, model, { kernel: 'auto' });
const output: GPUBuffer = executor.execute([color], 64, 64);
const precision: NativeUNetPrecision = resolveNativeUNetPrecision(device, 'auto');
const matrix: NativeSubgroupMatrixConfig | undefined = selectSubgroupMatrixConfig(device);
const unet = initUNetFromURL('weights.tza', { device });

async function denoiseWithFloatGuides() {
  const denoiser = await unet;
  const floatImage = { data: new Float32Array(64 * 64 * 4), width: 64, height: 64 };
  const byteImage = new ImageData(64, 64);
  denoiser.tileExecute({ color: floatImage, albedo: floatImage, normal: byteImage, done() {} });
  denoiser.tileExecute({ color: byteImage, albedo: byteImage, normal: floatImage, done() {} });
}

// @ts-expect-error The executor needs a WebGPU device.
new NativeUNetExecutor({}, model);

export { executor, output, precision, matrix, unet };
