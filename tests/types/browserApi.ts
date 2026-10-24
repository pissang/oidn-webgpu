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

// @ts-expect-error The executor needs a WebGPU device.
new NativeUNetExecutor({}, model);

export { executor, output, precision, matrix, unet };
