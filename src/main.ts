import { parseTZA } from './tza';
import UNet from './UNet';
import type { UNetEngineSetting, UNetExecutionStats } from './UNet';
import { initWebGPUBackend } from './backend';
import type { DynamicTileSetting } from './tileScheduler';
import type { UNetModelSpec } from './modelSpec';
import type {
  NativeSubgroupMatrixOptions,
  NativeUNetGemmOptions,
  NativeUNetKernelSetting,
  NativeUNetPrecisionSetting
} from './nativeUNet';
import type { HDRTransfer } from './process';

export { parseTZA, UNet };
export { planTileGrid } from './tileScheduler';
export type {
  DynamicTileOptions,
  DynamicTileSetting,
  PlannedTile,
  TilePlan,
  TileRect
} from './tileScheduler';
export {
  detectUNetModelSpec,
  OIDN_UNET_LARGE_SPEC,
  OIDN_UNET_SMALL_SPEC,
  validateUNetModel
} from './modelSpec';
export type {
  ModelNodeSpec,
  UNetModelGraph,
  UNetModelSpec,
  ValidatedUNetModel
} from './modelSpec';
export { optimizeModelGraph, planModelExecution } from './graphOptimizer';
export type {
  ExecutableModelNode,
  GraphOptimizationOptions,
  ModelExecutionPlan,
  ModelValueShape,
  OptimizedModelGraph
} from './graphOptimizer';
export {
  NativeUNetExecutor,
  resolveNativeUNetPrecision,
  selectSubgroupMatrixConfig
} from './nativeUNet';
export type {
  NativeSubgroupMatrixConfig,
  NativeSubgroupMatrixOptions,
  NativeUNetGemmWorkgroup,
  NativeUNetGemmOptions,
  NativeUNetExecutionProfile,
  NativeUNetKernel,
  NativeUNetKernelSetting,
  NativeUNetLayerTiming,
  NativeUNetOptions,
  NativeUNetPrecision,
  NativeUNetPrecisionSetting
} from './nativeUNet';

export interface UNetOptions {
  aux?: boolean;
  hdr?: boolean;
  /** HDR transfer function expected by the trained model. Defaults to PU. */
  hdrTransfer?: HDRTransfer;
  /** Hard upper bound for an output tile edge. Defaults to 512. */
  maxTileSize?: number;
  /** Adaptive GPU-time-based tile sizing. Enabled by default. */
  dynamicTile?: DynamicTileSetting;
  /** `auto` uses native WGSL; `webnn` opts into the WebNN backend. */
  engine?: UNetEngineSetting;
  /** `auto` selects FP16 when shader-f16 was enabled on the GPUDevice. */
  precision?: NativeUNetPrecisionSetting;
  /** `auto` uses implicit GEMM for FP16/FP32 convolutions, except the direct output layer. */
  kernel?: NativeUNetKernelSetting;
  /**
   * Experimental implicit-GEMM tuning for native WGSL execution. These knobs
   * exist for benchmarking and may change or be removed in any release.
   * @experimental
   */
  gemm?: NativeUNetGemmOptions;
  /**
   * Experimental `kernel: 'subgroup-matrix'` tuning. The kernel needs a
   * GPUDevice created with `subgroups` and `chromium-experimental-subgroup-matrix`.
   * @experimental
   */
  subgroupMatrix?: NativeSubgroupMatrixOptions;
  /** Versioned topology descriptor for future/custom OIDN TZA models. */
  modelSpec?: UNetModelSpec;
}

export type { UNetEngineSetting, UNetExecutionStats } from './UNet';
export type { HDRTransfer } from './process';
export { WebNNUNetExecutor } from './webnnUNet';
export type { WebNNRuntimeSupport, WebNNUNetOptions } from './webnnUNet';
export type {
  OIDNResourceKind,
  OIDNResourceSnapshot,
  OIDNResourceStats
} from './resourceTracker';

export async function initUNetFromBuffer(
  tzaBuffer: ArrayBuffer,
  backendParams?: { device: GPUDevice },
  opts?: UNetOptions
) {
  const device = backendParams?.device ?? await initWebGPUBackend();
  const tensors = parseTZA(tzaBuffer);
  const unet = new UNet(tensors, device, opts);
  await unet.prepare();
  return unet;
}

export async function initUNetFromURL(
  modelPath: string,
  backendParams?: { device: GPUDevice },
  opts?: UNetOptions
) {
  return fetch(modelPath)
    .then((res) => res.arrayBuffer())
    .then((ab) => {
      return initUNetFromBuffer(ab, backendParams, opts);
    });
}
