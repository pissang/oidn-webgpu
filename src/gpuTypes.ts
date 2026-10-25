/**
 * The WebGPU subset used by the native WGSL runtime, as structural types and
 * spec-defined flag values. A browser GPUDevice implements these operations,
 * but its branded TypeScript types need the conversion in nativeUNet.ts;
 * they also let an embedder supply its own device (for example one that
 * records the work for replay in a native wgpu host) without lib.dom or
 * WebGPU globals.
 */

/** GPUBufferUsage flag values from the WebGPU specification. */
export const BufferUsage = {
  MAP_READ: 0x0001,
  MAP_WRITE: 0x0002,
  COPY_SRC: 0x0004,
  COPY_DST: 0x0008,
  INDEX: 0x0010,
  VERTEX: 0x0020,
  UNIFORM: 0x0040,
  STORAGE: 0x0080,
  INDIRECT: 0x0100,
  QUERY_RESOLVE: 0x0200
} as const;

/** GPUMapMode flag values from the WebGPU specification. */
export const MapMode = {
  READ: 0x0001,
  WRITE: 0x0002
} as const;

// The synchronous core types exclude promises so scriptc's library mode can
// compile the core; browser-only code uses the asynchronous extensions below.

export interface OIDNGPUBuffer {
  readonly size: number;
  readonly mapState: string;
  getMappedRange(offset?: number, size?: number): ArrayBuffer;
  unmap(): void;
  destroy(): void;
}

export interface OIDNGPUMappableBuffer extends OIDNGPUBuffer {
  mapAsync(mode: number, offset?: number, size?: number): Promise<void>;
}

export interface OIDNGPUQuerySet {
  destroy(): void;
}

export interface OIDNGPUShaderModule {
  readonly label: string;
}

export interface OIDNGPUBindGroupLayout {
  readonly label: string;
}

export interface OIDNGPUBindGroup {
  readonly label: string;
}

export interface OIDNGPUComputePipeline {
  getBindGroupLayout(index: number): OIDNGPUBindGroupLayout;
}

export interface OIDNGPUCommandBuffer {
  readonly label: string;
}

export interface OIDNGPUBindGroupEntry {
  binding: number;
  resource: { buffer: OIDNGPUBuffer; offset?: number; size?: number };
}

export interface OIDNGPUComputePassTimestampWrites {
  querySet: OIDNGPUQuerySet;
  beginningOfPassWriteIndex?: number;
  endOfPassWriteIndex?: number;
}

export interface OIDNGPUComputePassEncoder {
  setPipeline(pipeline: OIDNGPUComputePipeline): void;
  setBindGroup(index: number, bindGroup: OIDNGPUBindGroup): void;
  dispatchWorkgroups(x: number, y?: number, z?: number): void;
  end(): void;
}

export interface OIDNGPUCommandEncoder {
  beginComputePass(descriptor?: {
    label?: string;
    timestampWrites?: OIDNGPUComputePassTimestampWrites;
  }): OIDNGPUComputePassEncoder;
  copyBufferToBuffer(
    source: OIDNGPUBuffer,
    sourceOffset: number,
    destination: OIDNGPUBuffer,
    destinationOffset: number,
    size: number
  ): void;
  resolveQuerySet(
    querySet: OIDNGPUQuerySet,
    firstQuery: number,
    queryCount: number,
    destination: OIDNGPUBuffer,
    destinationOffset: number
  ): void;
  finish(): OIDNGPUCommandBuffer;
}

export interface OIDNGPUQueue {
  submit(commandBuffers: OIDNGPUCommandBuffer[]): void;
  writeBuffer(
    buffer: OIDNGPUBuffer,
    bufferOffset: number,
    data: Uint8Array
  ): void;
}

export interface OIDNGPUAsyncQueue extends OIDNGPUQueue {
  onSubmittedWorkDone(): Promise<void>;
}

export interface OIDNGPULimits {
  readonly maxComputeWorkgroupStorageSize: number;
  readonly maxComputeInvocationsPerWorkgroup: number;
  readonly maxComputeWorkgroupSizeX: number;
  readonly maxComputeWorkgroupSizeY: number;
  readonly maxComputeWorkgroupsPerDimension: number;
}

export interface OIDNGPUFeatures {
  has(name: string): boolean;
}

/** Subset of GPUAdapterInfo used to select subgroup-matrix configurations. */
export interface OIDNGPUAdapterInfo {
  readonly subgroupMinSize?: number;
  readonly subgroupMaxSize?: number;
  readonly subgroupMatrixConfigs?: readonly {
    componentType: string;
    resultComponentType: string;
    M: number;
    N: number;
    K: number;
  }[];
}

export interface OIDNGPUDevice {
  readonly features: OIDNGPUFeatures;
  readonly limits: OIDNGPULimits;
  readonly queue: OIDNGPUQueue;
  createBuffer(descriptor: {
    label?: string;
    size: number;
    usage: number;
    mappedAtCreation?: boolean;
  }): OIDNGPUBuffer;
  createShaderModule(descriptor: { label?: string; code: string }): OIDNGPUShaderModule;
  createComputePipeline(descriptor: {
    label?: string;
    layout: 'auto';
    compute: { module: OIDNGPUShaderModule; entryPoint: string };
  }): OIDNGPUComputePipeline;
  createBindGroup(descriptor: {
    label?: string;
    layout: OIDNGPUBindGroupLayout;
    entries: OIDNGPUBindGroupEntry[];
  }): OIDNGPUBindGroup;
  createCommandEncoder(descriptor?: { label?: string }): OIDNGPUCommandEncoder;
  createQuerySet(descriptor: { label?: string; type: 'timestamp'; count: number }): OIDNGPUQuerySet;
}

export interface OIDNGPUAsyncDevice extends OIDNGPUDevice {
  readonly queue: OIDNGPUAsyncQueue;
  createComputePipelineAsync(descriptor: {
    label?: string;
    layout: 'auto';
    compute: { module: OIDNGPUShaderModule; entryPoint: string };
  }): Promise<OIDNGPUComputePipeline>;
}
