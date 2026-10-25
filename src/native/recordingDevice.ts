import type {
  OIDNGPUAdapterInfo,
  OIDNGPUBindGroup,
  OIDNGPUBindGroupEntry,
  OIDNGPUBindGroupLayout,
  OIDNGPUBuffer,
  OIDNGPUCommandBuffer,
  OIDNGPUCommandEncoder,
  OIDNGPUComputePassEncoder,
  OIDNGPUComputePipeline,
  OIDNGPUDevice,
  OIDNGPUFeatures,
  OIDNGPULimits,
  OIDNGPUQuerySet,
  OIDNGPUQueue,
  OIDNGPUShaderModule
} from '../gpuTypes.js';

/**
 * An OIDNGPUDevice that executes nothing: it records every resource and queue
 * operation so that a native host can create the same resources once and
 * replay the recorded commands with its own GPU API.
 */

export class RecordedBuffer implements OIDNGPUBuffer {
  mapState = 'unmapped';
  data?: Uint8Array;
  destroyed = false;

  constructor(
    readonly id: number,
    readonly size: number,
    readonly usage: number,
    readonly label: string,
    mapped: boolean
  ) {
    if (mapped) {
      this.data = new Uint8Array(size);
      this.mapState = 'mapped';
    }
  }

  getMappedRange(): ArrayBuffer {
    if (!this.data) throw new Error(`Recorded buffer ${this.label} is not mapped`);
    return this.data.buffer as ArrayBuffer;
  }

  unmap() {
    this.mapState = 'unmapped';
  }

  destroy() {
    this.destroyed = true;
  }
}

export class RecordedShaderModule implements OIDNGPUShaderModule {
  constructor(readonly id: number, readonly label: string, readonly code: string) {}
}

export class RecordedBindGroupLayout implements OIDNGPUBindGroupLayout {
  constructor(readonly label: string, readonly pipeline: number, readonly group: number) {}
}

export class RecordedPipeline implements OIDNGPUComputePipeline {
  constructor(
    readonly id: number,
    readonly label: string,
    readonly shader: number,
    readonly entryPoint: string
  ) {}

  getBindGroupLayout(index: number): OIDNGPUBindGroupLayout {
    return new RecordedBindGroupLayout(this.label, this.id, index);
  }
}

export interface RecordedBinding {
  binding: number;
  buffer: number;
  offset: number;
  /** Zero binds the rest of the buffer. */
  size: number;
}

export class RecordedBindGroup implements OIDNGPUBindGroup {
  constructor(
    readonly id: number,
    readonly label: string,
    readonly pipeline: number,
    readonly group: number,
    readonly entries: RecordedBinding[]
  ) {}
}

export interface RecordedCommand {
  /** 0 = dispatch, 1 = buffer copy. */
  kind: number;
  pipeline: number;
  bindGroup: number;
  x: number;
  y: number;
  z: number;
  source: number;
  sourceOffset: number;
  destination: number;
  destinationOffset: number;
  size: number;
}

export class RecordedCommandBuffer implements OIDNGPUCommandBuffer {
  constructor(readonly label: string, readonly commands: RecordedCommand[]) {}
}

export interface RecordedQueueOperation {
  /** 0 = submit, 1 = buffer write. */
  kind: number;
  commands: RecordedCommand[];
  buffer: number;
  offset: number;
  data?: Uint8Array;
}

function command(kind: number): RecordedCommand {
  return {
    kind,
    pipeline: -1,
    bindGroup: -1,
    x: 0,
    y: 0,
    z: 0,
    source: -1,
    sourceOffset: 0,
    destination: -1,
    destinationOffset: 0,
    size: 0
  };
}

class RecordingPass implements OIDNGPUComputePassEncoder {
  private _pipeline = -1;
  private _bindGroup = -1;

  constructor(private _commands: RecordedCommand[]) {}

  setPipeline(pipeline: OIDNGPUComputePipeline) {
    this._pipeline = (pipeline as RecordedPipeline).id;
  }

  setBindGroup(index: number, bindGroup: OIDNGPUBindGroup) {
    if (index !== 0) throw new Error('Recorded passes support bind group 0 only');
    this._bindGroup = (bindGroup as RecordedBindGroup).id;
  }

  dispatchWorkgroups(x: number, y?: number, z?: number) {
    const recorded = command(0);
    recorded.pipeline = this._pipeline;
    recorded.bindGroup = this._bindGroup;
    recorded.x = x;
    recorded.y = y ?? 1;
    recorded.z = z ?? 1;
    this._commands.push(recorded);
  }

  end() {}
}

class RecordingEncoder implements OIDNGPUCommandEncoder {
  private _commands: RecordedCommand[] = [];

  constructor(private _label: string) {}

  beginComputePass(): OIDNGPUComputePassEncoder {
    return new RecordingPass(this._commands);
  }

  copyBufferToBuffer(
    source: OIDNGPUBuffer,
    sourceOffset: number,
    destination: OIDNGPUBuffer,
    destinationOffset: number,
    size: number
  ) {
    const recorded = command(1);
    recorded.source = (source as RecordedBuffer).id;
    recorded.sourceOffset = sourceOffset;
    recorded.destination = (destination as RecordedBuffer).id;
    recorded.destinationOffset = destinationOffset;
    recorded.size = size;
    this._commands.push(recorded);
  }

  resolveQuerySet() {
    throw new Error('Recorded devices do not support timestamp queries');
  }

  finish(): OIDNGPUCommandBuffer {
    return new RecordedCommandBuffer(this._label, this._commands);
  }
}

class RecordingQueue implements OIDNGPUQueue {
  constructor(private _device: RecordingDevice) {}

  submit(commandBuffers: OIDNGPUCommandBuffer[]) {
    const commands: RecordedCommand[] = [];
    for (const commandBuffer of commandBuffers) {
      for (const recorded of (commandBuffer as RecordedCommandBuffer).commands) {
        commands.push(recorded);
      }
    }
    this._device.operations.push({ kind: 0, commands, buffer: -1, offset: 0 });
  }

  writeBuffer(
    buffer: OIDNGPUBuffer,
    bufferOffset: number,
    data: Uint8Array
  ) {
    const bytes = new Uint8Array(data.byteLength);
    bytes.set(data);
    this._device.operations.push({
      kind: 1,
      commands: [],
      buffer: (buffer as RecordedBuffer).id,
      offset: bufferOffset,
      data: bytes
    });
  }

}

class RecordedFeatures implements OIDNGPUFeatures {
  constructor(private _names: string[]) {}

  has(name: string) {
    return this._names.includes(name);
  }
}

class RecordedQuerySet implements OIDNGPUQuerySet {
  destroy() {}
}

export interface RecordingDeviceDescriptor {
  features: string[];
  limits: OIDNGPULimits;
  adapterInfo: OIDNGPUAdapterInfo;
}

export class RecordingDevice implements OIDNGPUDevice {
  readonly features: OIDNGPUFeatures;
  readonly limits: OIDNGPULimits;
  readonly adapterInfo: OIDNGPUAdapterInfo;
  readonly queue: OIDNGPUQueue;
  readonly buffers: RecordedBuffer[] = [];
  readonly shaders: RecordedShaderModule[] = [];
  readonly pipelines: RecordedPipeline[] = [];
  readonly bindGroups: RecordedBindGroup[] = [];
  operations: RecordedQueueOperation[] = [];

  constructor(descriptor: RecordingDeviceDescriptor) {
    this.features = new RecordedFeatures(descriptor.features);
    this.limits = descriptor.limits;
    this.adapterInfo = descriptor.adapterInfo;
    this.queue = new RecordingQueue(this);
  }

  createBuffer(descriptor: {
    label?: string;
    size: number;
    usage: number;
    mappedAtCreation?: boolean;
  }): OIDNGPUBuffer {
    const buffer = new RecordedBuffer(
      this.buffers.length,
      descriptor.size,
      descriptor.usage,
      descriptor.label ?? '',
      descriptor.mappedAtCreation === true
    );
    this.buffers.push(buffer);
    return buffer;
  }

  createShaderModule(descriptor: { label?: string; code: string }): OIDNGPUShaderModule {
    const module = new RecordedShaderModule(
      this.shaders.length,
      descriptor.label ?? '',
      descriptor.code
    );
    this.shaders.push(module);
    return module;
  }

  createComputePipeline(descriptor: {
    label?: string;
    layout: 'auto';
    compute: { module: OIDNGPUShaderModule; entryPoint: string };
  }): OIDNGPUComputePipeline {
    const pipeline = new RecordedPipeline(
      this.pipelines.length,
      descriptor.label ?? '',
      (descriptor.compute.module as RecordedShaderModule).id,
      descriptor.compute.entryPoint
    );
    this.pipelines.push(pipeline);
    return pipeline;
  }

  createBindGroup(descriptor: {
    label?: string;
    layout: OIDNGPUBindGroupLayout;
    entries: OIDNGPUBindGroupEntry[];
  }): OIDNGPUBindGroup {
    const layout = descriptor.layout as RecordedBindGroupLayout;
    const entries: RecordedBinding[] = descriptor.entries.map((entry) => ({
      binding: entry.binding,
      buffer: (entry.resource.buffer as RecordedBuffer).id,
      offset: entry.resource.offset ?? 0,
      size: entry.resource.size ?? 0
    }));
    const bindGroup = new RecordedBindGroup(
      this.bindGroups.length,
      descriptor.label ?? '',
      layout.pipeline,
      layout.group,
      entries
    );
    this.bindGroups.push(bindGroup);
    return bindGroup;
  }

  createCommandEncoder(descriptor?: { label?: string }): OIDNGPUCommandEncoder {
    return new RecordingEncoder(descriptor?.label ?? '');
  }

  createQuerySet(): OIDNGPUQuerySet {
    return new RecordedQuerySet();
  }
}
