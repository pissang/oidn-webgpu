import { optimizeModelGraph } from './graphOptimizer.js';
import {
  BufferUsage,
  MapMode,
  type OIDNGPUAsyncDevice,
  type OIDNGPUBuffer,
  type OIDNGPUComputePipeline,
  type OIDNGPUDevice,
  type OIDNGPUMappableBuffer,
  type OIDNGPUQuerySet
} from './gpuTypes.js';
import {
  createInputPackShader,
  NativeUNetCore,
  type CachedExecution,
  type NativeUNetExecutionProfile,
  type NativePipelineSpec
} from './nativeUNetCore.js';

export * from './nativeUNetCore.js';

const pendingPipelinesByDevice = new WeakMap<
  OIDNGPUDevice,
  Map<string, Promise<OIDNGPUComputePipeline>>
>();

/**
 * Native, model-driven OIDN U-Net executor for browser WebGPU devices. The
 * device must also provide the asynchronous WebGPU operations.
 */
export class NativeUNetExecutor extends NativeUNetCore {
  private _lastExecutionProfile?: Promise<NativeUNetExecutionProfile>;

  private get _asyncDevice() {
    return this._device as OIDNGPUAsyncDevice;
  }

  private get _pipelinePromises() {
    let pending = pendingPipelinesByDevice.get(this._device);
    if (!pending) {
      pending = new Map();
      pendingPipelinesByDevice.set(this._device, pending);
    }
    return pending;
  }

  getLastExecutionProfile() {
    return this._lastExecutionProfile;
  }

  private _pipelineAsync(key: string, code: string) {
    const ready = this._pipelineCache.get(key);
    if (ready) return Promise.resolve(ready);
    const pending = this._pipelinePromises.get(key);
    if (pending) return pending;
    const promise = this._asyncDevice.createComputePipelineAsync({
      label: `oidn/${key}`,
      layout: 'auto',
      compute: {
        module: this._device.createShaderModule({
          label: `oidn/${key}`,
          code
        }),
        entryPoint: 'main'
      }
    }).then((pipeline: OIDNGPUComputePipeline) => {
      this._pipelineCache.set(key, pipeline);
      this._pipelinePromises.delete(key);
      return pipeline;
    }, (error: unknown) => {
      this._pipelinePromises.delete(key);
      throw error;
    });
    this._pipelinePromises.set(key, promise);
    return promise;
  }

  protected override _retireExecution(execution: CachedExecution) {
    // Commands using an evicted plan may still be submitted. Defer actual
    // destruction until all work currently on the shared queue completes.
    this._retiredExecutions.add(execution);
    void this._asyncDevice.queue.onSubmittedWorkDone()
      .catch(() => undefined)
      .then(() => {
        this._retiredExecutions.delete(execution);
        this._destroyExecution(execution);
      });
  }

  /** Compiles all shape-independent kernels before the model reports ready. */
  async prepare() {
    if (this._disposed) throw new Error('Native OIDN executor is disposed');
    const graph = optimizeModelGraph(this._model, { fuseConvPool: false });
    const sourceCount = this._model.inputChannels / 3;
    const specs: NativePipelineSpec[] = [
      {
        key: `pack/${this.precision}/${sourceCount}`,
        code: createInputPackShader(this.precision, sourceCount)
      },
      ...graph.nodes.map((node) =>
        this._nodePipelineSpec(node, node.id === graph.spec.output)
      )
    ];
    await Promise.all(
      specs.map(({ key, code }) => this._pipelineAsync(key, code))
    );
  }

  protected override _readProfile(
    querySet: OIDNGPUQuerySet,
    resolveBuffer: OIDNGPUBuffer,
    readbackBuffer: OIDNGPUBuffer,
    labels: string[]
  ) {
    this._lastExecutionProfile = (async () => {
      try {
        await (readbackBuffer as OIDNGPUMappableBuffer).mapAsync(MapMode.READ);
        // Little-endian u64 nanoseconds as u32 pairs; differences stay exact.
        const words = new Uint32Array(readbackBuffer.getMappedRange());
        const timestamp = (query: number) =>
          words[query * 2 + 1] * 4294967296 + words[query * 2];
        const layers = labels.map((id, index) => ({
          id,
          durationMs:
            (timestamp(index * 2 + 1) - timestamp(index * 2)) / 1_000_000
        }));
        return {
          totalMs: layers.reduce(
            (sum, layer) => sum + layer.durationMs,
            0
          ),
          layers
        };
      } finally {
        if (readbackBuffer.mapState === 'mapped') {
          readbackBuffer.unmap();
        }
        this._releaseQuerySet(querySet);
        this._releaseBuffer(resolveBuffer);
        this._releaseBuffer(readbackBuffer);
        this._profileOperations--;
      }
    })();
  }

  /** Executes interleaved CPU image data through the native GPU runtime. */
  async executeCPU(
    interleavedInput: Float32Array,
    width: number,
    height: number
  ): Promise<Float32Array> {
    const expectedLength = width * height * this._model.inputChannels;
    if (interleavedInput.length !== expectedLength) {
      throw new Error(
        `Native OIDN CPU input has ${interleavedInput.length} values, expected ${expectedLength}`
      );
    }
    const execution = this._execution(width, height);
    const sourceCount = this._model.inputChannels / 3;
    const pixelCount = width * height;
    if (!execution.cpuInputBuffers) {
      execution.cpuInputBuffers = Array.from({ length: sourceCount }, (_, index) => {
        const buffer = this._device.createBuffer({
          label: `oidn/cpu-input/${width}x${height}/${index}`,
          size: pixelCount * 16,
          usage: BufferUsage.STORAGE | BufferUsage.COPY_DST
        });
        this._resources.track('gpu-buffer', buffer);
        execution.ownedBuffers.push(buffer);
        return buffer;
      });
      execution.cpuReadbackBuffer = this._device.createBuffer({
        label: `oidn/cpu-readback/${width}x${height}`,
        size: pixelCount * 16,
        usage: BufferUsage.COPY_DST | BufferUsage.MAP_READ
      });
      this._resources.track('gpu-buffer', execution.cpuReadbackBuffer);
      execution.ownedBuffers.push(execution.cpuReadbackBuffer);
    }

    for (let source = 0; source < sourceCount; source++) {
      const upload = new Float32Array(pixelCount * 4);
      for (let pixel = 0; pixel < pixelCount; pixel++) {
        const inputOffset =
          pixel * this._model.inputChannels + source * 3;
        const outputOffset = pixel * 4;
        upload[outputOffset] = interleavedInput[inputOffset];
        upload[outputOffset + 1] = interleavedInput[inputOffset + 1];
        upload[outputOffset + 2] = interleavedInput[inputOffset + 2];
      }
      this._device.queue.writeBuffer(
        execution.cpuInputBuffers[source],
        0,
        new Uint8Array(upload.buffer, upload.byteOffset, upload.byteLength)
      );
    }

    const output = this.execute(execution.cpuInputBuffers, width, height);
    const encoder = this._device.createCommandEncoder({
      label: `oidn/cpu-readback/${width}x${height}`
    });
    encoder.copyBufferToBuffer(
      output,
      0,
      execution.cpuReadbackBuffer!,
      0,
      pixelCount * 16
    );
    this._device.queue.submit([encoder.finish()]);
    await (execution.cpuReadbackBuffer as OIDNGPUMappableBuffer).mapAsync(MapMode.READ);
    const rgba = new Float32Array(
      execution.cpuReadbackBuffer!.getMappedRange()
    );
    const rgb = new Float32Array(pixelCount * 3);
    for (let pixel = 0; pixel < pixelCount; pixel++) {
      rgb[pixel * 3] = rgba[pixel * 4];
      rgb[pixel * 3 + 1] = rgba[pixel * 4 + 1];
      rgb[pixel * 3 + 2] = rgba[pixel * 4 + 2];
    }
    execution.cpuReadbackBuffer!.unmap();
    return rgb;
  }
}
