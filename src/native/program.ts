import { BufferUsage } from '../gpuTypes.js';
import { detectUNetModelSpec, validateUNetModel } from '../modelSpec.js';
import {
  NativeUNetCore,
  type NativeUNetKernelSetting,
  type NativeUNetPrecisionSetting
} from '../nativeUNetCore.js';
import { parseTZA } from '../tza.js';
import { RecordingDevice, type RecordedBuffer } from './recordingDevice.js';

/**
 * Records the native U-Net for one input size as a replayable program.
 *
 * Config (little-endian u32 words):
 *   0 version (1), 1 width, 2 height,
 *   3 precision (0 auto, 1 fp32, 2 fp16),
 *   4 kernel (0 auto, 1 implicit-gemm, 2 direct, 3 subgroup-matrix),
 *   5 features (1 shader-f16, 2 subgroups, 4 chromium-experimental-subgroup-matrix),
 *   6 maxComputeWorkgroupStorageSize, 7 maxComputeInvocationsPerWorkgroup,
 *   8 maxComputeWorkgroupSizeX, 9 maxComputeWorkgroupSizeY,
 *   10 maxComputeWorkgroupsPerDimension, 11 subgroupMinSize, 12 subgroupMaxSize,
 *   13 matrix configuration count, then per configuration:
 *      component type (0 f32, 1 f16), result type, M, N, K.
 *
 * Program (little-endian u32 words, strings and byte blobs padded to 4 bytes):
 *   magic 0x4e44494f ("OIDN"), version 1,
 *   input count, input buffer ids..., output buffer id,
 *   output width, output height, output channels,
 *   buffers: count, then size, usage, data length (0 = zero-filled), data, label
 *   shaders: count, then code
 *   pipelines: count, then shader id, entry point
 *   bind groups: count, then pipeline id, group, entry count,
 *     entries (binding, buffer id, offset, size; size 0 = rest of buffer)
 *   operations: count, then kind
 *     0 submit: command count, commands as kind
 *       0 dispatch: pipeline id, bind group id, x, y, z
 *       1 copy: source, source offset, destination, destination offset, size
 *     1 write: buffer id, offset, data length, data
 *
 * Input buffers are RGBA f32 images in the transfer space of the network
 * (as produced by the browser runtime's GPU preprocessing); the output buffer
 * holds RGBA f32 pixels in the same space.
 */
export function recordNetwork(tza: Uint8Array, config: Uint8Array): Uint8Array {
  const words = new Uint32Array(config.buffer, config.byteOffset, config.byteLength / 4);
  if (words[0] !== 1) throw new Error(`Unsupported native config version ${words[0]}`);
  const width = words[1];
  const height = words[2];
  const precisions: NativeUNetPrecisionSetting[] = ['auto', 'fp32', 'fp16'];
  const kernels: NativeUNetKernelSetting[] = ['auto', 'implicit-gemm', 'direct', 'subgroup-matrix'];
  const features: string[] = [];
  if (words[5] & 1) features.push('shader-f16');
  if (words[5] & 2) features.push('subgroups');
  if (words[5] & 4) features.push('chromium-experimental-subgroup-matrix');
  const configs: { componentType: string; resultComponentType: string; M: number; N: number; K: number }[] = [];
  for (let index = 0; index < words[13]; index++) {
    const base = 14 + index * 5;
    configs.push({
      componentType: words[base] === 1 ? 'f16' : 'f32',
      resultComponentType: words[base + 1] === 1 ? 'f16' : 'f32',
      M: words[base + 2],
      N: words[base + 3],
      K: words[base + 4]
    });
  }
  const device = new RecordingDevice({
    features,
    limits: {
      maxComputeWorkgroupStorageSize: words[6],
      maxComputeInvocationsPerWorkgroup: words[7],
      maxComputeWorkgroupSizeX: words[8],
      maxComputeWorkgroupSizeY: words[9],
      maxComputeWorkgroupsPerDimension: words[10]
    },
    adapterInfo: {
      subgroupMinSize: words[11],
      subgroupMaxSize: words[12],
      subgroupMatrixConfigs: configs
    }
  });

  const tensors = parseTZA(
    tza.buffer.slice(tza.byteOffset, tza.byteOffset + tza.byteLength) as ArrayBuffer
  );
  const model = validateUNetModel(tensors, detectUNetModelSpec(tensors));
  const executor = new NativeUNetCore(device, model, {
    precision: precisions[words[3]],
    kernel: kernels[words[4]]
  });
  const inputs: RecordedBuffer[] = [];
  for (let index = 0; index < model.inputChannels / 3; index++) {
    inputs.push(device.createBuffer({
      label: `oidn/external/input${index}`,
      size: width * height * 16,
      usage: BufferUsage.STORAGE | BufferUsage.COPY_DST
    }) as RecordedBuffer);
  }
  // The first execution creates the per-shape resources; the second records
  // exactly what every later execution submits.
  executor.execute(inputs, width, height);
  device.operations = [];
  const output = executor.execute(inputs, width, height) as RecordedBuffer;
  return serializeProgram(device, inputs, output, width, height, model.outputChannels);
}

class ProgramWriter {
  private _bytes = new Uint8Array(1 << 16);
  private _view = new DataView(this._bytes.buffer);
  length = 0;

  private _reserve(count: number) {
    if (this.length + count <= this._bytes.length) return;
    let capacity = this._bytes.length * 2;
    while (capacity < this.length + count) capacity *= 2;
    const bytes = new Uint8Array(capacity);
    bytes.set(this._bytes.subarray(0, this.length));
    this._bytes = bytes;
    this._view = new DataView(bytes.buffer);
  }

  u32(value: number) {
    this._reserve(4);
    this._view.setUint32(this.length, value, true);
    this.length += 4;
  }

  bytes(data: Uint8Array) {
    this.u32(data.length);
    const padded = (data.length + 3) & ~3;
    this._reserve(padded);
    this._bytes.set(data, this.length);
    this._bytes.fill(0, this.length + data.length, this.length + padded);
    this.length += padded;
  }

  string(value: string) {
    this.bytes(new TextEncoder().encode(value));
  }

  finish() {
    return this._bytes.slice(0, this.length);
  }
}

function serializeProgram(
  device: RecordingDevice,
  inputs: readonly RecordedBuffer[],
  output: RecordedBuffer,
  width: number,
  height: number,
  outputChannels: number
) {
  const writer = new ProgramWriter();
  writer.u32(0x4e44494f);
  writer.u32(1);
  writer.u32(inputs.length);
  for (const input of inputs) writer.u32(input.id);
  writer.u32(output.id);
  writer.u32(width);
  writer.u32(height);
  writer.u32(outputChannels);

  writer.u32(device.buffers.length);
  for (const buffer of device.buffers) {
    writer.u32(buffer.size);
    writer.u32(buffer.usage);
    writer.bytes(buffer.data ?? new Uint8Array(0));
    writer.string(buffer.label);
  }
  writer.u32(device.shaders.length);
  for (const shader of device.shaders) writer.string(shader.code);
  writer.u32(device.pipelines.length);
  for (const pipeline of device.pipelines) {
    writer.u32(pipeline.shader);
    writer.string(pipeline.entryPoint);
  }
  writer.u32(device.bindGroups.length);
  for (const bindGroup of device.bindGroups) {
    writer.u32(bindGroup.pipeline);
    writer.u32(bindGroup.group);
    writer.u32(bindGroup.entries.length);
    for (const entry of bindGroup.entries) {
      writer.u32(entry.binding);
      writer.u32(entry.buffer);
      writer.u32(entry.offset);
      writer.u32(entry.size);
    }
  }
  writer.u32(device.operations.length);
  for (const operation of device.operations) {
    writer.u32(operation.kind);
    if (operation.kind === 0) {
      writer.u32(operation.commands.length);
      for (const recorded of operation.commands) {
        writer.u32(recorded.kind);
        if (recorded.kind === 0) {
          writer.u32(recorded.pipeline);
          writer.u32(recorded.bindGroup);
          writer.u32(recorded.x);
          writer.u32(recorded.y);
          writer.u32(recorded.z);
        } else {
          writer.u32(recorded.source);
          writer.u32(recorded.sourceOffset);
          writer.u32(recorded.destination);
          writer.u32(recorded.destinationOffset);
          writer.u32(recorded.size);
        }
      }
    } else {
      writer.u32(operation.buffer);
      writer.u32(operation.offset);
      writer.bytes(operation.data ?? new Uint8Array(0));
    }
  }
  return writer.finish();
}

/** Error packet: magic 0x4f525245 ("ERRO") followed by the message. */
export function recordNetworkPacket(tza: Uint8Array, config: Uint8Array): Uint8Array {
  try {
    return recordNetwork(tza, config);
  } catch (error) {
    const writer = new ProgramWriter();
    writer.u32(0x4f525245);
    writer.string(error instanceof Error ? error.message : String(error));
    return writer.finish();
  }
}
