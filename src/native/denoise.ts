import {
  BufferUsage,
  type OIDNGPUBindGroupEntry,
  type OIDNGPUBuffer,
  type OIDNGPUDevice
} from '../gpuTypes.js';
import { hdrForwardWGSL, hdrInverseWGSL, type HDRTransfer } from '../hdrTransfer.js';
import { sanitizeWGSL } from '../sanitize.js';
import type { ValidatedUNetModel } from '../modelSpec.js';
import type { NativeUNetCore, NativeUNetPrecision } from '../nativeUNetCore.js';
import { OIDN_TILE_ALIGNMENT, planTileGrid } from '../tileScheduler.js';
import type { RecordedBuffer, RecordingDevice } from './recordingDevice.js';

export interface NativeDenoiseOptions {
  width: number;
  height: number;
  hdr: boolean;
  transfer: HDRTransfer;
  /** Maximum output tile edge; 0 denoises the whole image as one tile. */
  maxTileSize: number;
  inputScale: number;
}

const WORKGROUP = 8;

function alignUp(value: number, alignment: number) {
  return Math.ceil(value / alignment) * alignment;
}

/**
 * Input processing of one tile for buffer images: clamp-to-edge reads of the
 * tile's input rectangle, HDR transfer, and conversion directly into the
 * network's packed tensor. Auxiliary RGB channels cross vec4 boundaries;
 * unused lanes must be zero just as in createInputPackShader.
 */
function inputShader(sourceCount: number, hdr: boolean, transfer: HDRTransfer, precision: NativeUNetPrecision) {
  const names = ['color', 'albedo', 'normal'].slice(0, sourceCount);
  const bindings = names.map((name, index) => `
@group(0) @binding(${index}) var<storage, read> in_${name}: array<vec4f>;`).join('');
  // Like OIDN, callers pass signed [-1, 1] normals; the network reads n * 0.5 + 0.5.
  const copies = names.slice(1).map((name) => name === 'normal'
    ? 'let normal = sanitize(in_normal[inIdx].rgb, -1.0, 1.0) * 0.5 + 0.5;'
    : `let ${name} = sanitize(in_${name}[inIdx].rgb, 0.0, 1.0);`).join('\n  ');
  const channels = names.flatMap((name) => ['r', 'g', 'b'].map((channel) => `${name}.${channel}`));
  const blocks = Math.ceil(channels.length / 4);
  const type = precision === 'fp16' ? 'vec4<f16>' : 'vec4<f32>';
  const stores = Array.from({ length: blocks }, (_, block) =>
    `packed[outIdx * ${blocks} + ${block}] = ${type}(vec4f(${[0, 1, 2, 3].map((lane) => channels[block * 4 + lane] ?? '0.0').join(', ')}));`).join('\n  ');
  return /* wgsl */ `${precision === 'fp16' ? 'enable f16;\n' : ''}${hdrForwardWGSL(transfer)}${sanitizeWGSL}
struct Params {
  imageSize: vec2i,
  tileSize: vec2i,
  offset: vec2i,

}${bindings}
@group(0) @binding(${sourceCount}) var<storage, read_write> packed: array<${type}>;
@group(0) @binding(${sourceCount + 1}) var<uniform> params: Params;
${hdr ? `struct Frame { inputScale: f32, }
@group(0) @binding(${sourceCount + 2}) var<uniform> frame: Frame;` : ''}

@compute @workgroup_size(${WORKGROUP}, ${WORKGROUP}, 1)
fn main(@builtin(global_invocation_id) globalId: vec3u) {
  let position = vec2i(globalId.xy);
  if (position.x >= params.tileSize.x || position.y >= params.tileSize.y) {
    return;
  }
  let inputPosition = clamp(position + params.offset, vec2i(0), params.imageSize - vec2i(1));
  let inIdx = inputPosition.y * params.imageSize.x + inputPosition.x;
  let outIdx = position.y * params.tileSize.x + position.x;
  let col = in_color[inIdx];
  ${hdr
    ? `let scaled = sanitize(col.rgb * frame.inputScale, 0.0, FLT_MAX);
  let color = vec3f(HDRForward(scaled.r), HDRForward(scaled.g), HDRForward(scaled.b));`
    : 'let color = sanitize(col.rgb, 0.0, 1.0);'}
  ${copies}
  ${stores}
}
`;
}

/**
 * Output processing of one tile: the inverse HDR transfer of the network
 * output inside the tile's output rectangle, written with the input color's
 * alpha. Matches the browser runtime's GPU output pass.
 */
function outputShader(hdr: boolean, transfer: HDRTransfer) {
  return /* wgsl */ `${hdrInverseWGSL(transfer)}${sanitizeWGSL}
struct Params {
  imageWidth: i32,
  tileWidth: i32,
  outputOffset: vec2i,
  outputSize: vec2i,
  inputOffset: vec2i,

}
@group(0) @binding(0) var<storage, read> in_color: array<vec4f>;
@group(0) @binding(1) var<storage, read> in_raw: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> out_color: array<vec4f>;
@group(0) @binding(3) var<uniform> params: Params;
${hdr ? `struct Frame { inputScale: f32, }
@group(0) @binding(4) var<uniform> frame: Frame;` : ''}

@compute @workgroup_size(${WORKGROUP}, ${WORKGROUP}, 1)
fn main(@builtin(global_invocation_id) globalId: vec3u) {
  let x = i32(globalId.x);
  let y = i32(globalId.y);
  if (x >= params.outputSize.x || y >= params.outputSize.y) {
    return;
  }
  let inIdx = (y + params.inputOffset.y) * params.tileWidth + x + params.inputOffset.x;
  let outIdx = (y + params.outputOffset.y) * params.imageWidth + x + params.outputOffset.x;
  let col = in_color[inIdx];
  let raw = in_raw[outIdx];
  // The network output may contain negative values or NaNs.
  let denoised = sanitize(col.rgb, 0.0, ${hdr ? 'FLT_MAX' : '1.0'});
  ${hdr
    ? 'out_color[outIdx] = vec4f(vec3f(HDRInverse(denoised.r), HDRInverse(denoised.g), HDRInverse(denoised.b)) / frame.inputScale, raw.a);'
    : 'out_color[outIdx] = vec4f(denoised, raw.a);'}
}
`;
}

function uniformBuffer(device: OIDNGPUDevice, label: string, words: Int32Array) {
  const buffer = device.createBuffer({
    label,
    size: alignUp(words.byteLength, 16),
    usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
    mappedAtCreation: true
  });
  new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(words.buffer));
  buffer.unmap();
  return buffer;
}

function storageBuffer(device: OIDNGPUDevice, label: string, size: number) {
  return device.createBuffer({
    label,
    size,
    usage: BufferUsage.STORAGE | BufferUsage.COPY_SRC | BufferUsage.COPY_DST
  });
}

/**
 * Records the commands of one denoising of a `width` x `height` image: for
 * every planned tile the input pass, the network and the output pass. The
 * tiles follow `planTileGrid` with the model's receptive field as overlap,
 * like `UNet.tileExecute`.
 */
export function recordDenoiseExecution(
  device: RecordingDevice,
  executor: NativeUNetCore,
  model: ValidatedUNetModel,
  options: NativeDenoiseOptions
) {
  const { width, height } = options;
  if (width <= 0 || height <= 0) throw new Error('Image dimensions must be positive');
  const sourceCount = model.inputChannels / 3;
  const imageBytes = width * height * 16;
  const inputs: RecordedBuffer[] = [];
  for (const name of ['color', 'albedo', 'normal'].slice(0, sourceCount)) {
    inputs.push(storageBuffer(device, `oidn/external/${name}`, imageBytes) as RecordedBuffer);
  }
  const output = storageBuffer(device, 'oidn/external/output', imageBytes) as RecordedBuffer;

  const overlap = alignUp(model.spec.receptiveField / 2, OIDN_TILE_ALIGNMENT);
  const plan = planTileGrid(
    width,
    height,
    options.maxTileSize > 0 ? options.maxTileSize : alignUp(Math.max(width, height), OIDN_TILE_ALIGNMENT),
    overlap
  );

  const inputPipeline = device.createComputePipeline({
    label: 'oidn/native/input',
    layout: 'auto',
    compute: {
      module: device.createShaderModule({
        label: 'oidn/native/input',
        code: inputShader(sourceCount, options.hdr, options.transfer, executor.precision)
      }),
      entryPoint: 'main'
    }
  });
  const outputPipeline = device.createComputePipeline({
    label: 'oidn/native/output',
    layout: 'auto',
    compute: {
      module: device.createShaderModule({
        label: 'oidn/native/output',
        code: outputShader(options.hdr, options.transfer)
      }),
      entryPoint: 'main'
    }
  });
  const scaleWord = new Int32Array(new Float32Array([options.inputScale]).buffer)[0];

  const frame = uniformBuffer(device, 'oidn/native/frame', new Int32Array([scaleWord, 0, 0, 0])) as RecordedBuffer;

  // Network inputs are per tile shape; tiles run in order on one queue.
  const tileInputs = new Map<string, OIDNGPUBuffer>();
  for (const tile of plan.tiles) {
    const key = `${tile.input.width}x${tile.input.height}`;
    if (tileInputs.has(key)) continue;
    tileInputs.set(key, executor.packedInputBuffer(tile.input.width, tile.input.height));
  }
  device.operations = [];

  plan.tiles.forEach((tile, index) => {
    const packed = tileInputs.get(`${tile.input.width}x${tile.input.height}`)!;
    const inputParams = uniformBuffer(device, `oidn/native/input/${index}`, new Int32Array([
      width, height, tile.input.width, tile.input.height, tile.input.x, tile.input.y, 0, 0
    ]));
    const entries: OIDNGPUBindGroupEntry[] = [];
    for (let source = 0; source < sourceCount; source++) {
      entries.push({ binding: source, resource: { buffer: inputs[source] } });
    }
    entries.push({ binding: sourceCount, resource: { buffer: packed } });
    entries.push({ binding: sourceCount + 1, resource: { buffer: inputParams } });
    if (options.hdr) entries.push({ binding: sourceCount + 2, resource: { buffer: frame } });
    const inputBindings = device.createBindGroup({
      label: `oidn/native/input/${index}`,
      layout: inputPipeline.getBindGroupLayout(0),
      entries
    });
    const inputEncoder = device.createCommandEncoder({ label: `oidn/native/input/${index}` });
    const inputPass = inputEncoder.beginComputePass();
    inputPass.setPipeline(inputPipeline);
    inputPass.setBindGroup(0, inputBindings);
    inputPass.dispatchWorkgroups(
      Math.ceil(tile.input.width / WORKGROUP),
      Math.ceil(tile.input.height / WORKGROUP)
    );
    inputPass.end();
    device.queue.submit([inputEncoder.finish()]);

    const networkOutput = executor.executePacked(tile.input.width, tile.input.height);

    const outputParams = uniformBuffer(device, `oidn/native/output/${index}`, new Int32Array([
      width, tile.input.width,
      tile.output.x, tile.output.y,
      tile.output.width, tile.output.height,
      tile.output.x - tile.input.x, tile.output.y - tile.input.y,
      0, 0, 0, 0
    ]));
    const outputBindings = device.createBindGroup({
      label: `oidn/native/output/${index}`,
      layout: outputPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: networkOutput } },
        { binding: 1, resource: { buffer: inputs[0] } },
        { binding: 2, resource: { buffer: output } },
        { binding: 3, resource: { buffer: outputParams } },
        ...(options.hdr ? [{ binding: 4, resource: { buffer: frame } }] : [])
      ]
    });
    const outputEncoder = device.createCommandEncoder({ label: `oidn/native/output/${index}` });
    const outputPass = outputEncoder.beginComputePass();
    outputPass.setPipeline(outputPipeline);
    outputPass.setBindGroup(0, outputBindings);
    outputPass.dispatchWorkgroups(
      Math.ceil(tile.output.width / WORKGROUP),
      Math.ceil(tile.output.height / WORKGROUP)
    );
    outputPass.end();
    device.queue.submit([outputEncoder.finish()]);
  });
  return { inputs, output, tiles: plan.tiles.length, frame };
}
