import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import { Float16Array } from '@petamoriken/float16';
import { NativeUNetExecutor } from '../lib/nativeUNet.js';
import { validateUNetModel } from '../lib/modelSpec.js';
import { HostTensor, TensorDesc, parseTZA } from '../lib/tza.js';

const GPU_USAGE = {
  STORAGE: 1,
  COPY_SRC: 2,
  COPY_DST: 4,
  UNIFORM: 8,
  QUERY_RESOLVE: 16,
  MAP_READ: 32
};

const TEST_SPEC = {
  schemaVersion: 1,
  id: 'native-gemm-test-unet',
  family: 'native-gemm-test',
  input: 'input',
  output: 'output',
  receptiveField: 5,
  nodes: [
    {
      op: 'conv2d',
      id: 'hidden',
      input: 'input',
      weight: 'hidden.weight',
      bias: 'hidden.bias',
      activation: 'relu',
      padding: 'same'
    },
    {
      op: 'conv2d',
      id: 'output',
      input: 'hidden',
      weight: 'output.weight',
      bias: 'output.bias',
      activation: 'identity',
      padding: 'same'
    }
  ]
};

class FakeBuffer {
  constructor(size, mappedAtCreation) {
    this.data = new ArrayBuffer(size);
    this.destroyCalls = 0;
    this.mapState = mappedAtCreation ? 'mapped' : 'unmapped';
  }

  getMappedRange() {
    return this.data;
  }

  unmap() {
    this.mapState = 'unmapped';
  }

  destroy() {
    this.destroyCalls++;
  }
}

function fakeDevice({
  maxComputeWorkgroupStorageSize = 32768
} = {}) {
  const device = {
    features: new Set(['shader-f16']),
    limits: {
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupSizeX: 256,
      maxComputeWorkgroupSizeY: 256,
      maxComputeWorkgroupStorageSize
    },
    buffers: [],
    queue: {
      submit() {},
      onSubmittedWorkDone: () => Promise.resolve()
    },
    createBuffer(descriptor) {
      const buffer = new FakeBuffer(descriptor.size, descriptor.mappedAtCreation);
      this.buffers.push(buffer);
      return buffer;
    },
    createShaderModule() {
      return {};
    },
    createComputePipeline() {
      return { getBindGroupLayout: () => ({}) };
    },
    async createComputePipelineAsync() {
      return { getBindGroupLayout: () => ({}) };
    },
    createBindGroup() {
      return {};
    },
    createCommandEncoder() {
      return {
        beginComputePass() {
          return {
            setPipeline() {},
            setBindGroup() {},
            dispatchWorkgroups() {},
            end() {}
          };
        },
        finish() {
          return {};
        }
      };
    }
  };
  return device;
}

function tensor(dims, layout, dataType, values) {
  const desc = new TensorDesc();
  desc.dims = [...dims];
  desc.paddedDims = [...dims];
  desc.layout = layout;
  desc.dataType = dataType;
  const typed = dataType === 'Float16'
    ? new Float16Array(values)
    : new Float32Array(values);
  return new HostTensor(
    desc,
    new Uint8Array(typed.buffer, typed.byteOffset, typed.byteLength)
  );
}

function testModel(dataType) {
  const hiddenValues = Array.from({ length: 5 * 3 * 3 * 3 }, (_, index) =>
    1 + index / 16
  );
  const outputValues = Array.from({ length: 3 * 5 * 3 * 3 }, (_, index) =>
    11 + index / 16
  );
  const tensors = new Map([
    ['hidden.weight', tensor([5, 3, 3, 3], 'oihw', dataType, hiddenValues)],
    ['hidden.bias', tensor([5], 'x', dataType, [1, 2, 3, 4, 5])],
    ['output.weight', tensor([3, 5, 3, 3], 'oihw', dataType, outputValues)],
    ['output.bias', tensor([3], 'x', dataType, [7, 8, 9])]
  ]);
  return {
    model: validateUNetModel(tensors, TEST_SPEC),
    hiddenValues
  };
}

function expectedPacked(hiddenValues, dataType, layout) {
  const inputChannels = 3;
  const outputChannels = 5;
  const inputBlocks = Math.ceil(inputChannels / 4);
  const outputBlocks = Math.ceil(outputChannels / 4);
  const values = new Array(outputBlocks * 3 * 3 * inputBlocks * 16).fill(0);
  for (let outputBlock = 0; outputBlock < outputBlocks; outputBlock++) {
    for (let y = 0; y < 3; y++) {
      for (let x = 0; x < 3; x++) {
        for (let inputBlock = 0; inputBlock < inputBlocks; inputBlock++) {
          const packedBlock = layout === 'k-major'
            ? ((((y * 3 + x) * inputBlocks + inputBlock) * outputBlocks + outputBlock) * 16)
            : ((((outputBlock * 3 + y) * 3 + x) * inputBlocks + inputBlock) * 16);
          for (let outputLane = 0; outputLane < 4; outputLane++) {
            const outputChannel = outputBlock * 4 + outputLane;
            for (let inputLane = 0; inputLane < 4; inputLane++) {
              const inputChannel = inputBlock * 4 + inputLane;
              const index = packedBlock + inputLane * 4 + outputLane;
              if (outputChannel < outputChannels && inputChannel < inputChannels) {
                values[index] = hiddenValues[
                  ((outputChannel * inputChannels + inputChannel) * 3 + y) * 3 + x
                ];
              }
            }
          }
        }
      }
    }
  }
  if (dataType === 'Float16') {
    const converted = new Float16Array(values);
    return Array.from(new Uint16Array(converted.buffer, converted.byteOffset, converted.length));
  }
  return values;
}

function unpackWeightBuffer(buffer, dataType) {
  return dataType === 'Float16'
    ? Array.from(new Uint16Array(buffer.data))
    : Array.from(new Float32Array(buffer.data));
}

function withGpuUsage(run) {
  const previous = Object.getOwnPropertyDescriptor(globalThis, 'GPUBufferUsage');
  Object.defineProperty(globalThis, 'GPUBufferUsage', {
    configurable: true,
    value: GPU_USAGE
  });
  return Promise.resolve()
    .then(run)
    .finally(() => {
      if (previous) Object.defineProperty(globalThis, 'GPUBufferUsage', previous);
      else delete globalThis.GPUBufferUsage;
    });
}

function executorOptions(precision, weightLayout, overrides = {}) {
  return {
    precision,
    kernel: 'implicit-gemm',
    gemm: {
      addressMode: 'analytic',
      weightLayout,
      rowsPerThread: 4,
      workgroupSize: [8, 8],
      ...overrides
    }
  };
}

for (const [dataType, precision] of [
  ['Float32', 'fp32'],
  ['Float16', 'fp16']
]) {
  test(`packs ${dataType} output-major and k-major weights with padded lanes`, async () => {
    await withGpuUsage(() => {
      const outputMajorDevice = fakeDevice();
      const outputMajor = new NativeUNetExecutor(
        outputMajorDevice,
        testModel(dataType).model,
        executorOptions(precision, 'output-major')
      );
      const kMajorDevice = fakeDevice();
      const kMajor = new NativeUNetExecutor(
        kMajorDevice,
        testModel(dataType).model,
        executorOptions(precision, 'k-major')
      );
      const hiddenValues = testModel(dataType).hiddenValues;
      const expectedOutput = expectedPacked(hiddenValues, dataType, 'output-major');
      const expectedK = expectedPacked(hiddenValues, dataType, 'k-major');

      // Each executor packs hidden weights first, then hidden bias.
      assert.deepEqual(
        unpackWeightBuffer(outputMajorDevice.buffers[0], dataType),
        expectedOutput
      );
      assert.deepEqual(
        unpackWeightBuffer(kMajorDevice.buffers[0], dataType),
        expectedK
      );
      assert.notDeepEqual(expectedOutput, expectedK);
      assert.equal(outputMajorDevice.buffers[0].data.byteLength, kMajorDevice.buffers[0].data.byteLength);

      outputMajor.dispose();
      kMajor.dispose();
    });
  });
}

test('freezes normalized GEMM configuration without mutating caller input', async () => {
  await withGpuUsage(() => {
    const device = fakeDevice();
    const input = {
      addressMode: 'incremental',
      weightLayout: 'k-major',
      rowsPerThread: 2,
      workgroupSize: [4, 8]
    };
    const executor = new NativeUNetExecutor(
      device,
      testModel('Float32').model,
      { precision: 'fp32', kernel: 'implicit-gemm', gemm: input }
    );
    assert.deepEqual(executor.gemm, {
      decoderLoad: 'per-load',
      decoderSubPixel: false,
      unrollRows: false,
      poolLayout: 'channels',
      sharedLayout: 'padded',
      accumulationOrder: 'k-major',
      finalLayer: 'shared-auto',
      loadMode: 'native',
      addressMode: 'incremental',
      tilePolicy: 'fixed',
      weightLayout: 'k-major',
      rowsPerThread: 2,
      workgroupSize: [4, 8]
    });
    assert.ok(Object.isFrozen(executor.gemm));
    assert.ok(Object.isFrozen(executor.gemm.workgroupSize));
    assert.notStrictEqual(executor.gemm.workgroupSize, input.workgroupSize);
    assert.deepEqual(input, {
      addressMode: 'incremental',
      weightLayout: 'k-major',
      rowsPerThread: 2,
      workgroupSize: [4, 8]
    });
    assert.throws(() => {
      executor.gemm.addressMode = 'analytic';
    }, TypeError);
    executor.dispose();
  });
});

test('rejects unsupported GEMM address, layout, register, workgroup, and storage settings', async () => {
  await withGpuUsage(() => {
    const model = testModel('Float32').model;
    for (const gemm of [
      { decoderLoad: 'bad' },
      { poolLayout: 'bad' },
      { sharedLayout: 'bad' },
      { accumulationOrder: 'bad' },
      { addressMode: 'bad' },
      { loadMode: 'bad' },
      { finalLayer: 'bad' },
      { tilePolicy: 'bad' },
      { weightLayout: 'bad' },
      { rowsPerThread: 3 },
      { workgroupSize: [2, 8] },
      { workgroupSize: [8, 16] }
    ]) {
      assert.throws(
        () => new NativeUNetExecutor(
          fakeDevice(),
          model,
          { precision: 'fp32', kernel: 'implicit-gemm', gemm }
        ),
        /Unsupported GEMM/
      );
    }
    assert.throws(
      () => new NativeUNetExecutor(
        fakeDevice({ maxComputeWorkgroupStorageSize: 1 }),
        model,
        executorOptions('fp32', 'output-major')
      ),
      /Unsupported GEMM tile configuration/
    );
  });
});

function matrixDevice(storageBytes, size = 16) {
  const device = fakeDevice({ maxComputeWorkgroupStorageSize: storageBytes });
  device.features.add('subgroups');
  device.features.add('chromium-experimental-subgroup-matrix');
  device.adapterInfo = { subgroupMinSize: 32, subgroupMaxSize: 32,
    subgroupMatrixConfigs: [{ componentType: size === 16 ? 'f16' : 'f32', resultComponentType: 'f32', M: size, N: size, K: size }] };
  return device;
}

test('retains 16x16 matrices with one fragment when two exceed the shared-memory limit', () => {
  const executor = new NativeUNetExecutor(matrixDevice(32768), testModel('Float16').model,
    { precision: 'fp16', kernel: 'subgroup-matrix' });
  assert.equal(executor.subgroupMatrix?.size, 16);
  assert.equal(executor.subgroupMatrixOutputFragments, 1);
  executor.dispose();
});

test('padded shared-memory layout accepts its exact storage limit and rejects one byte below it', async () => {
  await withGpuUsage(() => {
    const model = testModel('Float16').model;
    const exact = new NativeUNetExecutor(
      fakeDevice({ maxComputeWorkgroupStorageSize: 6720 }),
      model,
      executorOptions('fp16', 'k-major', {
        sharedLayout: 'padded',
        rowsPerThread: 8,
        workgroupSize: [8, 8]
      })
    );
    exact.dispose();
    assert.throws(
      () => new NativeUNetExecutor(
        fakeDevice({ maxComputeWorkgroupStorageSize: 6719 }),
        model,
        executorOptions('fp16', 'k-major', {
          sharedLayout: 'padded',
          rowsPerThread: 8,
          workgroupSize: [8, 8]
        })
      ),
      /Unsupported GEMM tile configuration/
    );
  });
});


test('activation slots grow for retired values and never alias overlapping lifetimes', () => {
  for (const weights of ['rt_hdr_alb_nrm_small.tza', 'rt_hdr_alb_nrm.tza', 'rt_hdr_calb_cnrm_large.tza']) {
    const bytes = readFileSync(new URL(`../weights/${weights}`, import.meta.url));
    const model = validateUNetModel(parseTZA(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)));
    for (const [precision, kernel] of [['fp16', 'implicit-gemm'], ['fp32', 'implicit-gemm'], ['fp16', 'subgroup-matrix']]) {
      const device = fakeDevice();
      device.features.add('subgroups');
      device.features.add('wgpu-cooperative-matrix');
      device.adapterInfo = {
        subgroupMinSize: 32, subgroupMaxSize: 32,
        subgroupMatrixConfigs: [{ componentType: 'f16', resultComponentType: 'f32', M: 8, N: 8, K: 8 }]
      };
      const executor = new NativeUNetExecutor(device, model, { precision, kernel });
      const execution = executor._execution(80, 48);
      const { plan, valueBuffers, slots } = execution;
      const lifetimes = new Map([[plan.spec.input, { first: -1, last: -1 }]]);
      plan.nodes.forEach((node, index) => {
        lifetimes.set(node.id, { first: index, last: index });
        const inputs = node.op === 'fusedUpsampleConcatConv2d'
          ? node.inputs.map((input) => input.value) : [node.input];
        for (const input of inputs) lifetimes.get(input).last = index;
      });
      lifetimes.get(plan.spec.output).last = plan.nodes.length;
      for (const [value, interval] of lifetimes) {
        const buffer = valueBuffers.get(value);
        const shape = plan.valueShapes.get(value);
        const size = shape.width * shape.height * Math.ceil(shape.channels / 4) * 4 * (value === plan.spec.output || precision === 'fp32' ? 4 : 2);
        assert.ok(buffer.data.byteLength >= size, `${weights}/${kernel}/${value}: capacity`);
        for (const [other, otherInterval] of lifetimes) {
          if (value === other || valueBuffers.get(other) !== buffer) continue;
          assert.ok(interval.last < otherInterval.first || otherInterval.last < interval.first,
            `${weights}/${kernel}: ${value} overlaps ${other}`);
        }
      }
      if (model.spec.family === 'oidn-unet-large' && kernel === 'subgroup-matrix') {
        assert.equal(plan.nodes.filter((node) => node.op === 'fusedConvReluMaxPool2d').length, 4);
        assert.equal(slots.length, 6);
        // Includes the slots that grow from pooled encoder values to decoder values.
        assert.equal(slots.reduce((sum, slot) => sum + slot.capacity, 0), 1167360);
      }
      executor.dispose();
      assert.ok(device.buffers.every((buffer) => buffer.destroyCalls === 1));
    }
  }
});
