import test from 'node:test';
import assert from 'node:assert/strict';

import { UNet } from '../dist/oidn.js';
import { avgLogLum, hdrTransferFuncCPU } from '../lib/hdrTransfer.js';
import { planTileGrid } from '../lib/tileScheduler.js';

if (typeof globalThis.GPUBuffer === 'undefined') {
  globalThis.GPUBuffer = class GPUBuffer {};
}
if (typeof globalThis.GPUTexture === 'undefined') {
  globalThis.GPUTexture = class GPUTexture {};
}
if (typeof globalThis.ImageData === 'undefined') {
  globalThis.ImageData = class ImageData {
    constructor(width, height) {
      this.width = width;
      this.height = height;
      this.data = new Uint8ClampedArray(width * height * 4);
    }
  };
}

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

function createUNet(executeTile, queueDone = async () => {}, deviceLost) {
  const unet = Object.create(UNet.prototype);
  unet._aux = false;
  unet._hdr = false;
  unet._modelSpec = { receptiveField: 0 };
  unet._dynamicTileController = { tileSize: 16, observe() {} };
  unet._device = {
    queue: { onSubmittedWorkDone: queueDone },
    ...(deviceLost ? { lost: deviceLost } : {})
  };
  unet._processImageData = () => new Float32Array(32 * 16 * 3);
  unet._executeTile = executeTile;
  return unet;
}

function run(unet, overrides = {}) {
  return new Promise((resolve) => {
    const abort = unet.tileExecute({
      color: new ImageData(32, 16),
      done: (output) => resolve({ type: 'done', output }),
      error: (reason) => resolve({ type: 'error', reason }),
      ...overrides
    });
    if (overrides.captureAbort) overrides.captureAbort(abort);
  });
}

test('normalizes CPU auxiliary images by their own storage type', () => {
  const unet = Object.create(UNet.prototype);
  unet._aux = true;
  const image = (data) => ({ data, width: 2, height: 1 });
  const albedo8 = new Uint8ClampedArray([51, 102, 153, 255, 204, 255, 0, 255]);
  const normal8 = new Uint8ClampedArray([0, 128, 255, 255, 255, 64, 0, 255]);
  const normalized = (data) => Float32Array.from(data, (value) => value / 255);
  for (const hdr of [false, true]) {
    const color = image(hdr
      ? new Float32Array([0.5, 2, 4, 1, 8, 0.25, 1, 1])
      : new Uint8ClampedArray([32, 64, 128, 255, 255, 0, 64, 255]));
    const expected = new Float32Array(18);
    for (let pixel = 0; pixel < 2; pixel++) {
      for (let channel = 0; channel < 3; channel++) {
        expected[pixel * 9 + channel] = color.data[pixel * 4 + channel] / (hdr ? 1 : 255);
        expected[pixel * 9 + 3 + channel] = albedo8[pixel * 4 + channel] / 255;
        expected[pixel * 9 + 6 + channel] = normal8[pixel * 4 + channel] / 255;
      }
    }
    for (const albedo of [albedo8, normalized(albedo8)]) {
      for (const normal of [normal8, normalized(normal8)]) {
        assert.deepEqual(unet._processImageData(color, image(albedo), image(normal), hdr), expected);
      }
    }
  }
});

test('validates sizes of floating-point CPU auxiliary images', () => {
  const unet = Object.create(UNet.prototype);
  unet._aux = true;
  const image = (width, height) => ({ data: new Float32Array(width * height * 4), width, height });
  assert.throws(() => unet._processImageData(image(2, 1), image(1, 2), image(1, 2), true), /Image size mismatch/);
  assert.throws(() => unet._processImageData(image(2, 1), image(2, 1), image(1, 2), true), /Image size mismatch/);
});

test('CPU HDR tiles transfer real pixels and zero-pad after estimating exposure', async () => {
  for (const [width, height] of [[64, 23], [23, 64], [64, 32]]) {
    const color = { width, height, data: new Float32Array(width * height * 4) };
    const albedo = { width, height, data: new Float32Array(color.data.length).fill(0.375) };
    const normal = { width, height, data: new Float32Array(color.data.length).fill(0.75) };
    for (let pixel = 0; pixel < width * height; pixel++) {
      color.data.set([0.25 + pixel / 128, 0.75, 2, 0.5], pixel * 4);
    }
    for (const wholeImage of [true, false]) {
      const unet = createUNet();
      delete unet._executeTile;
      delete unet._processImageData;
      unet._hdr = true;
      unet._aux = true;
      unet._dynamicTileController.tileSize = 32;
      const { tiles } = planTileGrid(width, height, wholeImage ? 64 : 32, 16);
      let tileIndex = 0;
      unet._nativeExecutor = {
        async executeCPU(tensor, tensorWidth, tensorHeight) {
          const tile = tiles[tileIndex++].input;
          assert.equal(tensorWidth, tile.width);
          assert.equal(tensorHeight, tile.height);
          assert.equal(tensor.length, tile.width * tile.height * 9);
          const validPixels = [];
          for (let y = tile.y; y < Math.min(tile.y + tile.height, height); y++) {
            for (let x = tile.x; x < Math.min(tile.x + tile.width, width); x++) {
              validPixels.push(x + y * width);
            }
          }
          const rgb = Float32Array.from(validPixels.flatMap(
            (pixel) => Array.from(color.data.subarray(pixel * 4, pixel * 4 + 3))
          ));
          const encoded = hdrTransferFuncCPU({
            data: rgb, channels: 3, inputScale: avgLogLum({ data: rgb, channels: 3 })
          });
          const expected = new Float32Array(tensor.length);
          validPixels.forEach((pixel, index) => {
            const x = pixel % width - tile.x;
            const y = Math.floor(pixel / width) - tile.y;
            const offset = (y * tensorWidth + x) * 9;
            expected.set(encoded.subarray(index * 3, index * 3 + 3), offset);
            expected.set([0.375, 0.375, 0.375, 0.75, 0.75, 0.75], offset + 3);
          });
          assert.deepEqual(tensor, expected);
          const output = new Float32Array(tensorWidth * tensorHeight * 3);
          for (let pixel = 0; pixel < output.length / 3; pixel++) {
            output.set(tensor.subarray(pixel * 9, pixel * 9 + 3), pixel * 3);
          }
          return output;
        }
      };
      const result = await run(unet, { color, albedo, normal, wholeImage, tileOverlap: 16 });
      if (result.type === 'error') throw result.reason;
      assert.equal(tileIndex, tiles.length);
      assert.equal(result.output.width, width);
      assert.equal(result.output.height, height);
      for (let i = 0; i < color.data.length; i++) {
        if (i % 4 === 3) {
          assert.equal(result.output.data[i], 1);
        } else {
          assert.ok(Math.abs(result.output.data[i] - color.data[i]) < 1e-5, `pixel value ${i}`);
        }
      }
    }
  }
});

test('profiles every tile and sums matching network nodes for the complete image', async () => {
  let captured = 0;
  const unet = createUNet(async () => {});
  unet._device.features = new Set(['timestamp-query']);
  unet._nativeExecutor = {
    profileNextExecution() { captured++; return true; },
    getLastExecutionProfile: async () => ({ totalMs: captured + 0.5,
      layers: [{ id: 'input-pack', durationMs: 0.5 }, { id: 'conv', durationMs: captured }] })
  };
  assert.equal(unet.profileNextExecution(), true);
  assert.equal((await run(unet)).type, 'done');
  assert.equal(captured, 2);
  assert.deepEqual(await unet.getLastExecutionProfile(), {
    totalMs: 4, layers: [{ id: 'input-pack', durationMs: 1 }, { id: 'conv', durationMs: 3 }]
  });
  await run(unet);
  assert.equal(captured, 2);
});

test('reports first and later asynchronous tile failures exactly once', async () => {
  const firstFailure = new Error('first tile failed');
  const first = await run(createUNet(async () => { throw firstFailure; }));
  assert.equal(first.type, 'error');
  assert.equal(first.reason, firstFailure);

  let calls = 0;
  const laterFailure = new Error('later tile failed');
  const later = await run(createUNet(async () => {
    calls++;
    if (calls === 2) throw laterFailure;
  }));
  assert.equal(later.type, 'error');
  assert.equal(later.reason, laterFailure);
  assert.equal(calls, 2);
});

test('uses event-loop scheduling by default even when RAF never delivers', async () => {
  const originalRAF = globalThis.requestAnimationFrame;
  let rafCalls = 0;
  globalThis.requestAnimationFrame = () => {
    rafCalls++;
    return 1;
  };
  try {
    const result = await run(createUNet(async () => {}));
    assert.equal(result.type, 'done');
    assert.equal(rafCalls, 0);
  } finally {
    if (originalRAF === undefined) delete globalThis.requestAnimationFrame;
    else globalThis.requestAnimationFrame = originalRAF;
  }
});

async function withRAF(requestAnimationFrame, body) {
  const originalRAF = globalThis.requestAnimationFrame;
  const originalCancelRAF = globalThis.cancelAnimationFrame;
  const cancelled = [];
  globalThis.requestAnimationFrame = requestAnimationFrame;
  globalThis.cancelAnimationFrame = (id) => cancelled.push(id);
  try {
    return await body(cancelled);
  } finally {
    if (originalRAF === undefined) delete globalThis.requestAnimationFrame;
    else globalThis.requestAnimationFrame = originalRAF;
    if (originalCancelRAF === undefined) delete globalThis.cancelAnimationFrame;
    else globalThis.cancelAnimationFrame = originalCancelRAF;
  }
}

test('animation-frame scheduling advances tiles on display frames', async () => {
  let rafCalls = 0;
  await withRAF((callback) => {
    rafCalls++;
    queueMicrotask(callback);
    return rafCalls;
  }, async () => {
    const result = await run(createUNet(async () => {}), {
      scheduling: 'animation-frame'
    });
    assert.equal(result.type, 'done');
    assert.equal(rafCalls, 1);
  });
});

test('animation-frame scheduling completes when RAF never delivers', async () => {
  let rafCalls = 0;
  await withRAF(() => ++rafCalls, async (cancelled) => {
    const result = await run(createUNet(async () => {}), {
      scheduling: 'animation-frame'
    });
    assert.equal(result.type, 'done');
    assert.equal(rafCalls, 1);
    assert.deepEqual(cancelled, [1]);
  });
});

test('wholeImage executes an unaligned image as a single tile', async () => {
  const tiles = [];
  const unet = createUNet(async (_input, _tileData, _output, tile) => {
    tiles.push(tile);
  });
  unet._processImageData = () => new Float32Array(40 * 24 * 3);
  const result = await run(unet, {
    color: new ImageData(40, 24),
    wholeImage: true
  });
  assert.equal(result.type, 'done');
  assert.equal(tiles.length, 1);
  assert.deepEqual(tiles[0].output, { x: 0, y: 0, width: 40, height: 24 });
});

test('routes queue and callback failures through the error callback', async () => {
  const queueFailure = new Error('queue failed');
  const queue = await run(createUNet(
    async () => {},
    async () => { throw queueFailure; }
  ));
  assert.equal(queue.type, 'error');
  assert.equal(queue.reason, queueFailure);

  const progressFailure = new Error('progress failed');
  const progress = await run(createUNet(async () => {}), {
    progress: async () => { throw progressFailure; }
  });
  assert.equal(progress.type, 'error');
  assert.equal(progress.reason, progressFailure);

  const doneFailure = new Error('done failed');
  const done = await new Promise((resolve) => {
    createUNet(async () => {}).tileExecute({
      color: new ImageData(16, 16),
      done: async () => { throw doneFailure; },
      error: (reason) => resolve(reason)
    });
  });
  assert.equal(done, doneFailure);
});

test('cancel wins an in-flight tile race without done or error callbacks', async () => {
  const inFlight = deferred();
  let doneCalls = 0;
  let errorCalls = 0;
  const unet = createUNet(() => inFlight.promise);
  const abort = unet.tileExecute({
    color: new ImageData(16, 16),
    done: () => { doneCalls++; },
    error: () => { errorCalls++; }
  });
  abort();
  inFlight.resolve();
  await new Promise((resolve) => setTimeout(resolve, 10));
  assert.equal(doneCalls, 0);
  assert.equal(errorCalls, 0);
});

test('reports device loss once while a tile is in flight', async () => {
  const inFlight = deferred();
  const lost = deferred();
  const resultPromise = run(createUNet(
    () => inFlight.promise,
    async () => {},
    lost.promise
  ));
  lost.resolve({ message: 'adapter reset' });
  const result = await resultPromise;
  assert.equal(result.type, 'error');
  assert.match(String(result.reason), /adapter reset/);
  inFlight.resolve();
});

test('uses one device-loss observer and releases completed execution closures', async () => {
  const lost = deferred();
  let observerCount = 0;
  const observedLost = {
    then(onFulfilled, onRejected) {
      observerCount++;
      return lost.promise.then(onFulfilled, onRejected);
    }
  };
  const unet = createUNet(async () => {}, async () => {}, observedLost);

  for (let execution = 0; execution < 4; execution++) {
    assert.equal((await run(unet)).type, 'done');
    assert.equal(unet._activeExecutionFailures.size, 0);
  }
  assert.equal(observerCount, 1);

  const failure = new Error('later execution failed');
  unet._executeTile = async () => { throw failure; };
  const failed = await run(unet);
  assert.equal(failed.type, 'error');
  assert.equal(failed.reason, failure);
  assert.equal(unet._activeExecutionFailures.size, 0);
  assert.equal(observerCount, 1);

  const inFlight = deferred();
  unet._executeTile = () => inFlight.promise;
  const abort = unet.tileExecute({
    color: new ImageData(16, 16),
    done: () => assert.fail('cancelled execution completed'),
    error: () => assert.fail('cancelled execution failed')
  });
  assert.equal(unet._activeExecutionFailures.size, 1);
  abort();
  assert.equal(unet._activeExecutionFailures.size, 0);
  assert.equal(observerCount, 1);
  inFlight.resolve();
  lost.resolve({ message: 'loss after every execution settled' });
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.equal(unet._activeExecutionFailures.size, 0);
});
