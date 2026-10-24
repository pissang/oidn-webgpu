import assert from 'node:assert/strict';
import test from 'node:test';
import { FLT_MAX, sanitize, sanitizeWGSL } from '../lib/sanitize.js';
import {
  avgLogLum,
  hdrTransferFuncCPU,
  hdrTransferFuncInverseCPU
} from '../lib/hdrTransfer.js';

const special = [NaN, Infinity, -Infinity, -2, -0, 0, 0.5, 1, 7, FLT_MAX, -FLT_MAX];

test('sanitize maps NaN to zero and clamps like OIDN', () => {
  assert.equal(sanitize(NaN, 0, 1), 0);
  assert.equal(sanitize(Infinity, 0, FLT_MAX), FLT_MAX);
  assert.equal(sanitize(-Infinity, 0, FLT_MAX), 0);
  assert.equal(sanitize(-2, -1, 1), -1);
  assert.equal(sanitize(0.25, 0, 1), 0.25);
  assert.equal(Math.fround(FLT_MAX), FLT_MAX);
});

// Evaluates the WGSL function's bit tests on f32 values, so its select
// logic is checked without a GPU.
function sanitizeLikeWGSL(value, lo, hi) {
  const view = new DataView(new ArrayBuffer(4));
  view.setFloat32(0, value);
  const bits = view.getUint32(0);
  const magnitude = bits & 0x7fffffff;
  const negative = (bits & 0x80000000) !== 0;
  if (magnitude > 0x7f800000) return 0;
  if (magnitude === 0x7f800000) return negative ? lo : hi;
  return Math.min(Math.max(Math.fround(value), lo), hi);
}

test('WGSL sanitize logic matches the JS sanitize', () => {
  assert.equal(Number(sanitizeWGSL.match(/const FLT_MAX = ([^;]+);/)[1]), FLT_MAX);
  for (const [lo, hi] of [[0, 1], [-1, 1], [0, FLT_MAX]]) {
    for (const value of special) {
      assert.ok(
        Object.is(sanitizeLikeWGSL(value, lo, hi) + 0, sanitize(Math.fround(value), lo, hi) + 0),
        `${value} in [${lo}, ${hi}]`
      );
    }
  }
});

for (const transfer of ['pu', 'log']) {
  test(`${transfer} CPU transfer sanitizes HDR input and network output`, () => {
    const dirty = new Float32Array([NaN, Infinity, -3, 1, 2, -Infinity, 0.5, 1]);
    const clean = new Float32Array([0, FLT_MAX, 0, 1, 2, 0, 0.5, 1]);
    // Scale 1 keeps FLT_MAX in the clean image equal to sanitized +inf.
    const encode = (data) =>
      hdrTransferFuncCPU({ data, channels: 4, inputScale: 1, transfer });
    assert.deepEqual(encode(dirty), encode(clean));
    for (const value of encode(dirty)) assert.ok(Number.isFinite(value));

    const output = new Float32Array([NaN, -0.25, 0.2, 1]);
    const decoded = hdrTransferFuncInverseCPU({ data: output, channels: 4, inputScale: 0.5, transfer });
    assert.equal(decoded[0], 0);
    assert.equal(decoded[1], 0);
    assert.ok(decoded[2] > 0);
  });
}

test('auto-exposure ignores NaN and clamps infinite or negative color', () => {
  const dirty = new Float32Array([NaN, 0.2, 0.3, Infinity, 0.4, -1, 0.6, 0.7, 0.8]);
  const clean = new Float32Array([0, 0.2, 0.3, FLT_MAX, 0.4, 0, 0.6, 0.7, 0.8]);
  const scale = avgLogLum({ data: dirty, channels: 3 });
  assert.ok(Number.isFinite(scale) && scale > 0);
  assert.equal(scale, avgLogLum({ data: clean, channels: 3 }));
});
