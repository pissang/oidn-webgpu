// Build first: npm run build. Runs on ordinary Chrome without unsafe WebGPU.
// Checks every output value against a CPU convolution and FP32 network output.
import { createServer } from 'node:http';
import { existsSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const chrome = process.env.CHROME_PATH ?? [
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  '/usr/bin/google-chrome', '/usr/bin/chromium',
  ...['PROGRAMFILES', 'PROGRAMFILES(X86)', 'LOCALAPPDATA'].flatMap((key) =>
    process.env[key] ? [path.join(process.env[key], 'Google/Chrome/Application/chrome.exe')] : [])
].find(existsSync);
if (!chrome) throw new Error('Set CHROME_PATH to Chrome/Chromium');
const files = new Map([
  ['/oidn.js', ['dist/oidn.js', 'text/javascript']],
  ...['rt_hdr_alb_nrm', 'rt_hdr_alb_nrm_small', 'rt_hdr_calb_cnrm_large'].map((name) =>
    [`/${name}`, [`weights/${name}.tza`, 'application/octet-stream']])
]);
const server = createServer(async (request, response) => {
  const file = files.get(request.url);
  response.writeHead(200, { 'content-type': file?.[1] ?? 'text/html' });
  response.end(file ? await readFile(path.join(root, file[0])) : '<!doctype html><title>GEMM sub-pixel check</title>');
}).listen(0, '127.0.0.1');
await new Promise((resolve) => server.once('listening', resolve));
let browser;
try {
  browser = await chromium.launch({ executablePath: chrome, headless: true });
  const page = await browser.newPage();
  await page.goto(`http://127.0.0.1:${server.address().port}/`);
  const results = await page.evaluate(async () => {
    const oidn = await import('/oidn.js');
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter?.features.has('shader-f16')) throw new Error('shader-f16 is required');
    const device = await adapter.requestDevice({ requiredFeatures: ['shader-f16'] });
    const failures = [];
    device.addEventListener('uncapturederror', (event) => failures.push(event.error.message));
    const results = [];
    let state = 7;
    const random = () => ((state = (Math.imul(state, 1664525) + 1013904223) >>> 0) >>> 8) / 2 ** 24;
    const compare = (actual, expected, name, tolerance) => {
      let absolute = 0, relative = 0, max = 0, outliers = 0;
      if (actual.length !== expected.length) throw new Error(`${name}: length mismatch`);
      for (let i = 0; i < actual.length; i++) {
        if (!Number.isFinite(actual[i])) throw new Error(`${name}: non-finite output at ${i}`);
        const error = Math.abs(actual[i] - expected[i]);
        const rel = error / (Math.abs(actual[i]) + Math.abs(expected[i]) + 0.01);
        absolute += error; relative += rel; max = Math.max(max, error);
        if (error > 0.02 && rel > 0.05) outliers++;
      }
      const result = { name, mae: absolute / actual.length, avgerror: relative / actual.length, max, outliers };
      if (result.avgerror > tolerance || outliers) throw new Error(JSON.stringify(result));
      results.push(result);
    };
    // CPU cross-correlation uses the original nine taps, without combining
    // weights or sharing the GPU's layout, address cache or reduction code.
    const convolve = (input, width, height, inputs, outputs, weights, bias, relu) => {
      const output = new Float32Array(width * height * outputs);
      for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) for (let o = 0; o < outputs; o++) {
        let sum = bias[o];
        for (let ky = 0; ky < 3; ky++) for (let kx = 0; kx < 3; kx++) {
          const sx = x + kx - 1, sy = y + ky - 1;
          if (sx < 0 || sy < 0 || sx >= width || sy >= height) continue;
          for (let c = 0; c < inputs; c++) sum += input[(sy * width + sx) * inputs + c] * weights[((o * inputs + c) * 3 + ky) * 3 + kx];
        }
        output[(y * width + x) * outputs + o] = relu ? Math.max(sum, 0) : sum;
      }
      return output;
    };
    const tensor = (dims, values, layout) => ({
      desc: { dims, paddedDims: dims, layout, dataType: 'Float32', getByteSize: () => values.byteLength },
      data: new Uint8Array(values.buffer)
    });
    for (const reverse of [false, true]) {
      const channels = { branch: 12, skip: 4, decoder: 20, output: 3 };
      const conv = (id, input, activation = 'relu') => ({ op: 'conv2d', id, input, weight: `${id}.weight`, bias: `${id}.bias`, activation, padding: 'same' });
      const spec = {
        schemaVersion: 1, id: 'subpixel-gemm-check', family: 'test', input: 'input', output: 'output', receptiveField: 9,
        nodes: [conv('branch', 'input'),
          { op: 'maxPool2d', id: 'pool', input: 'branch', size: 2, stride: 2, padding: 'same' },
          { op: 'upsample2d', id: 'up', input: 'pool', scale: 2, mode: 'nearest' }, conv('skip', 'input'),
          { op: 'concat', id: 'join', inputs: reverse ? ['skip', 'up'] : ['up', 'skip'], axis: 'channels' },
          conv('decoder', 'join'), conv('output', 'decoder', 'identity')]
      };
      const tensors = new Map(), data = {};
      for (const [id, inputs] of [['branch', 3], ['skip', 3], ['decoder', 16], ['output', 20]]) {
        const outputs = channels[id];
        const weights = Float32Array.from({ length: outputs * inputs * 9 }, () => (random() - 0.5) * 0.08);
        const bias = Float32Array.from({ length: outputs }, () => 0.05 + random() * 0.1);
        data[id] = { weights, bias };
        tensors.set(`${id}.weight`, tensor([outputs, inputs, 3, 3], weights, 'oihw'));
        tensors.set(`${id}.bias`, tensor([outputs], bias, 'x'));
      }
      const model = oidn.validateUNetModel(tensors, spec);
      for (const [width, height] of [[2, 2], [10, 6], [34, 18]]) {
        const input = Float32Array.from({ length: width * height * 3 }, random);
        const branch = convolve(input, width, height, 3, 12, data.branch.weights, data.branch.bias, true);
        const skip = convolve(input, width, height, 3, 4, data.skip.weights, data.skip.bias, true);
        const joined = new Float32Array(width * height * 16);
        for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
          const pooled = new Float32Array(12);
          for (let c = 0; c < 12; c++) pooled[c] = Math.max(...[0, 1, 2, 3].map((p) =>
            branch[(((y >> 1) * 2 + (p >> 1)) * width + (x >> 1) * 2 + (p & 1)) * 12 + c]));
          const pixel = (y * width + x);
          joined.set(reverse ? skip.subarray(pixel * 4, pixel * 4 + 4) : pooled, pixel * 16);
          joined.set(reverse ? pooled : skip.subarray(pixel * 4, pixel * 4 + 4), pixel * 16 + (reverse ? 4 : 12));
        }
        const decoder = convolve(joined, width, height, 16, 20, data.decoder.weights, data.decoder.bias, true);
        const expected = convolve(decoder, width, height, 20, 3, data.output.weights, data.output.bias, false);
        for (const precision of ['fp32', 'fp16']) for (const loadMode of ['native', 'packed-all']) {
          const executor = new oidn.NativeUNetExecutor(device, model, { precision, kernel: 'implicit-gemm',
            gemm: { decoderSubPixel: true, unrollRows: loadMode === 'packed-all', loadMode, rowsPerThread: 2, workgroupSize: [4, 4] } });
          try {
            await executor.prepare();
            compare(await executor.executeCPU(input, width, height), expected,
              `cpu/${reverse ? 'skip-first' : 'up-first'}/${width}x${height}/${precision}/${loadMode}`, precision === 'fp32' ? 1e-5 : 0.003);
          } finally { executor.dispose(); }
        }
      }
    }
    for (const name of ['rt_hdr_alb_nrm', 'rt_hdr_alb_nrm_small', 'rt_hdr_calb_cnrm_large']) {
      const tensors = oidn.parseTZA(await (await fetch(`/${name}`)).arrayBuffer());
      const model = oidn.validateUNetModel(tensors, oidn.detectUNetModelSpec(tensors));
      const input = Float32Array.from({ length: 64 * 48 * 9 }, random);
      const reference = new oidn.NativeUNetExecutor(device, model, { precision: 'fp32', kernel: 'implicit-gemm' });
      const executor = new oidn.NativeUNetExecutor(device, model, { precision: 'fp16', kernel: 'implicit-gemm' });
      try {
        await reference.prepare(); await executor.prepare();
        compare(await executor.executeCPU(input, 64, 48), await reference.executeCPU(input, 64, 48), name, 0.003);
      } finally { reference.dispose(); executor.dispose(); }
    }
    await device.queue.onSubmittedWorkDone();
    device.destroy();
    if (failures.length) throw new Error(failures.join('\n'));
    return results;
  });
  for (const result of results) console.log(`PASS ${result.name}: avgerror ${result.avgerror.toExponential(3)}, max ${result.max.toExponential(3)}, outliers ${result.outliers}`);
} finally {
  await browser?.close();
  server.close();
}
