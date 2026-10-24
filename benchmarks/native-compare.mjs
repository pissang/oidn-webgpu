// Compares the native wgpu runtime (native/, via scriptc) with the browser
// runtime on the same network input tensor.
//
//   npm run build && cargo build --release --manifest-path native/Cargo.toml
//   node benchmarks/native-compare.mjs [--weights rt_hdr_alb_nrm.tza] [--size 256]
//     [--kernel implicit-gemm] [--chrome PATH] [--unchecked]
//     [--browser-kernel K] [--browser-precision fp16|fp32]
// The browser runs the same kernel and precision unless told otherwise, so
// the default comparison expects bit-identical outputs.
import { execFileSync } from 'node:child_process';
import { createServer } from 'node:http';
import { existsSync } from 'node:fs';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const args = process.argv.slice(2);
const option = (name, fallback) => (args.includes(name) ? args[args.indexOf(name) + 1] : fallback);
const weights = option('--weights', 'rt_hdr_alb_nrm.tza');
const size = Number(option('--size', '256'));
const kernel = option('--kernel', 'implicit-gemm');
const browserKernel = option('--browser-kernel', kernel);
const browserPrecision = option('--browser-precision', 'fp16');
const chrome = option('--chrome', process.env.CHROME_PATH) ??
  ['/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', '/usr/bin/google-chrome', '/usr/bin/chromium']
    .find((candidate) => existsSync(candidate));
const bench = path.join(root, 'native/target/release', process.platform === 'win32' ? 'oidn-web-bench.exe' : 'oidn-web-bench');

const temporary = await mkdtemp(path.join(os.tmpdir(), 'oidn-native-compare-'));
try {
  const tza = await readFile(path.join(root, 'weights', weights));
  const channels = weights.includes('alb_nrm') || weights.includes('calb_cnrm') ? 9 : weights.includes('alb') ? 6 : 3;
  const input = new Float32Array(size * size * channels);
  let state = 7;
  for (let index = 0; index < input.length; index++) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    input[index] = (state >>> 8) / 2 ** 24;
  }
  const inputPath = path.join(temporary, 'input.f32');
  const outputPath = path.join(temporary, 'native.f32');
  await writeFile(inputPath, Buffer.from(input.buffer));

  execFileSync(bench, [
    '--weights', path.join(root, 'weights', weights),
    '--width', String(size), '--height', String(size), '--kernel', kernel,
    '--batch', '2', '--rounds', '1', '--input', inputPath, '--output', outputPath,
    ...['--unchecked', '--double-buffer', '--no-double-buffer', '--f16-accumulation', '--no-sub-pixel']
      .filter((flag) => args.includes(flag))
  ], { stdio: 'inherit' });
  const native = new Float32Array((await readFile(outputPath)).buffer.slice(0));

  const files = new Map([
    ['/oidn.js', [path.join(root, 'dist/oidn.js'), 'text/javascript']],
    ['/model.tza', [path.join(root, 'weights', weights), 'application/octet-stream']],
    ['/input.f32', [inputPath, 'application/octet-stream']]
  ]);
  const server = createServer(async (request, response) => {
    const file = files.get(new URL(request.url, 'http://127.0.0.1').pathname);
    if (!file) {
      response.writeHead(200, { 'content-type': 'text/html' });
      response.end('<!doctype html><link rel="icon" href="data:,">');
      return;
    }
    response.writeHead(200, { 'content-type': file[1] });
    response.end(await readFile(file[0]));
  }).listen(0, '127.0.0.1');
  await new Promise((resolve) => server.once('listening', resolve));
  const browser = await chromium.launch({ executablePath: chrome, headless: true, args: ['--enable-unsafe-webgpu'] });
  let browserOutput;
  try {
    const page = await browser.newPage();
    await page.goto(`http://127.0.0.1:${server.address().port}/`);
    browserOutput = await page.evaluate(async ({ size, kernel, precision }) => {
      const oidn = await import('/oidn.js');
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
      const device = await adapter.requestDevice({ requiredFeatures: ['shader-f16'] });
      const tensors = oidn.parseTZA(await (await fetch('/model.tza')).arrayBuffer());
      const model = oidn.validateUNetModel(tensors, oidn.detectUNetModelSpec(tensors));
      const executor = new oidn.NativeUNetExecutor(device, model, { precision, kernel });
      await executor.prepare();
      const input = new Float32Array(await (await fetch('/input.f32')).arrayBuffer());
      return Array.from(await executor.executeCPU(input, size, size));
    }, { size, kernel: browserKernel, precision: browserPrecision });
  } finally {
    await browser.close();
    server.close();
  }

  let sum = 0;
  let relative = 0;
  let max = 0;
  let maxReference = 0;
  for (let index = 0; index < native.length; index++) {
    const error = Math.abs(native[index] - browserOutput[index]);
    sum += error;
    // Open Image Denoise's oidnDenoise -r metric (here in the network's transfer space).
    relative += error / (Math.abs(browserOutput[index]) + Math.abs(native[index]) + 0.01);
    max = Math.max(max, error);
    maxReference = Math.max(maxReference, Math.abs(browserOutput[index]));
  }
  const identical = native.every((value, index) => Object.is(value, Math.fround(browserOutput[index])));
  console.log(`${weights} ${size}x${size} native ${kernel} vs browser ${browserKernel}/${browserPrecision}: MAE ${(sum / native.length).toExponential(3)}, ` +
    `max ${max.toExponential(3)} (output max ${maxReference.toFixed(3)}), ` +
    `avgerror ${(relative / native.length).toExponential(3)}, bit-identical ${identical}`);
} finally {
  await rm(temporary, { recursive: true, force: true });
}
