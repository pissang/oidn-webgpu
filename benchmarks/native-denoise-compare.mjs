// End-to-end check of the native C API (native/include/oidn_web.h) against
// the browser runtime: both denoise examples/test (HDR color, albedo,
// normal) as RGBA f32 GPU buffers, and the outputs are compared.
//
//   npm run build && npm run build:lib && npm run native:build
//   node benchmarks/native-denoise-compare.mjs --denoise PATH_TO_EXAMPLE
//     [--weights rt_hdr_alb_nrm.tza] [--quality 0] [--max-tile 0] [--kernel 0-3]
//     [--browser-kernel implicit-gemm] [--browser-precision fp32]
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
const denoise = option('--denoise');
if (!denoise) throw new Error('--denoise must name the native/examples/denoise.c binary');
const quality = option('--quality', '0');
const nativeKernel = option('--kernel', '0');
const maxTile = option('--max-tile', '0');
const browserKernel = option('--browser-kernel', 'implicit-gemm');
const browserPrecision = option('--browser-precision', 'fp32');
const chrome = option('--chrome', process.env.CHROME_PATH) ??
  ['/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', '/usr/bin/google-chrome', '/usr/bin/chromium']
    .find((candidate) => existsSync(candidate));

const files = new Map([
  ['/oidn.js', [path.join(root, 'dist/oidn.js'), 'text/javascript']],
  ['/hdr.js', [path.join(root, 'lib/hdr.js'), 'text/javascript']],
  ['/model.tza', [path.join(root, 'weights', weights), 'application/octet-stream']],
  ['/noisy.hdr', [path.join(root, 'examples/test/noisy.hdr'), 'application/octet-stream']],
  ['/albedo.png', [path.join(root, 'examples/test/albedo.png'), 'image/png']],
  ['/normal.png', [path.join(root, 'examples/test/normal.png'), 'image/png']]
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
const temporary = await mkdtemp(path.join(os.tmpdir(), 'oidn-native-denoise-'));
const browser = await chromium.launch({ executablePath: chrome, headless: true, args: ['--enable-unsafe-webgpu'] });
try {
  const page = await browser.newPage();
  await page.goto(`http://127.0.0.1:${server.address().port}/`);
  const result = await page.evaluate(async ({ kernel, precision, maxTile }) => {
    const oidn = await import('/oidn.js');
    const { readHDR } = await import('/hdr.js');
    const hdr = readHDR(new Uint8Array(await (await fetch('/noisy.hdr')).arrayBuffer()));
    const { width, height } = hdr;
    const color = new Float32Array(width * height * 4);
    for (let pixel = 0; pixel < width * height; pixel++) {
      color.set(hdr.rgbFloat.subarray(pixel * 3, pixel * 3 + 3), pixel * 4);
      color[pixel * 4 + 3] = 1;
    }
    const image = async (url) => {
      const bitmap = await createImageBitmap(await (await fetch(url)).blob());
      const canvas = new OffscreenCanvas(width, height);
      const context = canvas.getContext('2d');
      context.drawImage(bitmap, 0, 0, width, height);
      return Float32Array.from(context.getImageData(0, 0, width, height).data, (value) => value / 255);
    };
    const albedo = await image('/albedo.png');
    const normal = await image('/normal.png');
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    const features = ['shader-f16', 'subgroups', 'chromium-experimental-subgroup-matrix']
      .filter((feature) => adapter.features.has(feature));
    const device = await adapter.requestDevice({
      requiredFeatures: features,
      requiredLimits: {
        maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize: adapter.limits.maxBufferSize
      }
    });
    const unet = await oidn.initUNetFromBuffer(
      await (await fetch('/model.tza')).arrayBuffer(),
      { device },
      { aux: true, hdr: true, kernel, precision, dynamicTile: false, maxTileSize: maxTile || 4096 }
    );
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const buffer = (data) => {
      const gpuBuffer = device.createBuffer({ size: data.byteLength, usage });
      device.queue.writeBuffer(gpuBuffer, 0, data);
      return gpuBuffer;
    };
    const output = await new Promise((resolve, reject) => unet.tileExecute({
      color: { data: buffer(color), width, height },
      albedo: { data: buffer(albedo), width, height },
      normal: { data: buffer(normal), width, height },
      wholeImage: !maxTile,
      done: resolve,
      error: reject
    }));
    const readback = device.createBuffer({ size: color.byteLength, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(output.data, 0, readback, 0, color.byteLength);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const denoised = new Float32Array(readback.getMappedRange().slice(0));
    const base64 = (array) => {
      const bytes = new Uint8Array(array.buffer);
      let text = '';
      for (let offset = 0; offset < bytes.length; offset += 0x8000) {
        text += String.fromCharCode(...bytes.subarray(offset, offset + 0x8000));
      }
      return btoa(text);
    };
    return { width, height, color: base64(color), albedo: base64(albedo), normal: base64(normal), denoised: base64(denoised) };
  }, { kernel: browserKernel, precision: browserPrecision, maxTile: Number(maxTile) });

  const paths = {};
  for (const name of ['color', 'albedo', 'normal']) {
    paths[name] = path.join(temporary, `${name}.f32`);
    await writeFile(paths[name], Buffer.from(result[name], 'base64'));
  }
  const outputPath = path.join(temporary, 'native.f32');
  execFileSync(denoise, [
    path.join(root, 'weights', weights), String(result.width), String(result.height),
    paths.color, paths.albedo, paths.normal, outputPath, quality, maxTile, '3', nativeKernel
  ], { stdio: 'inherit' });
  const native = new Float32Array((await readFile(outputPath)).buffer.slice(0));
  const reference = new Float32Array(Buffer.from(result.denoised, 'base64').buffer.slice(0));
  const color = new Float32Array(Buffer.from(result.color, 'base64').buffer.slice(0));
  let relative = 0;
  let absolute = 0;
  let alphaMismatches = 0;
  for (let pixel = 0; pixel < result.width * result.height; pixel++) {
    for (let channel = 0; channel < 3; channel++) {
      const expected = reference[pixel * 4 + channel];
      const actual = native[pixel * 4 + channel];
      absolute += Math.abs(expected - actual);
      relative += Math.abs(expected - actual) / (Math.abs(expected) + Math.abs(actual) + 0.01);
    }
    if (native[pixel * 4 + 3] !== color[pixel * 4 + 3]) alphaMismatches++;
  }
  const count = result.width * result.height * 3;
  if (args.includes('--debug')) {
    const stats = (array) => {
      let sum = 0;
      let max = -Infinity;
      for (let i = 0; i < array.length; i += 4) {
        sum += array[i];
        max = Math.max(max, array[i]);
      }
      return `mean ${(sum / (array.length / 4)).toFixed(4)} max ${max.toFixed(3)}`;
    };
    console.log(`color ${stats(color)}; browser ${stats(reference)}; native ${stats(native)}`);
    console.log('pixel 1000: color', color.slice(4000, 4004), 'browser', reference.slice(4000, 4004), 'native', native.slice(4000, 4004));
  }
  console.log(`${weights} ${result.width}x${result.height}: native (quality ${quality}, kernel ${nativeKernel}) vs browser ` +
    `${browserKernel}/${browserPrecision}: avgerror ${(relative / count).toExponential(3)}, ` +
    `MAE ${(absolute / count).toExponential(3)}, alpha mismatches ${alphaMismatches}, ` +
    `bit-identical ${native.every((value, index) => Object.is(value, reference[index]))}`);
} finally {
  await browser.close();
  server.close();
  await rm(temporary, { recursive: true, force: true });
}
