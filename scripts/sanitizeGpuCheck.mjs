// Checks OIDN-style input/output sanitizing in Chrome: denoising inputs that
// contain NaN, infinities and out-of-range values must give bit-identical
// results to denoising the same inputs sanitized on the CPU, and no output
// value may be NaN or negative (+inf input may saturate to +inf HDR output, as
// in OIDN). Covers GPU-buffer and CPU HDR inputs.
//
//   npm run build && node scripts/sanitizeGpuCheck.mjs   (CHROME_PATH selects the browser)
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { chromium } from 'playwright-core';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const chrome = process.env.CHROME_PATH ?? [
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  '/usr/bin/google-chrome',
  '/usr/bin/chromium',
  'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe'
].find((candidate) => fs.existsSync(candidate));
if (!chrome) throw new Error('Set CHROME_PATH to Chrome/Chromium');

const mime = new Map([['.js', 'text/javascript'], ['.mjs', 'text/javascript'], ['.tza', 'application/octet-stream']]);
const server = http.createServer((request, response) => {
  const requestPath = decodeURIComponent((request.url ?? '/').split('?')[0]);
  if (requestPath === '/') {
    response.writeHead(200, { 'Content-Type': 'text/html' });
    response.end('<!doctype html><title>OIDN sanitize GPU check</title>');
    return;
  }
  const file = path.resolve(root, `.${requestPath}`);
  if (!file.startsWith(root + path.sep) || !fs.existsSync(file) || !fs.statSync(file).isFile()) {
    response.writeHead(404);
    response.end();
    return;
  }
  response.writeHead(200, { 'Content-Type': mime.get(path.extname(file)) ?? 'application/octet-stream' });
  fs.createReadStream(file).pipe(response);
});
await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
const browser = await chromium.launch({ executablePath: chrome, headless: true, args: ['--enable-unsafe-webgpu'] });

try {
  const page = await browser.newPage();
  await page.goto(`http://127.0.0.1:${server.address().port}/`);
  const results = await page.evaluate(async () => {
    const oidn = await import('/dist/oidn.js');
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('No WebGPU adapter');
    const device = await adapter.requestDevice({
      requiredFeatures: ['shader-f16'].filter((feature) => adapter.features.has(feature))
    });
    const weights = await (await fetch('/weights/rt_hdr_alb_nrm_small.tza')).arrayBuffer();
    const width = 61;
    const height = 47;
    const FLT_MAX = 3.4028234663852886e38;
    const special = [NaN, Infinity, -Infinity, -3];
    const clean = (value, lo, hi) => (Number.isNaN(value) ? 0 : Math.min(Math.max(value, lo), hi));
    // Smooth images with a special value in every 7th pixel of each channel.
    const image = (base, hi, lo = 0) => {
      const dirty = new Float32Array(width * height * 4);
      const sanitized = new Float32Array(dirty.length);
      for (let pixel = 0; pixel < width * height; pixel++) {
        for (let channel = 0; channel < 4; channel++) {
          const index = pixel * 4 + channel;
          let value = channel === 3 ? 1 : base(pixel % width, Math.floor(pixel / width), channel);
          if (channel < 3 && (pixel + channel * 3) % 7 === 0) value = special[(pixel + channel) % special.length];
          if (channel < 3 && (pixel + channel) % 11 === 0) value = hi === 1 ? 1.5 : value;
          dirty[index] = value;
          sanitized[index] = channel === 3 ? value : clean(value, lo, hi);
        }
      }
      return { dirty, sanitized };
    };
    const color = image((x, y, c) => 0.2 + 2 * ((x * 3 + y * 5 + c * 7) % 17) / 17, FLT_MAX);
    const albedo = image((x, y, c) => 0.3 + 0.4 * ((x + c) % 5) / 5, 1);
    const normal = image((x, y, c) => (c === 2 ? 0.9 : 0.5 + 0.3 * Math.sin((x + y + c) * 0.3)), 1);

    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const upload = (data) => {
      const buffer = device.createBuffer({ size: data.byteLength, usage });
      device.queue.writeBuffer(buffer, 0, data);
      return buffer;
    };
    const download = async (buffer) => {
      const readback = device.createBuffer({ size: buffer.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const encoder = device.createCommandEncoder();
      encoder.copyBufferToBuffer(buffer, 0, readback, 0, buffer.size);
      device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const data = new Float32Array(readback.getMappedRange().slice(0));
      readback.destroy();
      return data;
    };
    const run = (unet, inputs) => new Promise((resolve, reject) => unet.tileExecute({
      ...inputs,
      wholeImage: true,
      done: resolve,
      error: reject
    }));
    const compare = (name, a, b) => {
      let mismatches = 0;
      let invalid = 0;
      for (let i = 0; i < a.length; i++) {
        if (!Object.is(a[i], b[i])) mismatches++;
        if (!(a[i] >= 0)) invalid++;
      }
      return { name, values: a.length, mismatches, invalid };
    };

    const results = [];
    for (const precision of ['fp32', ...(device.features.has('shader-f16') ? ['fp16'] : [])]) {
      const unet = await oidn.initUNetFromBuffer(weights, { device }, { hdr: true, aux: true, precision, dynamicTile: false });
      const gpu = async (set) => download((await run(unet, {
        color: { data: upload(color[set]), width, height },
        albedo: { data: upload(albedo[set]), width, height },
        normal: { data: upload(normal[set]), width, height }
      })).data);
      results.push(compare(`GPU buffers ${precision}`, await gpu('dirty'), await gpu('sanitized')));

      // Keep auxiliary images sanitized here to isolate HDR color sanitizing.
      // Auto-exposure scales CPU color before sanitizing, as OIDN does, so +inf
      // becomes FLT_MAX while a clean FLT_MAX becomes FLT_MAX * scale: compare
      // without +inf, and only require valid output with it.
      const aux8 = (data) => new ImageData(Uint8ClampedArray.from(data, (value) => value * 255), width, height);
      const cpu = async (data) => (await run(unet, {
        color: { data, width, height },
        albedo: aux8(albedo.sanitized),
        normal: aux8(normal.sanitized)
      })).data;
      const finite = color.dirty.map((value, index) => (value === Infinity ? color.sanitized[index] : value));
      results.push(compare(`CPU HDR ${precision}`, await cpu(finite), await cpu(color.sanitized)));
      const withInfinity = await cpu(color.dirty);
      results.push(compare(`CPU HDR +inf ${precision}`, withInfinity, withInfinity));
      unet.dispose();
    }
    return results;
  });

  let failed = false;
  for (const { name, values, mismatches, invalid } of results) {
    console.log(`${name}: ${values} values, ${mismatches} mismatches, ${invalid} NaN or negative`);
    failed ||= mismatches > 0 || invalid > 0;
  }
  if (failed) throw new Error('sanitize GPU check failed');
  console.log('sanitize GPU check: OK');
} finally {
  await browser.close();
  server.close();
}
