// Probes chromium-experimental-subgroup-matrix support and measures the
// register-only multiply-add peak of every reported configuration, next to a
// vec4 FMA peak, in headless Chrome.
//
//   node benchmarks/subgroup-matrix.mjs [--chrome PATH] [--json OUT]
import { createServer } from 'node:http';
import { existsSync } from 'node:fs';
import { writeFile } from 'node:fs/promises';
import path from 'node:path';
import { chromium } from 'playwright-core';

const args = process.argv.slice(2);
const option = (name) => (args.includes(name) ? args[args.indexOf(name) + 1] : undefined);

function defaultChrome() {
  return [
    process.env.CHROME_PATH,
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/Applications/Chromium.app/Contents/MacOS/Chromium',
    '/usr/bin/google-chrome',
    '/usr/bin/chromium',
    ...['PROGRAMFILES', 'PROGRAMFILES(X86)', 'LOCALAPPDATA'].map((name) =>
      process.env[name] && path.join(process.env[name], 'Google/Chrome/Application/chrome.exe'))
  ].find((candidate) => candidate && existsSync(candidate));
}

async function pageMain() {
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  const info = adapter.info;
  const configs = [...(info.subgroupMatrixConfigs ?? [])].map((c) => ({
    componentType: c.componentType,
    resultComponentType: c.resultComponentType,
    M: c.M,
    N: c.N,
    K: c.K
  }));
  const wanted = ['shader-f16', 'subgroups', 'timestamp-query', 'chromium-experimental-subgroup-matrix'];
  const requiredFeatures = wanted.filter((name) => adapter.features.has(name));
  const device = await adapter.requestDevice({ requiredFeatures });
  const report = {
    userAgent: navigator.userAgent,
    vendor: info.vendor,
    architecture: info.architecture,
    subgroupMinSize: info.subgroupMinSize,
    subgroupMaxSize: info.subgroupMaxSize,
    maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
    features: requiredFeatures,
    configs,
    peaks: []
  };
  if (!device.features.has('timestamp-query')) return report;

  const subgroupSize = info.subgroupMaxSize || 32;
  const workgroupSize = subgroupSize * 4;
  const workgroups = 8192;
  const iterations = 2048;
  const sink = device.createBuffer({ size: 1 << 20, usage: GPUBufferUsage.STORAGE });
  const source = device.createBuffer({ size: 1 << 16, usage: GPUBufferUsage.STORAGE });
  const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(params, 0, new Uint32Array([iterations, 0, 0, 0]));
  const querySet = device.createQuerySet({ type: 'timestamp', count: 2 });
  const resolve = device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
  const readback = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

  async function timeShader(code) {
    const module = device.createShaderModule({ code });
    const messages = (await module.getCompilationInfo()).messages.filter((m) => m.type === 'error');
    if (messages.length) throw new Error(messages.map((m) => m.message).join('\n'));
    const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } });
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: source } },
        { binding: 1, resource: { buffer: sink } },
        { binding: 2, resource: { buffer: params } }
      ]
    });
    let best = Infinity;
    for (let run = 0; run < 4; run++) {
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass({
        timestampWrites: { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 }
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(workgroups);
      pass.end();
      encoder.resolveQuerySet(querySet, 0, 2, resolve, 0);
      encoder.copyBufferToBuffer(resolve, 0, readback, 0, 16);
      device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const [begin, end] = new BigUint64Array(readback.getMappedRange());
      readback.unmap();
      if (run > 0) best = Math.min(best, Number(end - begin) / 1e6);
    }
    return best;
  }

  const bindings = (sourceType, sinkType) => `
@group(0) @binding(0) var<storage, read> source: array<${sourceType}>;
@group(0) @binding(1) var<storage, read_write> sink: array<${sinkType}>;
@group(0) @binding(2) var<uniform> params: vec4<u32>;`;

  for (const config of configs) {
    if (!['f16', 'f32'].includes(config.componentType)) continue;
    const { M, N, K } = config;
    const left = `subgroup_matrix_left<${config.componentType}, ${K}, ${M}>`;
    const right = `subgroup_matrix_right<${config.componentType}, ${N}, ${K}>`;
    const result = `subgroup_matrix_result<${config.resultComponentType}, ${N}, ${M}>`;
    for (const chains of [1, 2, 4, 8]) {
      const declare = Array.from({ length: chains }, (_, i) => `var c${i} = ${result}();`).join('\n  ');
      const step = Array.from({ length: chains }, (_, i) =>
        `c${i} = subgroupMatrixMultiplyAccumulate(a, b, c${i});`).join('\n    ');
      const store = Array.from({ length: chains }, (_, i) =>
        `subgroupMatrixStore<row_major>(&sink, (base + ${i}u) * ${M * N}u, c${i}, ${N}u);`).join('\n  ');
      const code = `enable f16;
enable chromium_experimental_subgroup_matrix;
${bindings(config.componentType, config.resultComponentType)}
@compute @workgroup_size(${workgroupSize})
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_id) sg: u32) {
  let a = subgroupMatrixLoad<${left}, row_major>(&source, 0u, ${K}u);
  let b = subgroupMatrixLoad<${right}, row_major>(&source, 256u, ${N}u);
  ${declare}
  for (var i = 0u; i < params.x; i++) {
    ${step}
  }
  let base = ((wg.x * 4u + sg) * ${chains}u) % ${Math.floor((1 << 18) / (M * N * chains)) * chains}u;
  ${store}
}`;
      try {
        const ms = await timeShader(code);
        const flops = 2 * M * N * K * chains * iterations * 4 * workgroups;
        report.peaks.push({ kind: 'mma', ...config, chains, ms, tflops: flops / ms / 1e9 });
      } catch (error) {
        report.peaks.push({ kind: 'mma', ...config, chains, error: String(error.message ?? error) });
      }
    }
  }

  for (const type of device.features.has('shader-f16') ? ['f16', 'f32'] : ['f32']) {
    for (const chains of [4, 8]) {
      const declare = Array.from({ length: chains }, (_, i) =>
        `var c${i} = vec4<${type}>(${type}(${i}) * ${type}(0.001));`).join('\n  ');
      const step = Array.from({ length: chains }, (_, i) => `c${i} = fma(c${i}, a, b + c${(i + 1) % chains});`).join('\n    ');
      const sum = Array.from({ length: chains }, (_, i) => `c${i}`).join(' + ');
      const code = `enable f16;
${bindings(`vec4<${type}>`, `vec4<${type}>`)}
@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let a = source[id.x % 1024u];
  let b = source[(id.x + 7u) % 1024u];
  ${declare}
  for (var i = 0u; i < params.x; i++) {
    ${step}
  }
  sink[id.x % 65536u] = ${sum};
}`;
      try {
        const ms = await timeShader(code);
        // Per vec4 lane step: one FMA (2 flops) plus the add that links the chains.
        const flops = 12 * chains * iterations * workgroupSize * workgroups;
        report.peaks.push({ kind: 'fma', componentType: type, chains, ms, tflops: flops / ms / 1e9 });
      } catch (error) {
        report.peaks.push({ kind: 'fma', componentType: type, chains, error: String(error.message ?? error) });
      }
    }
  }
  return report;
}

const chrome = option('--chrome') ?? defaultChrome();
if (!chrome) throw new Error('Chrome/Chromium not found; pass --chrome or CHROME_PATH');
const server = createServer((request, response) => {
  response.writeHead(200, { 'content-type': 'text/html' });
  response.end('<!doctype html><meta charset="utf-8"><link rel="icon" href="data:,"><title>subgroup matrix</title>');
}).listen(0, '127.0.0.1');
await new Promise((resolve) => server.once('listening', resolve));
const browser = await chromium.launch({ executablePath: chrome, headless: true, args: ['--enable-unsafe-webgpu'] });
try {
  const page = await browser.newPage();
  await page.goto(`http://127.0.0.1:${server.address().port}/`);
  const report = { browser: await browser.version(), ...(await page.evaluate(pageMain)) };
  console.log(`${report.browser}  ${report.vendor}/${report.architecture}  subgroup ${report.subgroupMinSize}-${report.subgroupMaxSize}  workgroup storage ${report.maxComputeWorkgroupStorageSize}`);
  console.log(`features: ${report.features.join(', ')}`);
  for (const c of report.configs) console.log(`config: ${c.componentType} -> ${c.resultComponentType} ${c.M}x${c.N}x${c.K}`);
  for (const p of report.peaks) {
    const name = p.kind === 'mma'
      ? `mma ${p.componentType}->${p.resultComponentType} ${p.M}x${p.N}x${p.K}`
      : `fma vec4<${p.componentType}>`;
    console.log(`${name.padEnd(28)} chains ${p.chains}: ${p.error ? `error: ${p.error}` : `${p.tflops.toFixed(2)} TFLOPS (${p.ms.toFixed(2)} ms)`}`);
  }
  const out = option('--json');
  if (out) await writeFile(out, `${JSON.stringify(report, null, 2)}\n`);
} finally {
  await browser.close();
  server.close();
}
