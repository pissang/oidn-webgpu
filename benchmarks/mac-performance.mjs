#!/usr/bin/env node
// Sequential, GPU-resident full-image latency across a pinned release, two
// current Chrome kernels, native wgpu and official OIDN Metal. See --help.
import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { existsSync } from 'node:fs';
import { copyFile, mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';
import { benchmarkVariant, defaultChrome, startServer, summarize } from './compare.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const argv = process.argv.slice(2);
const options = { baseline: 'v0.5.0', 'current-ref': 'HEAD', warmup: 3, runs: 7,
  models: 'small,base,large', sizes: '720p,1080p,2k', chrome: defaultChrome(),
  'oidn-library': process.env.OIDN_LIBRARY,
  output: 'benchmarks/results/mac-performance.json' };
if (argv.includes('--help')) {
  console.log(`Usage: node benchmarks/mac-performance.mjs --oidn-library /path/to/libOpenImageDenoise.dylib
  [--baseline v0.5.0] [--current-ref HEAD] [--chrome PATH]
  [--warmup 3] [--runs 7] [--models small,base,large] [--sizes 720p,1080p,2k]
  [--output benchmarks/results/mac-performance.json]

macOS only. Requires Node 22+, Chrome, Rust, cc and an OIDN SDK with Metal.
OIDN_LIBRARY and CHROME_PATH may supply paths. Builds pinned source revisions
in temporary worktrees; other uncommitted runtime edits are excluded. Runs all
GPUs sequentially. 720p = 1280x720; 2k = 2560x1440. --sizes 512 retains the
512x512 regression case. Input/output: GPU-resident RGBA32F, HDR,
albedo + normal, inputScale=1, one image per sync, 3 warmups/7 samples.
Native uses cooperative matrices when supported, runtime checks off (C API
default). OIDN loads the exact same TZA: Small/fast; Base/high; Large/high,
cleanAux=true. JSON retains samples, revisions, hashes, limits and timings.
Writes a Markdown report beside JSON; does not edit the changelog.`);
  process.exit(0);
}
for (let i = 0; i < argv.length; i += 2) {
  const key = argv[i].slice(2);
  if (!argv[i].startsWith('--') || !(key in options) || !argv[i + 1] || argv[i + 1].startsWith('--')) {
    throw new Error(`Unknown option or missing value: ${argv[i]}`);
  }
  options[key] = ['warmup', 'runs'].includes(key) ? Number(argv[i + 1]) : argv[i + 1];
}
if (process.platform !== 'darwin') throw new Error('This suite targets macOS / OIDN Metal');
if (!options['oidn-library'] || !existsSync(options['oidn-library'])) throw new Error('Set --oidn-library or OIDN_LIBRARY');
if (!options.chrome) throw new Error('Set --chrome or CHROME_PATH');
for (const key of ['warmup', 'runs']) if (!Number.isInteger(options[key]) || options[key] < (key === 'runs' ? 1 : 0)) {
  throw new Error(`${key} must be ${key === 'runs' ? 'positive' : 'non-negative'}`);
}
const models = {
  small: { weights: 'rt_hdr_alb_nrm_small.tza', oidn: 'fast' },
  base: { weights: 'rt_hdr_alb_nrm.tza', oidn: 'base' },
  large: { weights: 'rt_hdr_calb_cnrm_large.tza', oidn: 'large' }
};
const sizes = { '512': [512, 512], '720p': [1280, 720], '1080p': [1920, 1080], '2k': [2560, 1440] };
const modelNames = [...new Set(options.models.split(','))];
const sizeNames = [...new Set(options.sizes.split(','))];
if (modelNames.some((name) => !models[name]) || sizeNames.some((name) => !sizes[name])) throw new Error('Invalid model/size');
options.output = path.resolve(root, options.output);
const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');

async function run(command, args, { cwd = root, capture = false, env = process.env } = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { cwd, env, stdio: capture ? ['ignore', 'pipe', 'pipe'] : 'inherit' });
    let stdout = '', stderr = '';
    if (capture) {
      child.stdout.on('data', (chunk) => { stdout += chunk; });
      child.stderr.on('data', (chunk) => { stderr += chunk; });
    }
    child.on('error', reject);
    child.on('close', (code) => code === 0 ? resolve(stdout.trim()) : reject(new Error(`${command} exited ${code}\n${stdout}\n${stderr}`)));
  });
}
const git = (...args) => run('git', args, { capture: true });
const labels = ['oidn-web 0.5.0 (Chrome)', 'oidn-web (Chrome)', 'oidn-web (Chrome subgroup matrix)', 'oidn-web(native wgpu)', 'OIDN Metal'];
const report = { date: new Date().toISOString(), baselineCommit: await git('rev-parse', options.baseline),
  currentCommit: await git('rev-parse', options['current-ref']), benchmarkCommit: await git('rev-parse', 'HEAD'),
  benchmarkScriptSha256: sha256(await readFile(fileURLToPath(import.meta.url))),
  options, environment: { macOS: await run('sw_vers', ['-productVersion'], { capture: true }),
    osBuild: await run('sw_vers', ['-buildVersion'], { capture: true }), architecture: os.arch(),
    memoryBytes: os.totalmem(), cpu: os.cpus()[0]?.model, chrome: await run(options.chrome, ['--version'], { capture: true }),
    node: process.version, rust: await run('rustc', ['--version'], { capture: true }) },
  protocol: { metric: 'wall latency per GPU-resident image; preprocessing + network + output conversion + synchronization',
    excluded: ['device creation', 'model preparation', 'initial uploads', 'output readback'], batch: 1,
    precision: 'FP16 activations/weights, default accumulation', hdr: true, inputScale: 1,
    tilePolicy: 'one full image (height padded to 16 internally where required)', imageFormat: 'RGBA32F, RGB interpreted',
    nativeRuntimeChecks: false, ordinaryChromeUnsafeFlag: false, matrixChromeUnsafeFlag: true },
  models: {}, cases: [] };
for (const name of modelNames) report.models[name] = { ...models[name],
  sha256: sha256(await readFile(path.join(root, 'weights', models[name].weights))) };

function markdown() {
  const lines = ['# Mac denoising performance', '',
    `${report.environment.cpu}; macOS ${report.environment.macOS} (${report.environment.osBuild}); ${report.environment.chrome}; ${report.date}.`, '',
    `Baseline: \`${report.baselineCommit}\`; current: \`${report.currentCommit}\`.`, '',
    `Median ms per image, ${options.warmup} warmups and ${options.runs} measured synchronized executions. GPU-resident RGBA32F HDR color/albedo/normal; inputScale=1. Includes preprocessing and output conversion; excludes uploads, readback and preparation.`, '',
    'Percentages show lower latency relative to 0.5.0: (baseline − measured) / baseline × 100%. P95, preparation timings and every sample remain in the JSON report.', '',
    'All oidn-web paths use FP16. Ordinary Chrome has no unsafe WebGPU flag. Matrix Chrome uses `--enable-unsafe-webgpu`. Native runtime checks are disabled (C API default). OIDN uses the same TZA files: Small/fast, Base/high, Large/high with cleanAux=true.', '',
    `| Model | Input | ${labels.join(' | ')} |`,
    `| --- | --- | ${labels.map(() => '---:').join(' | ')} |`];
  for (const entry of report.cases) {
    const baselineMs = entry.results.find((result) => result.label === labels[0])?.summary?.medianMs;
    const values = labels.map((label, index) => {
      const ms = entry.results.find((result) => result.label === label)?.summary?.medianMs;
      if (ms == null) return '—';
      if (index === 0 || !baselineMs) return `${ms.toFixed(2)} ms`;
      const reduction = (baselineMs - ms) / baselineMs * 100;
      return `${ms.toFixed(2)} ms (${reduction >= 0 ? '↓' : '↑'}${Math.abs(reduction).toFixed(1)}%)`;
    });
    lines.push(`| ${entry.model} | ${entry.size} | ${values.join(' | ')} |`);
  }
  lines.push('', '## Model identity', '', '| Model | Weights | SHA-256 |', '| --- | --- | --- |');
  for (const [name, model] of Object.entries(report.models)) lines.push(`| ${name} | ${model.weights} | ${model.sha256} |`);
  lines.push('', 'These measurements describe this Mac and this workload. They do not measure image quality or imply the same performance on other GPUs. The JSON contains raw samples, selected kernels, workspace sizes and browser diagnostics.', '');
  return lines.join('\n');
}
async function save() {
  await mkdir(path.dirname(options.output), { recursive: true });
  await writeFile(options.output, JSON.stringify(report, null, 2) + '\n');
  await writeFile(options.output.replace(/\.json$/, '') + '.md', markdown());
}

const temporary = await mkdtemp(path.join(os.tmpdir(), 'oidn-mac-performance-'));
const worktrees = [];
let ordinary, matrix, server;
try {
  const bundles = {};
  for (const [name, commit] of [['baseline', report.baselineCommit], ['current', report.currentCommit]]) {
    const checkout = path.join(temporary, name);
    await run('git', ['-c', 'filter.lfs.smudge=', '-c', 'filter.lfs.process=', '-c', 'filter.lfs.required=false',
      '-c', 'core.hooksPath=/dev/null', 'worktree', 'add', '--detach', checkout, commit]);
    worktrees.push(checkout);
    await run('npm', ['ci', '--ignore-scripts'], { cwd: checkout });
    await run('npm', ['run', 'build'], { cwd: checkout });
    bundles[name] = path.join(checkout, 'dist/oidn.js');
    report[`${name}BundleSha256`] = sha256(await readFile(bundles[name]));
  }
  // Build the benchmark against the same pinned native core as the browser.
  const nativeSource = 'native/src/bin/oidn-webgpu-denoise-bench.rs';
  await copyFile(path.join(root, nativeSource), path.join(temporary, 'current', nativeSource));
  const target = path.join(root, 'native/target');
  await run('cargo', ['build', '--release', '--manifest-path', path.join(temporary, 'current/native/Cargo.toml'), '--bin', 'oidn-webgpu-denoise-bench'],
    { env: { ...process.env, CARGO_TARGET_DIR: target } });
  const nativeBinary = path.join(target, 'release/oidn-webgpu-denoise-bench');
  const oidnBinary = path.join(temporary, 'oidn-bench');
  await run('cc', ['-O2', '-Wall', '-Wextra', '-o', oidnBinary, path.join(root, 'benchmarks/oidn-sdk/oidn_bench.c')]);
  report.nativeBinarySha256 = sha256(await readFile(nativeBinary));
  report.nativeBenchmarkSha256 = sha256(await readFile(path.join(root, nativeSource)));
  report.oidnBenchmarkSha256 = sha256(await readFile(path.join(root, 'benchmarks/oidn-sdk/oidn_bench.c')));
  report.oidnLibrarySha256 = sha256(await readFile(options['oidn-library']));
  ordinary = await chromium.launch({ executablePath: options.chrome, headless: true });
  matrix = await chromium.launch({ executablePath: options.chrome, headless: true, args: ['--enable-unsafe-webgpu'] });
  const files = new Map([['/baseline/oidn.js', bundles.baseline], ['/current/oidn.js', bundles.current]]);
  for (const modelName of modelNames) for (const sizeName of sizeNames) {
    const model = models[modelName];
    const [width, height] = sizes[sizeName];
    console.log(`\n=== ${modelName} ${width}x${height} ===`);
    files.set('/weights/model.tza', path.join(root, 'weights', model.weights));
    server = await startServer(files);
    const entry = { model: modelName, size: sizeName, width, height, results: [] };
    for (const [index, bundle, kernel, browser] of [[0, 'baseline', 'implicit-gemm', ordinary],
      [1, 'current', 'implicit-gemm', ordinary], [2, 'current', 'subgroup-matrix', matrix]]) {
      const result = await benchmarkVariant(browser, server.origin,
        { width, height, tileSize: Math.max(width, height), warmup: options.warmup, runs: options.runs },
        { label: labels[index], bundle, baseline: index === 0, nativeOptions: true, precision: 'fp16', kernel });
      if (result.skipped) throw new Error(`${result.label}: ${result.skipped}`);
      if (kernel === 'subgroup-matrix' && !result.runtimeInfo?.kernel?.subgroupMatrix) throw new Error('Matrix path silently fell back');
      if (result.runtimeInfo?.lastExecution?.tileCount !== 1) throw new Error('Browser benchmark must use a single whole-image tile');
      if (result.outputSamples.some((value) => !Number.isFinite(value))) throw new Error('Non-finite browser output');
      entry.results.push(result);
      console.log(`${result.label}: ${result.summary.medianMs.toFixed(2)} ms`);
    }
    await server.close(); server = undefined;
    const native = JSON.parse(await run(nativeBinary, ['--weights', path.join(root, 'weights', model.weights),
      '--width', String(width), '--height', String(height), '--warmup', String(options.warmup), '--runs', String(options.runs)], { capture: true }));
    entry.results.push({ ...native, summary: summarize(native.timesMs) });
    const oidn = JSON.parse(await run(oidnBinary, [options['oidn-library'], 'metal', String(width), String(height),
      model.oidn, '1', String(options.runs), String(options.warmup), path.join(root, 'weights', model.weights)], { capture: true }));
    entry.results.push({ ...oidn, label: labels[4], summary: summarize(oidn.timesMs) });
    console.log(`${native.label}: ${summarize(native.timesMs).medianMs.toFixed(2)} ms; OIDN Metal: ${oidn.medianMs.toFixed(2)} ms`);
    report.cases.push(entry);
    await save();
  }
  report.completed = true;
  await save();
  console.log(`\nSaved ${options.output} and Markdown report`);
} finally {
  await ordinary?.close(); await matrix?.close(); await server?.close();
  for (const checkout of worktrees.reverse()) await run('git', ['worktree', 'remove', '--force', checkout]);
  await rm(temporary, { recursive: true, force: true });
}
