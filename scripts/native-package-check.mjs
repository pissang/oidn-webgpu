// Verify the archive a developer will download, including relocation, C ABI,
// planner execution and a standalone Rust consumer. No GPU is required.
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { copyFileSync, existsSync, mkdtempSync, mkdirSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { packageContext, sha256 } from './native-package.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const archive = path.resolve(process.argv[2] ?? '');
if (!archive.endsWith('.tar.gz') || !existsSync(archive)) throw new Error('Usage: node scripts/native-package-check.mjs ARCHIVE.tar.gz');
const checksum = readFileSync(`${archive}.sha256`, 'utf8').trim();
assert.equal(checksum, `${sha256(archive)}  ${path.basename(archive)}`, 'archive checksum mismatch');
const context = packageContext(root);
const work = mkdtempSync(path.join(os.tmpdir(), 'oidn-package-check-'));
const run = (command, args, env = {}) => execFileSync(command, args, { cwd: work, stdio: 'inherit', env: { ...process.env, ...env } });

try {
  const name = path.basename(archive, '.tar.gz');
  const members = execFileSync('tar', ['-tzf', path.basename(archive)], { cwd: path.dirname(archive), encoding: 'utf8' }).trim().split(/\r?\n/);
  assert(members.length > 0 && members.every((member) =>
    member.startsWith(`${name}/`) && !member.split('/').includes('..') && !member.includes('\\')
  ), 'archive must contain one named package directory');
  execFileSync('tar', ['-xzf', path.basename(archive), '-C', work], { cwd: path.dirname(archive), stdio: 'inherit' });
  const directory = path.join(work, name);
  const manifest = JSON.parse(readFileSync(path.join(directory, 'manifest.json'), 'utf8'));
  assert.equal(manifest.schema_version, 1);
  assert.equal(manifest.name, name);
  assert.equal(manifest.target, context.target);
  assert.equal(manifest.platform, context.platform);
  assert.equal(manifest.architecture, context.architecture);
  assert.equal(manifest.abi, context.abi);
  assert.equal(manifest.crate_version, context.crate_version);
  assert.equal(manifest.wgpu_requirement, context.wgpu_requirement);
  assert.equal(manifest.source.commit, context.source.commit);
  if (process.env.GITHUB_ACTIONS === 'true') assert.equal(manifest.source.dirty, false);
  for (const file of manifest.files) {
    assert(!file.path.startsWith('/') && !file.path.split('/').includes('..'));
    assert.equal(sha256(path.join(directory, file.path)), file.sha256, file.path);
    assert.equal(statSync(path.join(directory, file.path)).size, file.bytes, file.path);
  }
  for (const line of readFileSync(path.join(directory, 'SHA256SUMS'), 'utf8').trim().split('\n')) {
    const [, expected, file] = /^([a-f0-9]{64})  (.+)$/.exec(line) ?? [];
    assert(file && !file.startsWith('/') && !file.split('/').includes('..'), 'invalid checksum entry');
    assert.equal(sha256(path.join(directory, file)), expected, file);
  }

  const windows = process.platform === 'win32';
  const lib = path.join(directory, 'lib');
  const include = path.join(directory, 'include');
  const core = path.join(lib, windows ? 'oidn_webgpu_core.lib' : 'liboidn_webgpu_core.a');
  const shared = path.join(lib, windows ? 'oidn_webgpu.dll' : process.platform === 'darwin' ? 'liboidn_webgpu.dylib' : 'liboidn_webgpu.so');
  assert(existsSync(core) && existsSync(shared), 'package must contain both shared library and core archive');
  if (windows) assert(existsSync(path.join(lib, 'oidn_webgpu.dll.lib')), 'missing MSVC import library');
  const executable = (name) => path.join(work, `${name}${windows ? '.exe' : ''}`);
  const compile = (source, output, planner = false) => {
    if (windows) {
      const libraries = planner ? [core, ...['ws2_32', 'bcrypt', 'crypt32', 'iphlpapi', 'userenv', 'ntdll', 'advapi32'].map((name) => `${name}.lib`)] : [path.join(lib, 'oidn_webgpu.dll.lib')];
      run('cl.exe', ['/nologo', '/std:c11', '/O2', '/MD', `/I${include}`, source, `/Fe${output}`, '/link', ...libraries]);
    } else {
      const libraries = planner ? [core, ...(process.platform === 'linux' ? ['-lm', '-ldl', '-lpthread'] : [])] : [`-L${lib}`, '-loidn_webgpu', `-Wl,-rpath,${lib}`];
      run('cc', ['-std=c11', '-O2', '-I', include, source, ...libraries, '-o', output]);
    }
  };
  compile(path.join(root, 'native/tests/abi.c'), executable('abi'));
  compile(path.join(directory, 'examples/denoise.c'), executable('denoise'));
  compile(path.join(directory, 'examples/plan.c'), executable('plan'), true);
  // DLL lookup finds the extracted DLL, not the copy in the Cargo target dir.
  const dllEnv = windows ? { PATH: `${lib}${path.delimiter}${process.env.PATH ?? ''}` } : {};
  run(executable('abi'), [], dllEnv);
  const weights = path.join(root, 'weights/rt_hdr_alb_nrm_small.tza');
  run(executable('plan'), [weights, '16', '16']);

  if (windows) {
    // Compile the GPU validation host, but do not require a GPU on CI.
    compile(path.join(directory, 'examples/validate_windows.c'), executable('validate-windows'));
    compile(path.join(root, 'native/tests/msvc-runtime.c'), executable('msvc-runtime'), true);
    const mask = (1n << 64n) - 1n;
    const max = (1n << 128n) - 1n;
    let seed = 0x123456789abcdefn;
    const random = () => (seed = (seed * 6364136223846793005n + 1442695040888963407n) & mask);
    const pairs = [[0n, 1n], [max, 1n], [max, 3n], [max, max],
      [1n << 127n, 1n << 64n], [max, (1n << 127n) + 1n], [1n, max]];
    for (let bit = 0n; bit < 128n; bit++) {
      const numerator = (random() << 64n) | random();
      const denominator = (((random() << 64n) | random()) & ((1n << (bit + 1n)) - 1n)) || 1n;
      pairs.push([numerator, 1n << bit], [numerator, denominator]);
    }
    const input = pairs.map(([n, d]) => [n & mask, n >> 64n, d & mask, d >> 64n,
      (n / d) & mask, (n / d) >> 64n].map(v => v.toString(16)).join(' ')).join('\n') + '\n';
    execFileSync(executable('msvc-runtime'), [lib], {
      cwd: work, input, timeout: 30_000, stdio: ['pipe', 'inherit', 'inherit']
    });
  }

  const consumer = path.join(work, 'consumer');
  mkdirSync(consumer, { recursive: true });
  const exampleManifest = readFileSync(path.join(directory, 'examples/rust-app/Cargo.toml'), 'utf8');
  const pinnedSource = `git = "https://github.com/pissang/oidn-webgpu.git", rev = "${manifest.source.commit}"`;
  assert(exampleManifest.includes(pinnedSource), 'packaged Rust app must pin the package source commit');
  // CI stays offline, using this exact checkout instead of fetching it again.
  writeFileSync(path.join(consumer, 'Cargo.toml'), exampleManifest.replace(
    pinnedSource, `path = ${JSON.stringify(path.join(root, 'native'))}`
  ));
  copyFileSync(path.join(directory, 'examples/validate.rs'), path.join(work, 'validate.rs'));
  copyFileSync(path.join(root, 'native/Cargo.lock'), path.join(consumer, 'Cargo.lock'));
  const modelOutput = execFileSync('cargo', ['run', '--offline', '--release', '--target', context.target, '--target-dir', context.cargo_target_directory, '--manifest-path', path.join(consumer, 'Cargo.toml'), '--', '--inspect-model', weights], {
    cwd: work, encoding: 'utf8', stdio: ['ignore', 'pipe', 'inherit'],
    env: { ...process.env, OIDN_WEBGPU_CORE_ARCHIVE: core, OIDN_WEBGPU_NODE: path.join(work, 'node-must-not-be-invoked') }
  });
  assert.deepEqual(JSON.parse(modelOutput), { modelInputs: 3, gpuRequested: false });
  console.log('Packaged Rust application linked to core: OK (no GPU requested)');
  console.log(`Verified ${manifest.name}`);
} finally {
  rmSync(work, { recursive: true, force: true });
}
