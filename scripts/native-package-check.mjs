// Verify the archive a developer will download, including relocation, C ABI,
// planner execution and a standalone Rust consumer. No GPU is required.
import assert from 'node:assert/strict';
import { execFileSync, spawnSync } from 'node:child_process';
import { copyFileSync, existsSync, mkdtempSync, mkdirSync, readFileSync, renameSync, rmSync, statSync, writeFileSync } from 'node:fs';
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
  // The configured package must work after extraction into a different path,
  // including spaces, with no access to its original build-time prefix.
  const directory = path.join(work, 'relocated package');
  renameSync(path.join(work, name), directory);
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
  // DLL lookup finds the extracted DLL, not the copy in the Cargo target dir.
  const dllEnv = windows ? { PATH: `${lib}${path.delimiter}${process.env.PATH ?? ''}` } : {};
  run(executable('abi'), [], dllEnv);
  const weights = path.join(root, 'weights/rt_hdr_alb_nrm_small.tza');

  const cmakeConsumer = path.join(work, 'cmake-consumer');
  const cmakeBuild = path.join(work, 'cmake-build');
  const cmakePath = (value) => JSON.stringify(value.replaceAll('\\', '/'));
  mkdirSync(cmakeConsumer);
  writeFileSync(path.join(cmakeConsumer, 'CMakeLists.txt'), `
cmake_minimum_required(VERSION 3.20)
project(package_consumer LANGUAGES C)
set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreadedDLL)
find_package(oidn_webgpu 999 CONFIG QUIET)
if(oidn_webgpu_FOUND)
  message(FATAL_ERROR "An incompatible package version was accepted")
endif()
find_package(oidn_webgpu ${manifest.crate_version} EXACT CONFIG REQUIRED)
add_subdirectory(${cmakePath(path.join(directory, 'examples'))} examples)
add_executable(cmake_abi ${cmakePath(path.join(root, 'native/tests/abi.c'))})
target_link_libraries(cmake_abi PRIVATE oidn_webgpu::oidn_webgpu)
if(WIN32)
  add_custom_command(TARGET cmake_abi POST_BUILD
    COMMAND "\${CMAKE_COMMAND}" -E copy_if_different
      "$<TARGET_FILE:oidn_webgpu::oidn_webgpu>" "$<TARGET_FILE_DIR:cmake_abi>"
    VERBATIM)
endif()
file(GENERATE OUTPUT "\${CMAKE_BINARY_DIR}/executables-$<CONFIG>.txt"
  CONTENT "$<TARGET_FILE:cmake_abi>\n$<TARGET_FILE:denoise>\n$<TARGET_FILE:plan>\n")
`);
  run('cmake', ['-S', cmakeConsumer, '-B', cmakeBuild, `-DCMAKE_PREFIX_PATH=${directory}`, '-DCMAKE_BUILD_TYPE=Release']);
  run('cmake', ['--build', cmakeBuild, '--config', 'Release']);
  const [cmakeAbi, , cmakePlan] = readFileSync(path.join(cmakeBuild, 'executables-Release.txt'), 'utf8').trim().split(/\r?\n/);
  run(cmakeAbi, []);
  run(cmakePlan, [weights, '16', '16']);

  if (windows) {
    // CMake also compiles validate_windows, without requiring a GPU on CI.
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

  const deployed = path.join(work, 'deployed');
  run('cmake', ['--install', cmakeBuild, '--config', 'Release', '--prefix', deployed]);
  // Hide the SDK to prove installed applications find their shipped library.
  renameSync(directory, path.join(work, 'hidden package'));
  run(path.join(deployed, 'bin', windows ? 'plan.exe' : 'plan'), [weights, '16', '16']);
  const usage = spawnSync(path.join(deployed, 'bin', windows ? 'denoise.exe' : 'denoise'), [], { cwd: deployed, encoding: 'utf8' });
  assert.equal(usage.status, 2, usage.error?.message ?? usage.stderr);
  assert.match(usage.stderr, /usage:/);
  if (process.platform === 'darwin') {
    const metalUsage = spawnSync(path.join(deployed, 'bin', 'metal_gpu'), [], { cwd: deployed, encoding: 'utf8' });
    assert.equal(metalUsage.status, 2, metalUsage.error?.message ?? metalUsage.stderr);
    assert.match(metalUsage.stderr, /usage:/);
  }
  console.log('CMake targets, relocated package and installed C examples: OK (no GPU requested)');
  console.log(`Verified ${manifest.name}`);
} finally {
  rmSync(work, { recursive: true, force: true });
}
