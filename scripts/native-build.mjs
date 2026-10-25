// Builds the native C package into .native-build/package (or --output DIR):
//   include/oidn_webgpu.h, include/oidn_webgpu_core.h
//   lib/liboidn_webgpu.{dylib,so} or oidn_webgpu.dll (+ .dll.lib import library)
//   lib/liboidn_webgpu_core.a or oidn_webgpu_core.lib (planner only, no GPU)
//   bin/oidn-webgpu-bench, examples/{denoise,plan}.c, examples/host_buffers.rs
//   README.md, LICENSE, manifest.json, SHA256SUMS
// --archive also writes a target/SHA-labelled tarball and archive checksum.
// Requirements: Node 22+, npm, stable Rust, and a C toolchain. The pinned
// scriptc is installed automatically (or use OIDN_WEBGPU_SCRIPTC).
import { execFileSync } from 'node:child_process';
import { appendFileSync, copyFileSync, cpSync, existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { archivePackage, packageContext, writeManifest } from './native-package.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const args = process.argv.slice(2);
const option = (name) => {
  const index = args.indexOf(name);
  if (index === -1) return undefined;
  if (!args[index + 1] || args[index + 1].startsWith('--')) throw new Error(`${name} requires a value`);
  return args[index + 1];
};
const output = path.resolve(option('--output') ?? path.join(root, '.native-build/package'));
const context = packageContext(root, { target: option('--target'), requireClean: args.includes('--require-clean') });
const windows = process.platform === 'win32';
const run = (command, commandArgs, env = {}) =>
  execFileSync(command, commandArgs, { cwd: root, stdio: 'inherit', env: { ...process.env, ...env } });

const core = path.join(root, '.native-build', windows ? 'oidn_webgpu_core.lib' : 'liboidn_webgpu_core.a');
mkdirSync(path.dirname(core), { recursive: true });
run(process.execPath, [path.join(root, 'scripts/native-core-build.mjs'), '--output', core]);
run('cargo', ['build', '--release', '--locked', '--target', context.target, '--manifest-path', 'native/Cargo.toml'], { OIDN_WEBGPU_CORE_ARCHIVE: core });

const release = path.join(context.cargo_target_directory, context.target, 'release');
const libraries = process.platform === 'darwin'
  ? ['liboidn_webgpu.dylib']
  : windows
    ? ['oidn_webgpu.dll', 'oidn_webgpu.dll.lib']
    : ['liboidn_webgpu.so'];
rmSync(output, { recursive: true, force: true });
for (const directory of ['include', 'lib', 'bin', 'examples']) mkdirSync(path.join(output, directory), { recursive: true });
const copy = (source, destination) => {
  if (!existsSync(source)) throw new Error(`Missing build output ${source}`);
  copyFileSync(source, destination);
};
for (const header of ['oidn_webgpu.h', 'oidn_webgpu_core.h']) {
  copy(path.join(root, 'native/include', header), path.join(output, 'include', header));
}
for (const library of libraries) copy(path.join(release, library), path.join(output, 'lib', library));
copy(core, path.join(output, 'lib', path.basename(core)));
const bench = `oidn-webgpu-bench${windows ? '.exe' : ''}`;
copy(path.join(release, bench), path.join(output, 'bin', bench));
copy(path.join(root, 'native/examples/denoise.c'), path.join(output, 'examples/denoise.c'));
copy(path.join(root, 'native/examples/validate_windows.c'), path.join(output, 'examples/validate_windows.c'));
copy(path.join(root, 'native/examples/plan.c'), path.join(output, 'examples/plan.c'));
copy(path.join(root, 'native/examples/host_buffers.rs'), path.join(output, 'examples/host_buffers.rs'));
copy(path.join(root, 'native/examples/validate.rs'), path.join(output, 'examples/validate.rs'));
mkdirSync(path.join(output, 'examples/rust-app'), { recursive: true });
const rustExample = readFileSync(path.join(root, 'native/examples/rust-app/Cargo.toml'), 'utf8');
if (!rustExample.includes('path = "../.."')) throw new Error('Rust example must depend on the native crate');
writeFileSync(path.join(output, 'examples/rust-app/Cargo.toml'), rustExample.replace(
  'path = "../.."', `git = "https://github.com/pissang/oidn-webgpu.git", rev = "${context.source.commit}"`
));
copy(path.join(root, 'native/README.md'), path.join(output, 'README.md'));
copy(path.join(root, 'LICENSE'), path.join(output, 'LICENSE'));
copy(path.join(root, 'native/NOTICE'), path.join(output, 'NOTICE'));
cpSync(path.join(root, 'native/licenses'), path.join(output, 'licenses'), { recursive: true });
if (args.includes('--require-clean')) {
  const afterBuild = packageContext(root, { target: context.target, requireClean: true });
  if (afterBuild.source.commit !== context.source.commit) throw new Error('Source commit changed during the build');
}
writeManifest(output, context);
if (args.includes('--archive')) {
  const artifactsDirectory = path.join(root, '.native-build/artifacts');
  const { archive, checksum } = archivePackage(output, artifactsDirectory, context.name);
  if (process.env.GITHUB_OUTPUT) {
    appendFileSync(process.env.GITHUB_OUTPUT, `artifact_name=${context.name}\narchive_path=${archive}\nchecksum_path=${checksum}\n`);
  }
  console.log(`Native archive: ${archive}`);
}
console.log(`Native package: ${output}`);
