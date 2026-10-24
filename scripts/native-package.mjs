// Package identities come from the actual checkout and compiler host, never
// from GITHUB_SHA (which may identify a different PR head or merge commit).
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { cpSync, mkdirSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';

export const targets = {
  'aarch64-apple-darwin': { platform: 'macos', architecture: 'aarch64', nodePlatform: 'darwin', nodeArch: 'arm64', abi: 'darwin' },
  'x86_64-apple-darwin': { platform: 'macos', architecture: 'x86_64', nodePlatform: 'darwin', nodeArch: 'x64', abi: 'darwin' },
  'x86_64-pc-windows-msvc': { platform: 'windows', architecture: 'x86_64', nodePlatform: 'win32', nodeArch: 'x64', abi: 'msvc' },
  'x86_64-unknown-linux-gnu': { platform: 'linux', architecture: 'x86_64', nodePlatform: 'linux', nodeArch: 'x64', abi: 'gnu' }
};

const capture = (root, command, args) => execFileSync(command, args, { cwd: root, encoding: 'utf8' }).trim();
export const sha256 = (file) => createHash('sha256').update(readFileSync(file)).digest('hex');

export function packageContext(root, { target, requireClean = false } = {}) {
  const rustc = capture(root, 'rustc', ['--version', '--verbose']);
  const host = /^host: (.+)$/m.exec(rustc)?.[1];
  target ??= host;
  const info = targets[target];
  if (!info || host !== target || process.platform !== info.nodePlatform || process.arch !== info.nodeArch) {
    throw new Error(`Native package requires a matching Node/Rust host and supported target; Node ${process.platform}/${process.arch}, Rust ${host}, target ${target}`);
  }
  const commit = capture(root, 'git', ['rev-parse', 'HEAD']);
  if (!/^[0-9a-f]{40}$/.test(commit)) throw new Error('Expected a full source commit SHA');
  // Include untracked source files; ignored build outputs do not make a package dirty.
  const dirty = capture(root, 'git', ['status', '--porcelain', '--untracked-files=normal']) !== '';
  if (requireClean && dirty) throw new Error('Refusing to label a CI package from a dirty source checkout');
  const cargo = JSON.parse(capture(root, 'cargo', ['metadata', '--no-deps', '--format-version', '1', '--manifest-path', 'native/Cargo.toml']));
  const crate = cargo.packages.find((entry) => entry.name === 'oidn-webgpu-native');
  const version = crate.version;
  const name = `oidn-webgpu-native-${version}-${target}-${commit}${dirty ? '-dirty' : ''}`;
  return {
    name, target, platform: info.platform, architecture: info.architecture, abi: info.abi,
    crate_version: version,
    wgpu_requirement: crate.dependencies.find((entry) => entry.name === 'wgpu').req,
    source: { commit, dirty },
    build: {
      workflow_ref: process.env.GITHUB_REF ?? null,
      rustc, node: process.versions.node,
      scriptc: process.env.OIDN_WEBGPU_SCRIPTC ? 'custom override' : '0.3.2',
      os_release: os.release(), os_version: os.version(),
      macos_version: process.platform === 'darwin' ? capture(root, 'sw_vers', ['-productVersion']) : null,
      glibc: process.platform === 'linux' ? process.report.getReport().header.glibcVersionRuntime ?? null : null,
      runner_image: process.env.ImageOS ?? null,
      runner_image_version: process.env.ImageVersion ?? null
    },
    cargo_target_directory: cargo.target_directory
  };
}

function filesIn(directory, prefix = '') {
  return readdirSync(path.join(directory, prefix), { withFileTypes: true }).flatMap((entry) => {
    const relative = prefix ? `${prefix}/${entry.name}` : entry.name;
    return entry.isDirectory() ? filesIn(directory, relative) : [relative];
  }).sort();
}

export function writeManifest(directory, context) {
  const { cargo_target_directory, ...metadata } = context;
  const files = filesIn(directory).filter((file) => file !== 'manifest.json' && file !== 'SHA256SUMS');
  const manifest = {
    schema_version: 1, ...metadata,
    files: files.map((file) => ({ path: file, bytes: statSync(path.join(directory, file)).size, sha256: sha256(path.join(directory, file)) }))
  };
  writeFileSync(path.join(directory, 'manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`);
  const checksummed = [...files, 'manifest.json'].sort();
  writeFileSync(path.join(directory, 'SHA256SUMS'), checksummed.map((file) => `${sha256(path.join(directory, file))}  ${file}\n`).join(''));
  return manifest;
}

export function archivePackage(directory, artifactsDirectory, name) {
  mkdirSync(artifactsDirectory, { recursive: true });
  const staged = path.join(artifactsDirectory, name);
  const archive = path.join(artifactsDirectory, `${name}.tar.gz`);
  cpSync(directory, staged, { recursive: true });
  try {
    // Relative archive names also work with GNU tar on Windows, where a drive
    // letter in an absolute archive path could be interpreted as a remote host.
    execFileSync('tar', ['-czf', path.basename(archive), name], { cwd: artifactsDirectory, stdio: 'inherit' });
  } finally {
    rmSync(staged, { recursive: true, force: true });
  }
  const checksum = `${archive}.sha256`;
  writeFileSync(checksum, `${sha256(archive)}  ${path.basename(archive)}\n`);
  return { archive, checksum };
}
