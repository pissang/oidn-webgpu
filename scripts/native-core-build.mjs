// Shared core build entry for Cargo git/path dependencies and the C package.
// Toolchain installation stays in the caller's build directory, never in a
// Cargo git checkout. npm's lockfile pins the compiler and its integrity.
import { execFileSync } from 'node:child_process';
import { copyFileSync, existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const args = process.argv.slice(2);
const option = (name, fallback) => {
  const index = args.indexOf(name);
  if (index === -1) return fallback;
  if (!args[index + 1] || args[index + 1].startsWith('--')) throw new Error(`${name} requires a path`);
  return path.resolve(args[index + 1]);
};
const output = option('--output', path.join(root, '.native-build', process.platform === 'win32' ? 'oidn_webgpu_core.lib' : 'liboidn_webgpu_core.a'));
const toolchain = option('--toolchain', path.join(path.dirname(output), 'toolchain'));
if (Number(process.versions.node.split('.')[0]) < 22) throw new Error('Native core builds require Node 22+');
mkdirSync(path.dirname(output), { recursive: true });

let compiler = process.env.OIDN_WEBGPU_SCRIPTC;
if (!compiler) {
  mkdirSync(toolchain, { recursive: true });
  let changed = false;
  for (const file of ['compiler.mjs', 'package.json', 'package-lock.json']) {
    const source = path.join(root, 'native/toolchain', file);
    const target = path.join(toolchain, file);
    if (!existsSync(target) || !readFileSync(source).equals(readFileSync(target))) {
      copyFileSync(source, target);
      changed = true;
    }
  }
  const installed = path.join(toolchain, '.installed');
  if (changed || !existsSync(installed) || !existsSync(path.join(toolchain, 'node_modules'))) {
    // Remove the success marker before installation so interrupted/failed
    // installs are retried. Invoke npm via Node on Windows to avoid cmd.exe
    // quoting of build directories containing spaces or shell characters.
    rmSync(installed, { force: true });
    const npmArgs = ['ci', '--include=optional', '--ignore-scripts', '--no-audit', '--no-fund'];
    if (process.platform === 'win32') {
      const npmCommand = execFileSync('where.exe', ['npm.cmd'], { encoding: 'utf8' }).trim().split(/\r?\n/)[0];
      const npmCli = path.join(path.dirname(npmCommand), 'node_modules/npm/bin/npm-cli.js');
      execFileSync(process.execPath, [npmCli, ...npmArgs], { cwd: toolchain, stdio: 'inherit' });
    } else {
      execFileSync('npm', npmArgs, { cwd: toolchain, stdio: 'inherit' });
    }
  }
  compiler = execFileSync(process.execPath, [path.join(toolchain, 'compiler.mjs')], { encoding: 'utf8' }).trim();
  writeFileSync(installed, 'scriptc 0.3.2\n');
}
execFileSync(compiler, ['build', '--lib', '--profile', path.join(root, 'src/native/core-profile.json'), '-o', output], { cwd: root, stdio: 'inherit' });
