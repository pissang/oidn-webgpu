// Prints the path of the scriptc compiler installed by
//   npm ci --prefix native/toolchain --include=optional
import { createRequire } from 'node:module';
import { dirname, resolve } from 'node:path';
import { chmodSync, statSync } from 'node:fs';

const require = createRequire(import.meta.url);
const target = process.platform === 'darwin' && ['arm64', 'x64'].includes(process.arch)
  ? `darwin-${process.arch}`
  : process.platform === 'linux' && process.arch === 'x64'
    ? 'linux-x64-gnu'
    : process.platform === 'win32' && process.arch === 'x64'
      ? 'win32-x64-msvc'
      : null;
if (!target) throw new Error(`Unsupported scriptc platform: ${process.platform}/${process.arch}`);
const root = dirname(require.resolve(`@scriptc/cli-${target}/package.json`));
const binary = resolve(root, `dist/bin/scriptc${process.platform === 'win32' ? '.exe' : ''}`);
// The platform archives carry the executables as data; mark them executable.
if (process.platform !== 'win32') {
  for (const path of [
    binary,
    resolve(root, 'dist/lib/typescript/lib/tsc'),
    resolve(root, 'dist/lib/llvm/bin/scriptc-llvm-codegen'),
    resolve(root, 'dist/lib/scriptc-comptime')
  ]) {
    chmodSync(path, statSync(path).mode | 0o111);
  }
}
process.stdout.write(binary);
