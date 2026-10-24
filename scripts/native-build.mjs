// Builds the native C package into .native-build/package (or --output DIR):
//   include/oidn_web.h, include/oidn_web_core.h
//   lib/liboidn_web.{dylib,so} or oidn_web.dll (+ .dll.lib import library)
//   lib/liboidn_web_core.a or oidn_web_core.lib (planner only, no GPU)
//   bin/oidn-web-bench, examples/{denoise,plan}.c, README.md, LICENSE
// Requirements: Node, stable Rust, and scriptc from
//   npm ci --prefix native/toolchain --include=optional
// (or OIDN_WEB_SCRIPTC=/path/to/scriptc).
import { execFileSync } from 'node:child_process';
import { copyFileSync, existsSync, mkdirSync, rmSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const args = process.argv.slice(2);
const output = path.resolve(args.includes('--output') ? args[args.indexOf('--output') + 1] : path.join(root, '.native-build/package'));
const windows = process.platform === 'win32';
const run = (command, commandArgs, env = {}) =>
  execFileSync(command, commandArgs, { cwd: root, stdio: 'inherit', env: { ...process.env, ...env } });

const scriptc = process.env.OIDN_WEB_SCRIPTC ??
  execFileSync(process.execPath, [path.join(root, 'native/toolchain/compiler.mjs')], { encoding: 'utf8' });
const core = path.join(root, '.native-build', windows ? 'oidn_web_core.lib' : 'liboidn_web_core.a');
mkdirSync(path.dirname(core), { recursive: true });
run(scriptc, ['build', '--lib', '--profile', 'src/native/core-profile.json', '-o', core]);
run('cargo', ['build', '--release', '--locked', '--manifest-path', 'native/Cargo.toml'], { OIDN_WEB_CORE_ARCHIVE: core });

const release = path.join(root, 'native/target/release');
const libraries = process.platform === 'darwin'
  ? ['liboidn_web.dylib']
  : windows
    ? ['oidn_web.dll', 'oidn_web.dll.lib']
    : ['liboidn_web.so'];
rmSync(output, { recursive: true, force: true });
for (const directory of ['include', 'lib', 'bin', 'examples']) mkdirSync(path.join(output, directory), { recursive: true });
const copy = (source, destination) => {
  if (!existsSync(source)) throw new Error(`Missing build output ${source}`);
  copyFileSync(source, destination);
};
for (const header of ['oidn_web.h', 'oidn_web_core.h']) {
  copy(path.join(root, 'native/include', header), path.join(output, 'include', header));
}
for (const library of libraries) copy(path.join(release, library), path.join(output, 'lib', library));
copy(core, path.join(output, 'lib', path.basename(core)));
const bench = `oidn-web-bench${windows ? '.exe' : ''}`;
copy(path.join(release, bench), path.join(output, 'bin', bench));
copy(path.join(root, 'native/examples/denoise.c'), path.join(output, 'examples/denoise.c'));
copy(path.join(root, 'native/examples/plan.c'), path.join(output, 'examples/plan.c'));
copy(path.join(root, 'native/README.md'), path.join(output, 'README.md'));
copy(path.join(root, 'LICENSE'), path.join(output, 'LICENSE'));
console.log(`Native package: ${output}`);
