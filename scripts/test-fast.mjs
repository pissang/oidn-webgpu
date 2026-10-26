import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { testBuilds } from './test-inputs.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const cacheFile = path.join(root, '.native-build/test-fast.json');
const requested = process.argv.slice(2);
const tests = requested.length ? requested : readdirSync(path.join(root, 'tests'))
  .filter(name => name.endsWith('.test.mjs')).sort().map(name => `tests/${name}`);
for (const file of tests) {
  if (!file.endsWith('.test.mjs') || !readFileSync(path.resolve(root, file)).length) {
    throw new Error(`Expected a test file: ${file}`);
  }
}
const builds = testBuilds(tests.map(file => path.resolve(root, file)), root);

function fingerprint(paths) {
  const hash = createHash('sha256').update(process.version);
  const visit = file => {
    hash.update(JSON.stringify(path.relative(root, file)));
    let entries;
    try {
      entries = readdirSync(file, { withFileTypes: true });
    } catch (error) {
      if (error.code === 'ENOENT') { hash.update('missing'); return; }
      if (error.code !== 'ENOTDIR') throw error;
      hash.update('file').update(readFileSync(file));
      return;
    }
    hash.update('directory');
    for (const entry of entries.sort((a, b) => a.name.localeCompare(b.name))) {
      visit(path.join(file, entry.name));
    }
  };
  for (const file of paths) visit(path.resolve(root, file));
  return hash.digest('hex');
}

let cache;
try { cache = JSON.parse(readFileSync(cacheFile, 'utf8')); } catch { cache = {}; }
if (!cache || typeof cache !== 'object' || Array.isArray(cache)) cache = {};
mkdirSync(path.dirname(cacheFile), { recursive: true });
const saveCache = () => writeFileSync(cacheFile, JSON.stringify(cache, null, 2) + '\n');
const common = [
  'src', 'tsconfig.json', 'package.json', 'package-lock.json',
  'node_modules/.package-lock.json', 'scripts/test-fast.mjs'
];

function run(label, args) {
  const start = performance.now();
  const result = spawnSync(process.execPath, args, { cwd: root, stdio: 'inherit' });
  console.log(`[test:fast] ${label}: ${((performance.now() - start) / 1000).toFixed(2)}s`);
  if (result.error) throw result.error;
  if (result.status !== 0) process.exit(result.status ?? 1);
}

function build(label, inputs, output, command) {
  if (!builds.has(label)) {
    console.log(`[test:fast] ${label}: not needed`);
    return;
  }
  const input = fingerprint([...common, ...inputs]);
  if (cache[label]?.input === input && cache[label]?.output === fingerprint([output])) {
    console.log(`[test:fast] ${label}: cached`);
    return;
  }
  delete cache[label];
  saveCache();
  if (output === 'lib') rmSync(path.join(root, output), { recursive: true, force: true });
  run(label, command);
  cache[label] = { input, output: fingerprint([output]) };
  saveCache();
}

build('library', ['node_modules/typescript/package.json'], 'lib',
  ['node_modules/typescript/bin/tsc']);
build('bundle', ['vite.config.ts', 'node_modules/vite/package.json'], 'dist',
  ['node_modules/vite/bin/vite.js', 'build']);
run('tests', ['--test', ...tests]);
