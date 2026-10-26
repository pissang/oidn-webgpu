import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import test from 'node:test';
import { testBuilds, walkModuleInputs } from '../scripts/test-inputs.mjs';

const root = fileURLToPath(new URL('../', import.meta.url));

test('native Cargo inputs cover the entry dependency graph without browser modules', () => {
  const profile = JSON.parse(readFileSync(new URL('../src/native/core-profile.json', import.meta.url)));
  const entry = path.resolve(root, 'src/native', profile.entry);
  const actual = [...walkModuleInputs([entry])]
    .map(file => path.relative(root, file).replaceAll('\\', '/'))
    .filter(file => !file.startsWith('src/native/')).sort();
  const declared = JSON.parse(readFileSync(new URL('../src/native/core-inputs.json', import.meta.url)));
  assert.deepEqual(declared, actual, 'Update core-inputs.json when native shared imports change');
  for (const file of declared) assert.ok(file.startsWith('src/') && file.endsWith('.ts'));
  assert.ok(!actual.includes('src/UNet.ts'));
  assert.ok(!actual.includes('src/nativeUNet.ts'));
});

test('local checks build only the artifacts imported by the selected tests', () => {
  const builds = name => [...testBuilds([path.resolve(root, 'tests', name)], root)].sort();
  assert.deepEqual(builds('tilePlanner.test.mjs'), ['library']);
  assert.deepEqual(builds('tileExecute.test.mjs'), ['bundle']);
  assert.deepEqual(builds('types.test.mjs'), ['library']);
  assert.deepEqual(builds('nativeMsvc.test.mjs'), []);
  assert.deepEqual(builds('nativeRelease.test.mjs'), []);
});

test('test dependency discovery follows helpers and missing generated output', t => {
  const directory = mkdtempSync(path.join(os.tmpdir(), 'oidn-test-inputs-'));
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  mkdirSync(path.join(directory, 'tests'));
  writeFileSync(path.join(directory, 'tests/case.mjs'), "import './helper.mjs';\n");
  writeFileSync(path.join(directory, 'tests/helper.mjs'), "export * from '../lib/missing.js';\nimport('../dist/missing.js');\n");
  assert.deepEqual([...testBuilds([path.join(directory, 'tests/case.mjs')], directory)].sort(), ['bundle', 'library']);
});
