import assert from 'node:assert/strict';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import ts from 'typescript';

// Browser callers pass WebGPU objects, not the structural OIDNGPU* types.
test('browser API declarations accept GPUDevice and GPUBuffer', () => {
  const fixture = fileURLToPath(new URL('./types/browserApi.ts', import.meta.url));
  const program = ts.createProgram([fixture], {
    strict: true,
    noEmit: true,
    target: ts.ScriptTarget.ESNext,
    module: ts.ModuleKind.ESNext,
    moduleResolution: ts.ModuleResolutionKind.Node10,
    lib: ['lib.esnext.d.ts', 'lib.dom.d.ts'],
    types: [],
    skipLibCheck: true
  });
  const diagnostics = ts.formatDiagnostics(ts.getPreEmitDiagnostics(program), {
    getCanonicalFileName: (fileName) => fileName,
    getCurrentDirectory: () => process.cwd(),
    getNewLine: () => '\n'
  });
  assert.equal(diagnostics, '');
});
