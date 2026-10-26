import { readFileSync } from 'node:fs';
import path from 'node:path';
import ts from 'typescript';

export function localImports(file) {
  const source = ts.createSourceFile(file, readFileSync(file, 'utf8'), ts.ScriptTarget.Latest, true);
  const imports = new Set();
  const add = node => {
    if (!node || !ts.isStringLiteralLike(node) || !node.text.startsWith('.')) return;
    const resolved = ts.resolveModuleName(node.text, file, {
      allowJs: true,
      moduleResolution: ts.ModuleResolutionKind.Node10
    }, ts.sys).resolvedModule;
    imports.add(resolved?.resolvedFileName ?? path.resolve(path.dirname(file), node.text));
  };
  const visit = node => {
    if (ts.isImportDeclaration(node) || ts.isExportDeclaration(node)) add(node.moduleSpecifier);
    if (ts.isImportEqualsDeclaration(node) && ts.isExternalModuleReference(node.moduleReference)) {
      add(node.moduleReference.expression);
    }
    if (ts.isCallExpression(node) && (node.expression.kind === ts.SyntaxKind.ImportKeyword ||
        (ts.isIdentifier(node.expression) && node.expression.text === 'require'))) {
      add(node.arguments[0]);
    }
    if (ts.isNewExpression(node) && ts.isIdentifier(node.expression) && node.expression.text === 'URL') {
      const argument = node.arguments?.[0];
      if (argument && ts.isStringLiteralLike(argument) && /\.[cm]?[jt]sx?$/.test(argument.text)) add(argument);
    }
    ts.forEachChild(node, visit);
  };
  visit(source);
  return [...imports].map(file => path.resolve(file));
}

export function walkModuleInputs(entries, stop = () => false) {
  const seen = new Set();
  const visit = file => {
    file = path.resolve(file);
    if (seen.has(file)) return;
    seen.add(file);
    if (!stop(file)) for (const dependency of localImports(file)) visit(dependency);
  };
  for (const file of entries) visit(file);
  return seen;
}

export function testBuilds(entries, root) {
  const builds = new Set();
  walkModuleInputs(entries, file => {
    for (const [directory, build] of [['lib', 'library'], ['dist', 'bundle']]) {
      if (file.startsWith(path.join(root, directory) + path.sep)) {
        builds.add(build);
        return true;
      }
    }
    return false;
  });
  return builds;
}
