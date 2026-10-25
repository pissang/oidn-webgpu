// Adapt the pinned scriptc Windows archive to MSVC without changing its C ABI.
// Its runtime objects use MinGW CRT names and GNU constructor sections, even
// though its program objects use the Windows x64 calling convention.
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import path from 'node:path';

/** Register GNU constructors with the MSVC CRT. Archive offsets stay intact. */
export function msvcConstructors(archive) {
  const bytes = Buffer.from(archive);
  assert.equal(bytes.subarray(0, 8).toString(), '!<arch>\n', 'invalid core archive');
  let member = 8;
  while (member < bytes.length) {
    assert(member + 60 <= bytes.length, 'truncated archive header');
    assert.equal(bytes.subarray(member + 58, member + 60).toString(), '`\n', 'invalid archive member');
    const size = Number(bytes.subarray(member + 48, member + 58).toString().trim());
    assert(Number.isSafeInteger(size) && size >= 0, 'invalid archive member size');
    const object = member + 60;
    const end = object + size;
    assert(end <= bytes.length, 'truncated archive member');
    const memberName = bytes.subarray(member, member + 16).toString().trim();
    let sections = 0;
    let table = 0;
    const linkerMember = ['/', '//', '/SYM64/'].includes(memberName);
    if (!linkerMember && size >= 20 && bytes.readUInt16LE(object) === 0x8664) {
      sections = bytes.readUInt16LE(object + 2);
      table = object + 20 + bytes.readUInt16LE(object + 16);
    } else if (!linkerMember && size >= 56 && bytes.readUInt16LE(object) === 0 &&
        bytes.readUInt16LE(object + 2) === 0xffff && bytes.readUInt16LE(object + 4) >= 2 &&
        bytes.readUInt16LE(object + 6) === 0x8664) {
      sections = bytes.readUInt32LE(object + 44);
      table = object + 56;
    }
    if (table) {
      assert(table + sections * 40 <= end, 'truncated COFF section table');
      for (let index = 0; index < sections; index++) {
        const section = table + index * 40;
        const name = bytes.subarray(section, section + 8).toString('ascii').replace(/\0.*$/, '');
        if (name !== '.ctors') continue;
        bytes.write('.CRT$XCU', section, 8, 'ascii');
        // MSVC's constructor range is read-only. A writable section is kept
        // separate from __xc_a/__xc_z and its constructors still never run.
        bytes.writeUInt32LE((bytes.readUInt32LE(section + 36) & 0x7fffffff) >>> 0, section + 36);
      }
    }
    member = end + size % 2;
    assert(member <= bytes.length, 'missing archive alignment byte');
  }
  return bytes;
}

export function adaptMsvcCore(root, output) {
  root = path.resolve(root);
  output = path.resolve(output);
  const run = (args, options = {}) => execFileSync('zig', args, { stdio: 'inherit', ...options });
  // zig env uses ZON in recent releases and JSON in older releases. Both
  // formats use the same escaped string spelling for these Windows paths.
  const env = run(['env'], { encoding: 'utf8', stdio: 'pipe' });
  const field = (name) => {
    const match = new RegExp(`(?:\\.${name}\\s*=\\s*|"${name}"\\s*:\\s*)("(?:\\\\.|[^"\\\\])*")`).exec(env);
    assert(match, `zig env did not report ${name}`);
    return JSON.parse(match[1]);
  };
  const lib = field('lib_dir');
  const work = mkdtempSync(path.join(path.dirname(output), 'msvc-runtime-'));
  try {
    const compat = path.join(work, 'msvc-compat.obj');
    const dirent = path.join(work, 'mingw-dirent.obj');
    run(['cc', '-target', 'x86_64-windows-gnu', '-O2', '-c',
      path.join(root, 'native/toolchain/msvc-compat.c'), '-o', compat]);
    // dirent.c is the public-domain MinGW implementation shipped with Zig.
    run(['cc', '-target', 'x86_64-windows-gnu', '-O2', '-c',
      path.join(lib, 'libc/mingw/misc/dirent.c'), '-o', dirent]);
    writeFileSync(output, msvcConstructors(readFileSync(output)));
    run(['ar', 'rcs', output, compat, dirent]);
  } finally {
    assert.equal(path.dirname(work), path.dirname(output));
    rmSync(work, { recursive: true, force: true });
  }
}
