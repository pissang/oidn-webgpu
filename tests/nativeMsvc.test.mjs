import test from 'node:test';
import assert from 'node:assert/strict';
import { msvcConstructors } from '../scripts/native-core-msvc.mjs';

function object(name, big = false) {
  const header = big ? 56 : 20;
  const bytes = Buffer.alloc(header + 40 + 8);
  if (big) {
    bytes.writeUInt16LE(0xffff, 2);
    bytes.writeUInt16LE(2, 4);
    bytes.writeUInt16LE(0x8664, 6);
    bytes.writeUInt32LE(1, 44);
  } else {
    bytes.writeUInt16LE(0x8664, 0);
    bytes.writeUInt16LE(1, 2);
  }
  bytes.write(name, header, 8, 'ascii');
  bytes.writeUInt32LE(8, header + 16);
  bytes.writeUInt32LE(header + 40, header + 20);
  bytes.writeUInt32LE(0xc0400040, header + 36);
  bytes.writeBigUInt64LE(0x1234567890abcdefn, header + 40);
  return bytes;
}

function archive(members) {
  return Buffer.concat([Buffer.from('!<arch>\n'), ...members.flatMap(([name, bytes]) => {
    const header = Buffer.from(name.padEnd(16) + '0'.padEnd(12) + '0'.padEnd(6) +
      '0'.padEnd(6) + '100644'.padEnd(8) + String(bytes.length).padEnd(10) + '`\n');
    return [header, bytes, ...(bytes.length % 2 ? [Buffer.from('\n')] : [])];
  })]);
}

for (const big of [false, true]) {
  test(`MSVC registers GNU constructors as read-only CRT initializers (${big ? 'BigObj' : 'COFF'})`, () => {
    const original = archive([['core.obj/', object('.ctors', big)]]);
    const before = Buffer.from(original);
    const result = msvcConstructors(original);
    const section = 8 + 60 + (big ? 56 : 20);
    assert.equal(result.subarray(section, section + 8).toString(), '.CRT$XCU');
    assert.equal(result.readUInt32LE(section + 36), 0x40400040);
    // Preserve archive/symbol offsets, constructor pointers and relocations.
    const expected = Buffer.from(original);
    expected.write('.CRT$XCU', section, 8);
    expected.writeUInt32LE(0x40400040, section + 36);
    assert.deepEqual(result, expected);
    assert.deepEqual(original, before);
    assert.deepEqual(msvcConstructors(result), result);
  });
}

test('MSVC archive adaptation preserves linker tables, other sections and odd-length members', () => {
  const original = archive([
    ['/', Buffer.from([0x64, 0x86, 1])],
    ['//', Buffer.from('long object name.obj/\n')],
    ['core.obj/', object('.rdata')]
  ]);
  assert.deepEqual(msvcConstructors(original), original);
});

test('MSVC archive adaptation rejects truncated headers, bodies and section tables', () => {
  assert.throws(() => msvcConstructors(Buffer.from('not an archive')), /invalid core archive/);
  const valid = archive([['core.obj/', object('.ctors')]]);
  assert.throws(() => msvcConstructors(valid.subarray(0, 20)), /truncated archive header/);
  assert.throws(() => msvcConstructors(valid.subarray(0, valid.length - 1)), /truncated archive member/);
  const corrupt = Buffer.from(valid);
  corrupt.writeUInt16LE(100, 8 + 60 + 2);
  assert.throws(() => msvcConstructors(corrupt), /truncated COFF section table/);
});
