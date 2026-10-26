import assert from 'node:assert/strict';
import {test} from 'node:test';
import {checkPublishedRelease} from '../scripts/native-release-check.mjs';

const version='0.2.0',crateSha256='a'.repeat(64);
function fixture(overrides={}) {
  const content=[
    {version:{num:version,checksum:crateSha256,yanked:false}},
    JSON.stringify({vers:version,cksum:crateSha256,yanked:false}),
    '<h1>Native API</h1>'
  ];
  const calls=[];
  const fetchImpl=async url=>{
    calls.push(url);
    const i=url.includes('/api/')?0:url.includes('index.crates.io')?1:2;
    const value=overrides[i]??content[i];
    if(value instanceof Error)throw value;
    return value instanceof Response?value:new Response(i===0?JSON.stringify(value):value);
  };
  return {calls,content,options:{version,crateSha256,fetchImpl,timeoutMs:0}};
}
test('native publication requires registry metadata, Cargo index and rendered README',async()=>{
  const f=fixture();await checkPublishedRelease(f.options);assert.equal(f.calls.length,3);
});
test('accepted upload with missing index and README is not a completed release',async()=>{
  const f=fixture({1:JSON.stringify({vers:'0.1.1'}),2:new Response('AccessDenied',{status:403})});
  await assert.rejects(checkPublishedRelease(f.options),/Cargo index: 0\.2\.0 is not available; README: HTTP 403/);
});
test('native publication waits for asynchronous index and README processing',async()=>{
  const f=fixture();f.content[1]='';f.content[2]='';
  let retries=0;
  await checkPublishedRelease({...f.options,timeoutMs:500,retryMs:0,onPending:()=>{
    retries++;f.content[1]=JSON.stringify({vers:version,cksum:crateSha256,yanked:false});f.content[2]='<h1>Ready</h1>';
  }});
  assert.equal(retries,1);assert.equal(f.calls.length,6);
});
test('native publication rejects registry and index checksum mismatches without retrying',async()=>{
  for(const overrides of [
    {0:{version:{num:version,checksum:'b'.repeat(64),yanked:false}}},
    {1:JSON.stringify({vers:version,cksum:'b'.repeat(64),yanked:false})}
  ]) {
    const f=fixture(overrides);
    await assert.rejects(checkPublishedRelease({...f.options,timeoutMs:500,retryMs:0}),/checksum differs/);
    assert.equal(f.calls.length,3);
  }
});
test('native publication reports network failure and empty README',async()=>{
  const f=fixture({0:new Error('connection failed'),2:'  '});
  await assert.rejects(checkPublishedRelease(f.options),/version metadata: connection failed; README: empty HTML/);
});
test('native publication rejects yanked versions',async()=>{
  const f=fixture({0:{version:{num:version,checksum:crateSha256,yanked:true}}});
  await assert.rejects(checkPublishedRelease(f.options),/version is yanked/);
});
