import assert from 'node:assert/strict';
import {test} from 'node:test';
import {createServer} from 'node:http';
import {createHash} from 'node:crypto';
import {mkdtempSync,writeFileSync,rmSync} from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {uploadRelease} from '../scripts/native-release-upload.mjs';

async function fixture(t) {
  const assets=mkdtempSync(path.join(os.tmpdir(),'oidn-release-test-'));
  writeFileSync(path.join(assets,'a.lib'),'first core');writeFileSync(path.join(assets,'b.a'),'second core');
  const manifest={release_tag:'native-v0.1.0',source_commit:'a'.repeat(40),crate_version:'0.1.0'};
  const state={release:null,tag:null,uploads:0,published:0,failUpload:false};
  const server=createServer(async(req,res)=>{
    const chunks=[];for await(const chunk of req)chunks.push(chunk);
    const body=Buffer.concat(chunks),url=new URL(req.url,'http://localhost');
    const json=(status,value)=>{res.writeHead(status,{'Content-Type':'application/json'});res.end(JSON.stringify(value));};
    if(req.method==='GET'&&url.pathname.endsWith('/releases/tags/native-v0.1.0'))return json(state.release?200:404,state.release);
    if(req.method==='GET'&&url.pathname.includes('/commits/')) {
      const ref=url.pathname.split('/').at(-1);const sha=ref===manifest.source_commit?ref:state.tag;
      return json(sha?200:404,sha?{sha}:{});
    }
    if(req.method==='POST'&&url.pathname.endsWith('/releases')) {
      const value=JSON.parse(body);assert.equal(value.target_commitish,manifest.source_commit);assert.equal(value.draft,true);
      state.release={...value,id:1,assets:[],upload_url:`http://127.0.0.1:${server.address().port}/upload{?name,label}`};return json(201,state.release);
    }
    if(req.method==='POST'&&url.pathname==='/upload') {
      if(state.failUpload){state.failUpload=false;return json(503,{});}
      state.uploads++;state.release.assets.push({name:url.searchParams.get('name'),digest:'sha256:'+createHash('sha256').update(body).digest('hex')});return json(201,{});
    }
    if(req.method==='PATCH'&&url.pathname.endsWith('/releases/1')) {
      state.published++;state.release.draft=false;state.tag=manifest.source_commit;return json(200,state.release);
    }
    json(500,{unexpected:req.url});
  }).listen(0,'127.0.0.1');
  await new Promise(resolve=>server.once('listening',resolve));
  t.after(async()=>{server.closeAllConnections();await new Promise(resolve=>server.close(resolve));assert.equal(path.dirname(assets),os.tmpdir());rmSync(assets,{recursive:true,force:true});});
  return {state,assets,manifest,run:()=>uploadRelease({assets,manifest,token:'test-token',apiBase:`http://127.0.0.1:${server.address().port}`})};
}
test('native release uploads to a draft and retries without replacing published files',async t=>{
  const f=await fixture(t);await f.run();assert.equal(f.state.uploads,2);assert.equal(f.state.published,1);await f.run();assert.equal(f.state.uploads,2);assert.equal(f.state.published,1);
});
test('native release rejects changed content in an existing asset',async t=>{
  const f=await fixture(t);await f.run();writeFileSync(path.join(f.assets,'a.lib'),'modified');await assert.rejects(f.run(),/refusing to replace/);assert.equal(f.state.uploads,2);
});
test('native release rejects a tag pointing to another source revision',async t=>{
  const f=await fixture(t);f.state.tag='b'.repeat(40);await assert.rejects(f.run(),/another commit/);assert.equal(f.state.uploads,0);
});
test('native release never adds files to an already public release',async t=>{
  const f=await fixture(t);await f.run();writeFileSync(path.join(f.assets,'new.a'),'new');await assert.rejects(f.run(),/refusing to add files/);assert.equal(f.state.uploads,2);
});
test('native release resumes an interrupted draft without publishing incomplete assets',async t=>{
  const f=await fixture(t);f.state.failUpload=true;await assert.rejects(f.run(),/HTTP 503/);assert.equal(f.state.published,0);assert.equal(f.state.release.draft,true);await f.run();assert.equal(f.state.uploads,2);assert.equal(f.state.published,1);
});
test('native release refuses an unrelated existing draft commit',async t=>{
  const f=await fixture(t);f.state.release={id:1,draft:true,assets:[],target_commitish:'other-branch'};await assert.rejects(f.run(),/draft release points/);assert.equal(f.state.uploads,0);
});
