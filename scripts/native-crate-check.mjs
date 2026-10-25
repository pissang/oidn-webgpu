// Verify the actual .crate outside the checkout, with no Node/core override.
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {existsSync,mkdtempSync,readFileSync,rmSync} from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {sha256} from './native-package.mjs';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const assets=path.resolve(process.argv[2]??'.native-build/release-assets');
const manifest=JSON.parse(readFileSync(path.join(assets,'core-manifest.json'),'utf8'));
const name=`oidn-webgpu-native-${manifest.crate_version}`,archive=path.join(assets,name+'.crate');
const digest=readFileSync(path.join(assets,'SHA256SUMS'),'utf8').split('\n').find(l=>l.endsWith('  '+name+'.crate'))?.split(' ')[0];
assert.equal(sha256(archive),digest,'crate checksum mismatch');
const host=/^host: (.+)$/m.exec(execFileSync('rustc',['--version','--verbose'],{encoding:'utf8'}))?.[1];
assert(manifest.targets[host],`no release core for host ${host}`);
const work=mkdtempSync(path.join(os.tmpdir(),'oidn-crate-check-'));
const env={...process.env,OIDN_WEBGPU_NODE:path.join(work,'node-must-not-run'),OIDN_WEBGPU_SCRIPTC:path.join(work,'scriptc-must-not-run')};
delete env.OIDN_WEBGPU_CORE_ARCHIVE;
const run=(exe,args,options={})=>execFileSync(exe,args,{cwd:work,env,stdio:'inherit',...options});
try {
  const entries=run('tar',['-tzf',path.basename(archive)],{cwd:assets,encoding:'utf8',stdio:'pipe'}).trim().split(/\r?\n/);
  assert(entries.every(e=>e.startsWith(name+'/')&&!e.split('/').includes('..')&&!e.includes('\\')),'unsafe crate archive');
  run('tar',['-xzf',path.basename(archive),'-C',work],{cwd:assets});
  const directory=path.join(work,name),cargo=path.join(directory,'Cargo.toml');
  assert.deepEqual(JSON.parse(readFileSync(path.join(directory,'prebuilt/manifest.json'),'utf8')),manifest);
  for(const file of ['NOTICE','LICENSE','licenses/scriptc/runtime-LICENSE.txt','licenses/scriptc/v8-number-radix-LICENSE.txt']) {
    assert(existsSync(path.join(directory,file)),`crate must retain ${file}`);
  }
  const metadata=JSON.parse(run('cargo',['metadata','--no-deps','--format-version','1','--manifest-path',cargo],{encoding:'utf8',stdio:'pipe'}));
  assert.equal(metadata.packages.find(p=>p.name==='oidn-webgpu-native').license,'MIT AND Apache-2.0 AND BSD-3-Clause AND BSL-1.0','crate must declare bundled component licenses');
  const target=path.join(root,'.native-build/crate-check-target');
  const flags=['--locked','--manifest-path',cargo,'--target',host,'--target-dir',target];
  run('cargo',['publish','--dry-run',...flags]);
  run('cargo',['build','--offline','--examples',...flags]);
  const exe=path.join(target,host,'debug/examples',process.platform==='win32'?'validate.exe':'validate');
  const model=run(exe,['--inspect-model',path.join(root,'weights/rt_hdr_alb_nrm_small.tza')],{encoding:'utf8',stdio:['ignore','pipe','inherit']});
  assert.deepEqual(JSON.parse(model),{modelInputs:3,gpuRequested:false});
  console.log(`Independent crate publish dry-run and Rust application: OK (${host}; no Node, Zig or GPU required)`);
} finally {
  assert.equal(path.dirname(work),os.tmpdir());
  rmSync(work,{recursive:true,force:true});
}
