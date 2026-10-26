import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import path from 'node:path';
import {setTimeout as sleep} from 'node:timers/promises';
import {pathToFileURL} from 'node:url';

class IntegrityError extends Error {}

export async function checkPublishedRelease({version,crateSha256,fetchImpl=fetch,timeoutMs=300_000,retryMs=10_000,onPending=()=>{}}) {
  assert(/^[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$/.test(version),'invalid native version');
  assert(/^[a-f0-9]{64}$/.test(crateSha256),'invalid crate checksum');
  assert(Number.isFinite(timeoutMs)&&timeoutMs>=0,'timeout must be nonnegative');
  assert(Number.isFinite(retryMs)&&retryMs>=0,'retry interval must be nonnegative');
  const name='oidn-webgpu-native',encoded=encodeURIComponent(version),deadline=Date.now()+timeoutMs;
  const urls=[
    `https://crates.io/api/v1/crates/${name}/${encoded}`,
    `https://index.crates.io/oi/dn/${name}`,
    `https://static.crates.io/readmes/${name}/${name}-${encoded}.html`
  ];
  const labels=['version metadata','Cargo index','README'];
  const integrity=(condition,message)=>{if(!condition)throw new IntegrityError(message);};
  const notReady=message=>new Error(`${message}. Retry this check after registry processing recovers; do not republish the same version.`);
  let lastMessage;
  for(;;) {
    if(lastMessage&&Date.now()>=deadline)throw notReady(lastMessage);
    const checks=await Promise.allSettled(urls.map(async(url,i)=>{
      const response=await fetchImpl(url,{
        headers:{'User-Agent':'oidn-webgpu-native-release-check','Cache-Control':'no-cache'},
        signal:AbortSignal.timeout(Math.max(1,Math.min(10_000,deadline-Date.now())))
      });
      if(!response.ok)throw new Error(`HTTP ${response.status}`);
      return i===0?response.json():response.text();
    }));
    const pending=[];
    for(let i=0;i<checks.length;i++) {
      const check=checks[i];
      if(check.status==='rejected') {pending.push(`${labels[i]}: ${check.reason.message}`);continue;}
      if(i===0) {
        const metadata=check.value.version;
        if(!metadata?.checksum) {pending.push('version metadata: missing checksum');continue;}
        integrity(metadata.num===version,'registry returned another native version');
        integrity(metadata.checksum===crateSha256,'registry crate checksum differs from the release asset');
        integrity(metadata.yanked===false,'published native version is yanked');
      } else if(i===1) {
        let entries;
        try {entries=check.value.trim().split(/\r?\n/).filter(Boolean).map(line=>JSON.parse(line));}
        catch {pending.push('Cargo index: invalid response');continue;}
        const entry=entries.find(entry=>entry.vers===version);
        if(!entry) {pending.push(`Cargo index: ${version} is not available`);continue;}
        integrity(entry.cksum===crateSha256,'Cargo index checksum differs from the release asset');
        integrity(entry.yanked===false,'native version is yanked in the Cargo index');
      } else if(!check.value.trim())pending.push('README: empty HTML');
    }
    if(pending.length===0)return;
    const message=`Native ${version} is not publicly ready: ${pending.join('; ')}`;
    if(Date.now()>=deadline)throw notReady(message);
    lastMessage=message;
    onPending(message);
    await sleep(Math.min(retryMs,Math.max(0,deadline-Date.now())));
  }
}

if(process.argv[1]&&import.meta.url===pathToFileURL(path.resolve(process.argv[1])).href) {
  const assets=path.resolve(process.argv[2]??'.native-build/release-assets');
  const {crate_version:version}=JSON.parse(readFileSync(path.join(assets,'core-manifest.json'),'utf8'));
  assert(typeof version==='string'&&!/[\\/]/.test(version),'invalid native version');
  const crateSha256=createHash('sha256').update(readFileSync(path.join(assets,`oidn-webgpu-native-${version}.crate`))).digest('hex');
  await checkPublishedRelease({version,crateSha256,onPending:message=>console.log(message)});
  console.log(`Native ${version}: public Cargo index, README and release crate checksums are ready`);
}
