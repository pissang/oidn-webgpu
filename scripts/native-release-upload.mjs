import assert from 'node:assert/strict';
import {readFileSync,readdirSync} from 'node:fs';
import path from 'node:path';
import {sha256} from './native-package.mjs';

// Published assets are never replaced. A partial draft can be resumed only
// when every existing file has the same server-reported SHA-256.
export async function uploadRelease({assets,manifest,token,apiBase='https://api.github.com'}) {
  assert(token,'GH_TOKEN is required');
  const tag=manifest.release_tag,commit=manifest.source_commit;
  const api=async(route,options={})=>{
    const response=await fetch(`${apiBase}/repos/pissang/oidn-webgpu/${route}`,{...options,headers:{Accept:'application/vnd.github+json',Authorization:`Bearer ${token}`,'X-GitHub-Api-Version':'2026-03-10',...options.headers}});
    if(response.status===404)return null;
    assert(response.ok,`GitHub API ${route}: HTTP ${response.status}`);return response.json();
  };
  let release=await api(`releases/tags/${encodeURIComponent(tag)}`);
  // A missing commit ref can return 422. The exact Git tag-ref endpoint
  // returns 404 instead; resolve existing tags to commits to support annotations.
  const tagRef=await api(`git/ref/tags/${encodeURIComponent(tag)}`);
  const ref=tagRef?await api(`commits/${encodeURIComponent(`refs/tags/${tag}`)}`):null;
  if(tagRef)assert.equal(ref?.sha,commit,'release tag already refers to another commit');
  if(!release)release=await api('releases',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({tag_name:tag,target_commitish:commit,name:`Native ${manifest.crate_version}`,draft:true,prerelease:manifest.crate_version.includes('-'),body:`Source: ${commit}\n\nIncludes independently buildable Rust crate, four native packages and SHA-256 verified planner cores.`})});
  assert(release,'cannot create release');
  if(!ref) {
    const destination=await api(`commits/${encodeURIComponent(release.target_commitish)}`);
    assert.equal(destination?.sha,commit,'draft release points to another source commit');
  }
  for(const name of readdirSync(assets).sort()) {
    assert(name===path.basename(name)&&!/[\\/]/.test(name),'unsafe asset path');
    const file=path.join(assets,name),digest=sha256(file);
    const old=release.assets.find(a=>a.name===name);
    if(old) {assert.equal(old.digest,`sha256:${digest}`,`refusing to replace release asset ${name}`);continue;}
    assert(release.draft,`refusing to add files to published release ${tag}`);
    const url=release.upload_url.replace(/\{.*$/,'')+'?name='+encodeURIComponent(name);
    const response=await fetch(url,{method:'POST',headers:{Authorization:`Bearer ${token}`,'Content-Type':'application/octet-stream'},body:readFileSync(file)});
    assert(response.ok,`upload ${name}: HTTP ${response.status}`);
  }
  if(release.draft)assert(await api(`releases/${release.id}`,{method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify({draft:false})}),'cannot publish the draft release');
}
