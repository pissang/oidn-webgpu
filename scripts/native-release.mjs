// Assemble verified native packages into an independently buildable crate and
// immutable, versioned GitHub Release assets. No publishing on collect/stage.
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {appendFileSync,copyFileSync,cpSync,existsSync,mkdirSync,readFileSync,readdirSync,rmSync,statSync,writeFileSync} from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {archivePackage,sha256,targets,writeManifest} from './native-package.mjs';
import {uploadRelease} from './native-release-upload.mjs';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const args=process.argv.slice(2), command=args[0];
const option=(name,fallback)=>{const i=args.indexOf(name);if(i<0)return fallback;assert(args[i+1]&&!args[i+1].startsWith('--'),`${name} requires a value`);return args[i+1];};
const run=(exe,argv,options={})=>execFileSync(exe,argv,{cwd:root,stdio:'inherit',...options});
const capture=(exe,argv)=>run(exe,argv,{stdio:'pipe',encoding:'utf8'}).trim();
const commit=capture('git',['rev-parse','HEAD']);
const cargo=JSON.parse(capture('cargo',['metadata','--no-deps','--format-version','1','--manifest-path','native/Cargo.toml']));
const version=cargo.packages.find(p=>p.name==='oidn-webgpu-native').version;
const tag=`native-v${version}`;
const assets=path.resolve(option('--assets',path.join(root,'.native-build/release-assets')));
const stage=path.join(root,'.native-build/crate-stage');
const dirty=capture('git',['status','--porcelain','--untracked-files=normal'])!=='';
const allTargets=Object.keys(targets).sort();
const compilerVersions=new Set(Object.values(JSON.parse(readFileSync(path.join(root,'native/toolchain/package.json'),'utf8')).optionalDependencies));
assert.equal(compilerVersions.size,1,'all compiler platforms must use the same version');
const compilerVersion=[...compilerVersions][0];
const safeFile=name=>typeof name==='string'&&name===path.basename(name)&&!/[\\/]/.test(name)&&!['.','..'].includes(name);
function reset(directory) {
  const relative=path.relative(path.join(root,'.native-build'),directory);
  assert(relative&&!relative.startsWith('..')&&!path.isAbsolute(relative),'cleanup must stay inside .native-build');
  rmSync(directory,{recursive:true,force:true});mkdirSync(directory,{recursive:true});
}
function files(directory) {
  return readdirSync(directory,{withFileTypes:true}).flatMap(e=>e.isDirectory()?files(path.join(directory,e.name)):[path.join(directory,e.name)]);
}
function sums() {
  const names=readdirSync(assets).filter(n=>n!=='SHA256SUMS').sort();
  writeFileSync(path.join(assets,'SHA256SUMS'),names.map(n=>`${sha256(path.join(assets,n))}  ${n}\n`).join(''));
}
function checkAssets() {
  for(const line of readFileSync(path.join(assets,'SHA256SUMS'),'utf8').trim().split('\n')) {
    const [,digest,name]=/^([a-f0-9]{64})  (.+)$/.exec(line)??[];
    assert(safeFile(name),'unsafe checksum path');assert.equal(sha256(path.join(assets,name)),digest,name);
  }
  const m=JSON.parse(readFileSync(path.join(assets,'core-manifest.json'),'utf8'));
  assert.equal(m.schema_version,1);assert.equal(m.crate_version,version);assert.equal(m.source_commit,commit);assert.equal(m.release_tag,tag);
  assert.deepEqual(Object.keys(m.targets).sort(),allTargets);
  for(const target of allTargets) {
    const e=m.targets[target], ext=target.endsWith('-msvc')?'lib':'a';
    assert.equal(e.file,`oidn-webgpu-core-${version}-${target}.${ext}`);
    assert.equal(statSync(path.join(assets,e.file)).size,e.bytes);assert.equal(sha256(path.join(assets,e.file)),e.sha256);
    assert(readFileSync(path.join(assets,e.file)).subarray(0,8).equals(Buffer.from('!<arch>\n')));
  }
  return m;
}
function stageCrate() {
  const m=checkAssets();reset(stage);
  for(const file of ['Cargo.toml','Cargo.lock','build.rs','LICENSE','NOTICE','README.md'])copyFileSync(path.join(root,'native',file),path.join(stage,file));
  for(const dir of ['src','include','build_support','licenses'])cpSync(path.join(root,'native',dir),path.join(stage,dir),{recursive:true});
  mkdirSync(path.join(stage,'examples'));
  for(const file of readdirSync(path.join(root,'native/examples')).filter(f=>/\.(rs|c)$/.test(f)))copyFileSync(path.join(root,'native/examples',file),path.join(stage,'examples',file));
  mkdirSync(path.join(stage,'prebuilt'));copyFileSync(path.join(assets,'core-manifest.json'),path.join(stage,'prebuilt/manifest.json'));
  for(const e of Object.values(m.targets))copyFileSync(path.join(assets,e.file),path.join(stage,'prebuilt',e.file));
  // Crates.io renders README links against the repository, not native/.
  const readme=readFileSync(path.join(stage,'README.md'),'utf8').replace(/\]\(((?:include|examples)\/[^)]+)\)/g,`](${'https://github.com/pissang/oidn-webgpu/blob/'}${commit}/native/$1)`);
  writeFileSync(path.join(stage,'README.md'),readme);
}
if(command==='plan') {
  assert(!dirty,'release source must be clean');
  assert.equal(option('--version',version),version,'requested version must match native/Cargo.toml');
  if(process.env.GITHUB_OUTPUT)appendFileSync(process.env.GITHUB_OUTPUT,`version=${version}\ntag=${tag}\ncommit=${commit}\n`);
  console.log(JSON.stringify({version,tag,commit}));
} else if(command==='collect') {
  assert(!dirty||args.includes('--allow-dirty'),'release source must be clean');
  const input=path.resolve(option('--input','.native-build/release-input'));
  const archives=files(input).filter(f=>f.endsWith('.tar.gz'));
  assert.equal(archives.length,allTargets.length,'expected exactly four native packages');
  reset(assets);
  assert(readFileSync(path.join(root,'native/NOTICE'),'utf8').includes(`scriptc ${compilerVersion}`),'update runtime notices when changing the compiler');
  const m={schema_version:1,crate_version:version,release_tag:tag,source_commit:commit,source_dirty:dirty,scriptc:compilerVersion,targets:{}};
  const unpack=path.join(root,'.native-build/release-unpack');reset(unpack);
  for(const archive of archives) {
    const name=path.basename(archive,'.tar.gz');
    assert.equal(readFileSync(archive+'.sha256','utf8').trim(),`${sha256(archive)}  ${path.basename(archive)}`);
    const entries=run('tar',['-tzf',path.basename(archive)],{cwd:path.dirname(archive),stdio:'pipe',encoding:'utf8'}).trim().split(/\r?\n/);
    assert(entries.every(e=>e.startsWith(name+'/')&&!e.split('/').includes('..')&&!e.includes('\\')),'unsafe package archive');
    run('tar',['-xzf',path.basename(archive),'-C',unpack],{cwd:path.dirname(archive)});
    const directory=path.join(unpack,name),p=JSON.parse(readFileSync(path.join(directory,'manifest.json'),'utf8'));
    assert.equal(p.source.commit,commit);assert.equal(p.source.dirty,false);assert.equal(p.crate_version,version);
    assert.equal(p.build.scriptc,compilerVersion,'release cores must use the pinned compiler');
    assert(allTargets.includes(p.target)&&!m.targets[p.target],'duplicate or unsupported target');
    for(const f of p.files) {
      assert(!path.isAbsolute(f.path)&&!f.path.split(/[\\/]/).includes('..'));
      assert.equal(sha256(path.join(directory,f.path)),f.sha256,f.path);assert.equal(statSync(path.join(directory,f.path)).size,f.bytes);
    }
    const core=p.target.endsWith('-msvc')?'oidn_webgpu_core.lib':'liboidn_webgpu_core.a';
    const file=`oidn-webgpu-core-${version}-${p.target}.${p.target.endsWith('-msvc')?'lib':'a'}`;
    copyFileSync(path.join(directory,'lib',core),path.join(assets,file));
    m.targets[p.target]={file,bytes:statSync(path.join(assets,file)).size,sha256:sha256(path.join(assets,file))};
    // Keep CI archives source-labelled; release packages use the version and
    // target, with the full source revision retained in their manifest.
    const releaseName=`oidn-webgpu-native-${version}-${p.target}`;
    writeManifest(directory,{...p,name:releaseName});
    archivePackage(directory,assets,releaseName);
  }
  copyFileSync(path.join(root,'native/LICENSE'),path.join(assets,'LICENSE'));
  copyFileSync(path.join(root,'native/NOTICE'),path.join(assets,'NOTICE'));
  for(const file of readdirSync(path.join(root,'native/licenses/scriptc')))copyFileSync(path.join(root,'native/licenses/scriptc',file),path.join(assets,'scriptc-'+file));
  writeFileSync(path.join(assets,'core-manifest.json'),JSON.stringify(m,null,2)+'\n');sums();stageCrate();
  run('cargo',['package','--locked','--allow-dirty','--no-verify','--manifest-path',path.join(stage,'Cargo.toml')]);
  const crate=`oidn-webgpu-native-${version}.crate`;
  copyFileSync(path.join(stage,'target/package',crate),path.join(assets,crate));
  assert(statSync(path.join(assets,crate)).size<10_000_000,'crate exceeds the 10 MB release limit');sums();
  console.log(`Prepared ${crate}: ${statSync(path.join(assets,crate)).size} bytes`);
} else if(command==='stage') {
  stageCrate();
} else if(command==='upload') {
  assert(!dirty,'release source must be clean');
  const m=checkAssets();assert.equal(m.source_dirty,false,'cannot publish a dirty-source bundle');
  assert.equal(process.env.GITHUB_REPOSITORY,'pissang/oidn-webgpu','publishing is restricted to the canonical repository');
  await uploadRelease({assets,manifest:m,token:process.env.GH_TOKEN});
  console.log(`Published immutable native assets: https://github.com/pissang/oidn-webgpu/releases/tag/${tag}`);
} else {
  throw new Error('Usage: native-release.mjs plan|collect|stage|upload [--input DIR] [--assets DIR]');
}
