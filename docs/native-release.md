# Native crate and binary releases

The native crate version in `native/Cargo.toml` is independent of the npm version.
The manually triggered [Publish native release workflow](../.github/workflows/native-release.yml)
uses that version and the selected source commit. It defaults to validation only.

## What is distributed

| Distribution | Contents | Lifetime |
| --- | --- | --- |
| Native CI artifacts | Four C/native packages, an assembled crate and release assets | 30 days; for review and testing |
| GitHub Release `native-v<VERSION>` | Native packages, four raw core archives, core manifest, `.crate`, license notices, `SHA256SUMS` | No automatic expiry |
| crates.io `oidn-webgpu-native` | Rust sources, license, examples and all four precompiled cores | Registry version archive |

The crate includes cores for Windows x86_64 MSVC, Linux x86_64 GNU, macOS x86_64
and macOS ARM64. `build.rs` selects the Cargo `TARGET`, verifies the version,
source commit, asset name, size, archive signature and SHA-256, then links it.
Consumers need Rust and their platform linker; they do not need Node, npm, Zig,
or a build-time download from GitHub. Offline builds work when Cargo dependencies
are already cached. The assembled crate currently compresses to about 3.2 MiB;
assembly rejects packages at or above 10,000,000 bytes.

The crate and native packages carry `NOTICE` and the required scriptc runtime
and V8 license texts under `licenses/scriptc/`. The same texts accompany raw
Release cores. See [Licensing](../native/README.md#licensing) for the component
licenses and the crate's SPDX expression. Recheck the bundled runtime code when
updating scriptc or changing the planner's runtime feature usage.

Git/path source checkouts retain automatic scriptc builds for shared-core
development. `OIDN_WEBGPU_CORE_ARCHIVE` remains an explicit override; a release
crate verifies its checksum against the bundled target manifest. Other targets
require a source checkout and a caller-provided matching core.

## Verification before publication

The [Native packages workflow](../.github/workflows/native.yml), also called by
the release workflow, completes these stages for the same source commit:

1. Build four native packages and run their C/Rust interface checks and native
   unit tests. Run the shared JavaScript tests on Linux.
2. Verify package checksums, target identities, source revision and crate version.
   Copy their actual cores into a staged crate and generate `prebuilt/manifest.json`.
3. On all four platforms, unpack the `.crate` outside the checkout, remove the
   core override and disable Node/scriptc. Run `cargo publish --dry-run`, build
   the Rust examples offline, and run the model inspection mode without a GPU.

No GPU is available on hosted runners. GPU correctness and performance still
require the hardware checks described in the [development guide](development.md#benchmarks-and-gpu-checks).

To reproduce assembly locally, check out the workflow run's exact commit and
download its four native artifacts into a fresh directory. On Windows, use an
x64 Visual Studio Native Tools PowerShell with an MSVC Rust toolchain. Commands
run from the repository root:

```shell
gh run download RUN_ID --pattern 'oidn-webgpu-native-*' --dir .native-build/release-input-RUN_ID
node scripts/native-release.mjs collect --input .native-build/release-input-RUN_ID
node scripts/native-crate-check.mjs
```

This prepares `.native-build/release-assets` and `.native-build/crate-stage`.
It does not publish. Raw `cargo publish` from `native/` is not the release path:
the source checkout has no bundled cores, and its packaged verification fails
with a message directing the maintainer to this workflow.

## Manual release

1. Set the native version, update the changelog, commit the sources, and run GPU
   checks when changing kernels. Never reuse a published native version for a
   different source commit.
2. Run **Publish native release** from GitHub Actions. Select the intended branch
   or tag, enter the exact native crate version, and leave `publish` unchecked.
   All build and isolated-crate checks must pass.
3. Configure the `crates-io` GitHub environment and registry authentication below.
4. Run the workflow for the same source commit and version with `publish` checked.
   It reruns verification, authenticates, publishes the permanent native assets,
   and invokes `cargo publish` on the assembled crate.

Only `workflow_dispatch` can publish. Pushes, tags and pull requests run checks
without registry credentials or Release write permissions. The workflow file
must reach the default branch before GitHub exposes its manual Run workflow UI.

The publication job uses the `crates-io` environment, `contents: write` for
Release assets and `id-token: write` for Trusted Publishing. Environment approval
rules may be configured in repository settings when desired.

### Registry authentication

For the first publication, configure a crates.io API token as the
`CARGO_REGISTRY_TOKEN` secret in the `crates-io` environment. It must permit
publishing the new crate. The workflow never prints this token.

After the crate exists, configure a crates.io Trusted Publisher for:

| Setting | Value |
| --- | --- |
| GitHub owner | `pissang` |
| Repository | `oidn-webgpu` |
| Workflow | `native-release.yml` |
| Environment | `crates-io` |

Remove the bootstrap secret after Trusted Publishing is configured. With no
secret set, `rust-lang/crates-io-auth-action` obtains a short-lived token via OIDC
and revokes it on job completion. See the [official action](https://github.com/rust-lang/crates-io-auth-action).

### Retry behavior

Publication starts with a draft Release and publishes it after every asset is
uploaded. A retry reuses matching SHA-256 assets in an existing draft or published
Release. It refuses a tag pointing to another commit, differing existing asset
digests, or additions to an already published Release. It never uses an asset
overwrite operation. If registry authentication fails after assets are published,
fix authentication and rerun the same source version; the matching assets remain.

Crates.io versions cannot be overwritten. If the upload response is interrupted,
check crates.io before retrying; an already uploaded version must not be published
again. A new source change requires a new native version.
