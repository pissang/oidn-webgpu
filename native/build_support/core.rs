use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Path, PathBuf},
};

pub fn archive_name(target: &str) -> &'static str {
    if target.ends_with("-msvc") {
        "oidn_webgpu_core.lib"
    } else {
        "liboidn_webgpu_core.a"
    }
}

/// Select and verify the exact target's bundled core before linking it.
pub fn bundled_core(
    root: &Path,
    target: &str,
    version: &str,
    supplied: Option<&Path>,
) -> Result<PathBuf, String> {
    let text = fs::read_to_string(root.join("prebuilt/manifest.json")).map_err(|e| {
        format!(
            "missing bundled core manifest: {e}; prepare the crate with the native release workflow"
        )
    })?;
    let manifest: Value =
        serde_json::from_str(&text).map_err(|e| format!("invalid core manifest: {e}"))?;
    if manifest["schema_version"] != 1
        || manifest["crate_version"] != version
        || manifest["source_commit"].as_str().is_none_or(|s| {
            s.len() != 40
                || !s
                    .bytes()
                    .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
        })
    {
        return Err("core manifest version or source revision mismatch".into());
    }
    let entry = &manifest["targets"][target];
    let file = entry["file"].as_str().ok_or_else(|| {
        if target.contains("-windows-gnu") {
            format!(
                "no bundled core for {target}; build Windows Rust applications for x86_64-pc-windows-msvc; Windows GNU source builds are not validated"
            )
        } else {
            format!(
                "no bundled core for {target}; choose a supported target listed in the crate README; custom source builds for other targets are not validated"
            )
        }
    })?;
    let extension = if target.ends_with("-msvc") {
        "lib"
    } else {
        "a"
    };
    if file != format!("oidn-webgpu-core-{version}-{target}.{extension}") {
        return Err("core asset name does not match the target and version".into());
    }
    let expected = entry["sha256"].as_str().ok_or("missing core checksum")?;
    if expected.len() != 64
        || !expected
            .bytes()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
    {
        return Err("invalid core checksum".into());
    }
    let path = supplied
        .map(Path::to_path_buf)
        .unwrap_or_else(|| root.join("prebuilt").join(file));
    let bytes = fs::read(&path).map_err(|e| format!("cannot read core {}: {e}", path.display()))?;
    if entry["bytes"].as_u64() != Some(bytes.len() as u64)
        || !bytes.starts_with(b"!<arch>\n")
        || format!("{:x}", Sha256::digest(&bytes)) != expected
    {
        return Err(format!(
            "core size, archive signature or SHA-256 mismatch: {}",
            path.display()
        ));
    }
    Ok(path)
}
