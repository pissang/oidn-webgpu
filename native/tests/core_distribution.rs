#[path = "../build_support/core.rs"]
mod core;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let directory = std::env::temp_dir().join(format!(
            "oidn-core-test-{}-{}",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&directory).unwrap();
        fs::create_dir(directory.join("prebuilt")).unwrap();
        let bytes = b"!<arch>\ncore fixture";
        let file = "oidn-webgpu-core-0.1.0-x86_64-pc-windows-msvc.lib";
        fs::write(directory.join("prebuilt").join(file), bytes).unwrap();
        let manifest = json!({"schema_version":1,"crate_version":"0.1.0","source_commit":"a".repeat(40),"targets":{"x86_64-pc-windows-msvc":{"file":file,"bytes":bytes.len(),"sha256":format!("{:x}",Sha256::digest(bytes))}}});
        fs::write(
            directory.join("prebuilt/manifest.json"),
            manifest.to_string(),
        )
        .unwrap();
        Self(directory)
    }
    fn change(&self, edit: impl FnOnce(&mut serde_json::Value)) {
        let file = self.0.join("prebuilt/manifest.json");
        let mut value = serde_json::from_slice(&fs::read(&file).unwrap()).unwrap();
        edit(&mut value);
        fs::write(file, value.to_string()).unwrap();
    }
    fn load(&self) -> Result<PathBuf, String> {
        core::bundled_core(&self.0, "x86_64-pc-windows-msvc", "0.1.0", None)
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        assert_eq!(self.0.parent(), Some(std::env::temp_dir().as_path()));
        fs::remove_dir_all(&self.0).unwrap();
    }
}

#[test]
fn verifies_the_target_core_and_explicit_override() {
    let f = Fixture::new();
    let path = f.load().unwrap();
    assert_eq!(
        core::archive_name("x86_64-pc-windows-msvc"),
        "oidn_webgpu_core.lib"
    );
    assert_eq!(
        core::archive_name("x86_64-unknown-linux-gnu"),
        "liboidn_webgpu_core.a"
    );
    let copy = f.0.join("supplied.lib");
    fs::copy(&path, &copy).unwrap();
    assert_eq!(
        core::bundled_core(&f.0, "x86_64-pc-windows-msvc", "0.1.0", Some(&copy)).unwrap(),
        copy
    );
    fs::write(&copy, b"!<arch>\ntampered").unwrap();
    assert!(core::bundled_core(&f.0, "x86_64-pc-windows-msvc", "0.1.0", Some(&copy)).is_err());
}
#[test]
fn rejects_different_version_target_and_source() {
    let f = Fixture::new();
    assert!(core::bundled_core(&f.0, "x86_64-pc-windows-msvc", "0.2.0", None).is_err());
    assert!(core::bundled_core(&f.0, "aarch64-apple-darwin", "0.1.0", None).is_err());
    f.change(|m| m["source_commit"] = json!("not a commit"));
    assert!(f.load().is_err());
}
#[test]
fn unsupported_target_error_is_not_bypassed_by_an_archive_override() {
    let f = Fixture::new();
    let supplied =
        f.0.join("prebuilt/oidn-webgpu-core-0.1.0-x86_64-pc-windows-msvc.lib");
    let error =
        core::bundled_core(&f.0, "x86_64-pc-windows-gnu", "0.1.0", Some(&supplied)).unwrap_err();
    assert!(error.contains("no bundled core for x86_64-pc-windows-gnu"));
    assert!(error.contains("x86_64-pc-windows-msvc"));
    assert!(error.contains("Windows GNU source builds are not validated"));
    assert!(!error.contains("custom OIDN_WEBGPU_CORE_ARCHIVE"));
}
#[test]
fn rejects_unsafe_asset_names_and_corruption() {
    let f = Fixture::new();
    let path = f.load().unwrap();
    fs::write(path, b"!<arch>\ncorrupted").unwrap();
    assert!(f.load().is_err());
    f.change(|m| m["targets"]["x86_64-pc-windows-msvc"]["file"] = json!("../../external.lib"));
    assert!(f.load().is_err());
}
#[test]
fn rejects_invalid_json_and_checksums() {
    let f = Fixture::new();
    f.change(|m| m["targets"]["x86_64-pc-windows-msvc"]["sha256"] = json!("0".repeat(64)));
    assert!(f.load().is_err());
    fs::write(f.0.join("prebuilt/manifest.json"), b"invalid json").unwrap();
    assert!(f.load().is_err());
}
