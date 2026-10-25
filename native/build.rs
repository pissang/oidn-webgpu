use std::{env, fs, path::Path, process::Command};

fn watch_tree(path: &Path) {
    println!("cargo:rerun-if-changed={}", path.display());
    if path.is_dir() {
        for entry in fs::read_dir(path).unwrap() {
            watch_tree(&entry.unwrap().path());
        }
    }
}

fn main() {
    for name in ["OIDN_WEBGPU_CORE_ARCHIVE", "OIDN_WEBGPU_SCRIPTC", "OIDN_WEBGPU_NODE"] {
        println!("cargo:rerun-if-env-changed={name}");
    }
    let manifest = std::path::PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let root = manifest.parent().unwrap();
    let output = std::path::PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let name = if env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        "oidn_webgpu_core.lib"
    } else {
        "liboidn_webgpu_core.a"
    };
    let destination = output.join(name);
    if let Some(archive) = env::var_os("OIDN_WEBGPU_CORE_ARCHIVE") {
        let archive = std::path::PathBuf::from(archive);
        assert!(
            archive.is_file(),
            "OIDN_WEBGPU_CORE_ARCHIVE does not exist: {}",
            archive.display()
        );
        println!("cargo:rerun-if-changed={}", archive.display());
        fs::copy(&archive, &destination).expect("cannot copy precompiled core archive");
    } else {
        assert_eq!(
            env::var("HOST").unwrap(),
            env::var("TARGET").unwrap(),
            "automatic scriptc builds require HOST == TARGET; for cross compilation set OIDN_WEBGPU_CORE_ARCHIVE to an archive built for TARGET"
        );
        watch_tree(&root.join("src"));
        for file in [
            "scripts/native-core-build.mjs",
            "scripts/native-core-msvc.mjs",
            "native/toolchain/msvc-compat.c",
            "native/toolchain/compiler.mjs",
            "native/toolchain/package.json",
            "native/toolchain/package-lock.json",
        ] {
            watch_tree(&root.join(file));
        }
        let node = env::var_os("OIDN_WEBGPU_NODE").unwrap_or_else(|| "node".into());
        let status = Command::new(node)
            .arg(root.join("scripts/native-core-build.mjs"))
            .arg("--output")
            .arg(&destination)
            .arg("--toolchain")
            .arg(output.join("toolchain"))
            .current_dir(root)
            .status()
            .expect(
                "cannot run Node.js; install Node 22+ or set OIDN_WEBGPU_CORE_ARCHIVE to a matching precompiled core",
            );
        assert!(
            status.success(),
            "core build failed; install Node 22+, npm and a C toolchain, or set OIDN_WEBGPU_CORE_ARCHIVE to a matching precompiled core"
        );
    }
    println!("cargo:rustc-link-search=native={}", output.display());
    println!("cargo:rustc-link-lib=static=oidn_webgpu_core");
    match env::var("CARGO_CFG_TARGET_OS").as_deref() {
        Ok("macos") => println!("cargo:rustc-cdylib-link-arg=-Wl,-install_name,@rpath/liboidn_webgpu.dylib"),
        Ok("linux") => println!("cargo:rustc-cdylib-link-arg=-Wl,-soname,liboidn_webgpu.so"),
        _ => {}
    }
    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        for library in [
            "ws2_32", "bcrypt", "crypt32", "iphlpapi", "userenv", "ntdll", "advapi32",
        ] {
            println!("cargo:rustc-link-lib={library}");
        }
    }
}
