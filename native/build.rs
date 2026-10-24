// Links the scriptc library built from src/native/core-profile.json:
//   scriptc build --lib --profile src/native/core-profile.json -o .native-build/oidn_core.lib.a
fn main() {
    println!("cargo:rerun-if-env-changed=OIDN_WEB_CORE_ARCHIVE");
    let manifest = std::path::PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let archive = std::env::var_os("OIDN_WEB_CORE_ARCHIVE")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| manifest.join("../.native-build/oidn_core.lib.a"));
    assert!(
        archive.is_file(),
        "scriptc archive missing: {}. Build it with scriptc or set OIDN_WEB_CORE_ARCHIVE.",
        archive.display()
    );
    println!("cargo:rerun-if-changed={}", archive.display());
    let output = std::path::PathBuf::from(std::env::var_os("OUT_DIR").unwrap());
    let name = if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        "oidn_web_core.lib"
    } else {
        "liboidn_web_core.a"
    };
    std::fs::copy(&archive, output.join(name)).unwrap();
    println!("cargo:rustc-link-search=native={}", output.display());
    println!("cargo:rustc-link-lib=static=oidn_web_core");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        for library in ["ws2_32", "bcrypt", "crypt32", "iphlpapi", "userenv", "ntdll", "advapi32"] {
            println!("cargo:rustc-link-lib={library}");
        }
    }
}
