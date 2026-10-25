//! Build native-host SPIR-V assets. No GPU or denoiser execution is requested.
#[allow(dead_code)]
#[path = "interop/common.rs"]
mod common;
fn main() -> common::Result<()> {
    let directory = std::env::args()
        .nth(1)
        .ok_or("usage: interop_shaders OUTPUT_DIR")?;
    std::fs::create_dir_all(&directory)?;
    std::fs::write(
        std::path::Path::new(&directory).join("host.hlsl"),
        include_str!("interop/host.hlsl"),
    )?;
    for entry in ["produce", "consume"] {
        let words = common::spirv(entry)?;
        let bytes: Vec<_> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
        std::fs::write(
            std::path::Path::new(&directory).join(format!("{entry}.spv")),
            bytes,
        )?;
    }
    Ok(())
}
