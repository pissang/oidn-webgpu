//! Native DX12 shared-buffer/fence feasibility test. No public GPU C ABI.
#[cfg(target_os = "windows")]
#[path = "interop/dx12.rs"]
mod dx12;
#[cfg(target_os = "windows")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    dx12::run()
}
#[cfg(not(target_os = "windows"))]
fn main() {
    eprintln!("dx12_interop requires Windows");
    std::process::exit(1);
}
