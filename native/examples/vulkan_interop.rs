//! Native Vulkan host-device import feasibility test. No public GPU C ABI.
#[cfg(target_os = "windows")]
#[path = "interop/vulkan.rs"]
mod vulkan;
#[cfg(target_os = "windows")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    vulkan::run()
}
#[cfg(not(target_os = "windows"))]
fn main() {
    eprintln!("vulkan_interop currently requires Windows");
    std::process::exit(1);
}
