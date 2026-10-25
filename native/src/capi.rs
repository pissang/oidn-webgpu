//! C API of the standalone denoiser; documented in `include/oidn_webgpu.h`.

use crate::{DenoiseConfig, Kernel, Network, NetworkConfig, PipelineMap, Precision, WeightMap};
use std::ffi::c_char;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Mutex};
use std::time::Duration;

pub const OIDNW_OK: i32 = 0;
pub const OIDNW_ERROR_INVALID_ARGUMENT: i32 = 1;
pub const OIDNW_ERROR_UNAVAILABLE: i32 = 2;
pub const OIDNW_ERROR_MODEL: i32 = 3;
pub const OIDNW_ERROR_GPU: i32 = 4;
pub const OIDNW_ERROR_BUSY: i32 = 5;
pub const OIDNW_ERROR_DEVICE_LOST: i32 = 6;
pub const OIDNW_ERROR_TIMEOUT: i32 = 7;
pub const OIDNW_ERROR_QUEUE_FULL: i32 = 8;

pub const OIDNW_BACKEND_AUTO: u32 = 0;
pub const OIDNW_BACKEND_METAL: u32 = 1;
pub const OIDNW_BACKEND_VULKAN: u32 = 2;
pub const OIDNW_BACKEND_DX12: u32 = 3;

pub const OIDNW_QUALITY_HIGH: u32 = 0;
pub const OIDNW_QUALITY_BALANCED: u32 = 1;
pub const OIDNW_QUALITY_FAST: u32 = 2;

pub const OIDNW_KERNEL_AUTO: u32 = 0;
pub const OIDNW_KERNEL_IMPLICIT_GEMM: u32 = 1;
pub const OIDNW_KERNEL_DIRECT: u32 = 2;
pub const OIDNW_KERNEL_SUBGROUP_MATRIX: u32 = 3;

pub const OIDNW_TRANSFER_PU: u32 = 0;
pub const OIDNW_TRANSFER_LOG: u32 = 1;

pub const OIDNW_FILTER_RUNTIME_CHECKS: u32 = 1;

pub const OIDNW_FEATURE_SHADER_F16: u32 = 1;
pub const OIDNW_FEATURE_SUBGROUPS: u32 = 2;
pub const OIDNW_FEATURE_COOPERATIVE_MATRIX: u32 = 4;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct OidnwFilterDesc {
    pub hdr: u32,
    pub transfer: u32,
    pub quality: u32,
    pub kernel: u32,
    pub max_tile_size: u32,
    pub input_scale: f32,
    pub flags: u32,
}

struct Gpu {
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

/// Opaque `oidnw_device`.
pub struct OidnwDevice {
    gpu: Arc<Gpu>,
}

struct Execution {
    width: u32,
    height: u32,
    network: Network,
    inputs: Vec<wgpu::Buffer>,
    readback: wgpu::Buffer,
}

#[derive(Default)]
struct ExecutionState {
    execution: Option<Execution>,
    pipelines: PipelineMap,
    weights: WeightMap,
}

/// Opaque `oidnw_filter`.
pub struct OidnwFilter {
    gpu: Arc<Gpu>,
    weights: Vec<u8>,
    desc: OidnwFilterDesc,
    sources: usize,
    execution: Mutex<ExecutionState>,
}

#[derive(Clone, Debug)]
pub(crate) struct Failure(pub(crate) i32, pub(crate) String);

pub(crate) type Outcome<T> = std::result::Result<T, Failure>;

pub(crate) fn invalid(message: impl Into<String>) -> Failure {
    Failure(OIDNW_ERROR_INVALID_ARGUMENT, message.into())
}

/// Copies `message` into caller storage, truncated on a UTF-8 boundary.
pub(crate) unsafe fn write_error(error: *mut c_char, capacity: usize, message: &str) {
    if error.is_null() || capacity == 0 {
        return;
    }
    let mut length = message.len().min(capacity - 1);
    while !message.is_char_boundary(length) {
        length -= 1;
    }
    unsafe {
        std::ptr::copy_nonoverlapping(message.as_ptr(), error.cast::<u8>(), length);
        *error.add(length) = 0;
    }
}

/// Runs an entry point, turning failures and panics into status codes.
pub(crate) unsafe fn guard(error: *mut c_char, capacity: usize, body: impl FnOnce() -> Outcome<()>) -> i32 {
    let outcome = catch_unwind(AssertUnwindSafe(body)).unwrap_or_else(|panic| {
        let message = panic
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| panic.downcast_ref::<&str>().map(|text| text.to_string()))
            .unwrap_or_else(|| "internal error".into());
        Err(Failure(OIDNW_ERROR_GPU, message))
    });
    match outcome {
        Ok(()) => {
            unsafe { write_error(error, capacity, "") };
            OIDNW_OK
        }
        Err(Failure(status, message)) => {
            unsafe { write_error(error, capacity, &message) };
            status
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn oidnw_api_version() -> u32 {
    2 << 16
}

fn wanted_features() -> wgpu::Features {
    wgpu::Features::SHADER_F16 | wgpu::Features::SUBGROUP | wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn oidnw_filter_desc_default(desc: *mut OidnwFilterDesc) {
    if desc.is_null() {
        return;
    }
    unsafe {
        *desc = OidnwFilterDesc {
            hdr: 1,
            transfer: OIDNW_TRANSFER_PU,
            quality: OIDNW_QUALITY_HIGH,
            kernel: OIDNW_KERNEL_AUTO,
            max_tile_size: 0,
            input_scale: 1.0,
            flags: 0,
        };
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn oidnw_device_create(
    backend: u32,
    output: *mut *mut OidnwDevice,
    error: *mut c_char,
    capacity: usize,
) -> i32 {
    unsafe {
        guard(error, capacity, || {
            if output.is_null() {
                return Err(invalid("output is NULL"));
            }
            let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
            descriptor.backends = match backend {
                OIDNW_BACKEND_AUTO => wgpu::Backends::PRIMARY,
                OIDNW_BACKEND_METAL => wgpu::Backends::METAL,
                // Cooperative matrices need Vulkan on Windows and Linux.
                OIDNW_BACKEND_VULKAN => wgpu::Backends::VULKAN,
                OIDNW_BACKEND_DX12 => wgpu::Backends::DX12,
                _ => return Err(invalid(format!("unknown backend {backend}"))),
            };
            let instance = wgpu::Instance::new(descriptor);
            let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            }))
            .map_err(|failure| Failure(OIDNW_ERROR_UNAVAILABLE, format!("no GPU adapter: {failure}")))?;
            let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("oidn-webgpu"),
                required_features: adapter.features() & wanted_features(),
                required_limits: adapter.limits(),
                // SAFETY: cooperative matrices are an experimental wgpu feature
                // used only by the generated subgroup-matrix kernels.
                experimental_features: wgpu::ExperimentalFeatures::enabled(),
                ..Default::default()
            }))
            .map_err(|failure| Failure(OIDNW_ERROR_UNAVAILABLE, format!("no GPU device: {failure}")))?;
            *output = Box::into_raw(Box::new(OidnwDevice {
                gpu: Arc::new(Gpu { adapter, device, queue }),
            }));
            Ok(())
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn oidnw_device_info(
    device: *const OidnwDevice,
    name: *mut c_char,
    name_capacity: usize,
    backend: *mut u32,
    features: *mut u32,
) -> i32 {
    let Some(device) = (unsafe { device.as_ref() }) else {
        return OIDNW_ERROR_INVALID_ARGUMENT;
    };
    let info = device.gpu.adapter.get_info();
    unsafe { write_error(name, name_capacity, &info.name) };
    if let Some(backend) = unsafe { backend.as_mut() } {
        *backend = match info.backend {
            wgpu::Backend::Metal => OIDNW_BACKEND_METAL,
            wgpu::Backend::Vulkan => OIDNW_BACKEND_VULKAN,
            wgpu::Backend::Dx12 => OIDNW_BACKEND_DX12,
            _ => OIDNW_BACKEND_AUTO,
        };
    }
    if let Some(features) = unsafe { features.as_mut() } {
        let enabled = device.gpu.device.features();
        *features = [
            (wgpu::Features::SHADER_F16, OIDNW_FEATURE_SHADER_F16),
            (wgpu::Features::SUBGROUP, OIDNW_FEATURE_SUBGROUPS),
            (
                wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX,
                OIDNW_FEATURE_COOPERATIVE_MATRIX,
            ),
        ]
        .iter()
        .filter(|(feature, _)| enabled.contains(*feature))
        .map(|(_, bit)| bit)
        .sum();
    }
    OIDNW_OK
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn oidnw_device_destroy(device: *mut OidnwDevice) {
    if !device.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| drop(unsafe { Box::from_raw(device) })));
    }
}

fn network_config(gpu: &Gpu, desc: &OidnwFilterDesc, width: u32, height: u32) -> Outcome<NetworkConfig> {
    let kernel = match desc.kernel {
        OIDNW_KERNEL_AUTO
            if gpu
                .device
                .features()
                .contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) =>
        {
            Kernel::SubgroupMatrix
        }
        OIDNW_KERNEL_AUTO | OIDNW_KERNEL_IMPLICIT_GEMM => Kernel::ImplicitGemm,
        OIDNW_KERNEL_DIRECT => Kernel::Direct,
        OIDNW_KERNEL_SUBGROUP_MATRIX => Kernel::SubgroupMatrix,
        other => return Err(invalid(format!("unknown kernel {other}"))),
    };
    if desc.quality > OIDNW_QUALITY_FAST {
        return Err(invalid(format!("unknown quality {}", desc.quality)));
    }
    if desc.transfer > OIDNW_TRANSFER_LOG {
        return Err(invalid(format!("unknown transfer {}", desc.transfer)));
    }
    if !(desc.input_scale.is_finite() && desc.input_scale >= 0.0) {
        return Err(invalid("input_scale must be finite and non-negative"));
    }
    Ok(NetworkConfig {
        precision: Precision::Auto,
        kernel,
        // f16 accumulation exceeds OIDN's 0.003 avgerror on real images
        // (see the header), so only the fast quality uses it.
        f16_accumulation: desc.quality == OIDNW_QUALITY_FAST,
        denoise: Some(DenoiseConfig {
            hdr: desc.hdr != 0,
            log_transfer: desc.transfer == OIDNW_TRANSFER_LOG,
            max_tile_size: desc.max_tile_size,
            input_scale: if desc.input_scale == 0.0 { 1.0 } else { desc.input_scale },
        }),
        ..NetworkConfig::new(width, height)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn oidnw_filter_create(
    device: *const OidnwDevice,
    weights: *const u8,
    weights_size: usize,
    desc: *const OidnwFilterDesc,
    output: *mut *mut OidnwFilter,
    error: *mut c_char,
    capacity: usize,
) -> i32 {
    unsafe {
        guard(error, capacity, || {
            let device = device.as_ref().ok_or_else(|| invalid("device is NULL"))?;
            if output.is_null() || weights.is_null() || weights_size == 0 {
                return Err(invalid("weights and output are required"));
            }
            let mut filter_desc = std::mem::zeroed::<OidnwFilterDesc>();
            oidnw_filter_desc_default(&mut filter_desc);
            if let Some(desc) = desc.as_ref() {
                filter_desc = *desc;
            }
            let weights = std::slice::from_raw_parts(weights, weights_size).to_vec();
            // Validate options and model without compiling shaders or allocating GPU buffers.
            let config = network_config(&device.gpu, &filter_desc, 16, 16)?;
            let _ = config;
            let sources = crate::Model::from_tza(&weights)
                .map_err(|error| Failure(OIDNW_ERROR_MODEL, error.to_string()))?
                .input_count();
            *output = Box::into_raw(Box::new(OidnwFilter {
                gpu: device.gpu.clone(),
                weights,
                desc: filter_desc,
                sources,
                execution: Mutex::new(ExecutionState::default()),
            }));
            Ok(())
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn oidnw_filter_input_count(filter: *const OidnwFilter) -> u32 {
    unsafe { filter.as_ref() }.map_or(0, |filter| filter.sources as u32)
}

fn create_execution(filter: &OidnwFilter, state: &mut ExecutionState, width: u32, height: u32) -> Outcome<Execution> {
    let gpu = &filter.gpu;
    let size = width as u64 * height as u64 * 16;
    let storage = |label: &str, usage: wgpu::BufferUsages| {
        gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::STORAGE | usage,
            mapped_at_creation: false,
        })
    };
    let inputs = (0..filter.sources)
        .map(|_| storage("oidn-webgpu input", wgpu::BufferUsages::COPY_DST))
        .collect::<Vec<_>>();
    let output = storage("oidn-webgpu output", wgpu::BufferUsages::COPY_SRC);
    let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("oidn-webgpu readback"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let config = network_config(gpu, &filter.desc, width, height)?;
    let unchecked = filter.desc.flags & OIDNW_FILTER_RUNTIME_CHECKS == 0;
    let started = std::time::Instant::now();
    let program = crate::plan(&gpu.device, &gpu.adapter, &filter.weights, &config)
        .map_err(|message| Failure(OIDNW_ERROR_MODEL, message))?;
    // Unchecked mode follows the C caller's explicit API contract.
    let network = Network::from_program(
        &gpu.device,
        program,
        Some(&inputs),
        Some(&output),
        unchecked,
        crate::ResourceCaches {
            pipelines: &mut state.pipelines,
            weights: &mut state.weights,
        },
        started.elapsed().as_secs_f64() * 1e3,
    )
    .map_err(|message| Failure(OIDNW_ERROR_MODEL, message))?;
    Ok(Execution {
        width,
        height,
        network,
        inputs,
        readback,
    })
}

fn validate_size(width: u32, height: u32) -> Outcome<()> {
    if !(1..=16384).contains(&width) || !(1..=16384).contains(&height) {
        return Err(invalid("width and height must be in 1..=16384"));
    }
    Ok(())
}

fn ensure_execution(filter: &OidnwFilter, state: &mut ExecutionState, width: u32, height: u32) -> Outcome<()> {
    if state
        .execution
        .as_ref()
        .is_none_or(|execution| (execution.width, execution.height) != (width, height))
    {
        let execution = create_execution(filter, state, width, height)?;
        state.execution = Some(execution);
    }
    Ok(())
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn oidnw_filter_prepare(
    filter: *const OidnwFilter,
    width: u32,
    height: u32,
    error: *mut c_char,
    capacity: usize,
) -> i32 {
    unsafe {
        guard(error, capacity, || {
            let filter = filter.as_ref().ok_or_else(|| invalid("filter is NULL"))?;
            validate_size(width, height)?;
            let mut state = filter
                .execution
                .try_lock()
                .map_err(|_| Failure(OIDNW_ERROR_BUSY, "filter is busy".into()))?;
            ensure_execution(filter, &mut state, width, height)
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn oidnw_filter_execute(
    filter: *const OidnwFilter,
    width: u32,
    height: u32,
    color: *const f32,
    albedo: *const f32,
    normal: *const f32,
    output: *mut f32,
    error: *mut c_char,
    capacity: usize,
) -> i32 {
    unsafe {
        guard(error, capacity, || {
            let filter = filter.as_ref().ok_or_else(|| invalid("filter is NULL"))?;
            validate_size(width, height)?;
            let sources = [color, albedo, normal];
            if output.is_null() || sources[..filter.sources].iter().any(|source| source.is_null()) {
                return Err(invalid(format!(
                    "color{} and output are required",
                    ["", " and albedo", ", albedo and normal"][filter.sources - 1]
                )));
            }
            let mut slot = filter
                .execution
                .try_lock()
                .map_err(|_| Failure(OIDNW_ERROR_BUSY, "the filter is executing on another thread".into()))?;
            ensure_execution(filter, &mut slot, width, height)?;
            let execution = slot.execution.as_ref().unwrap();
            let gpu = &filter.gpu;
            let floats = width as usize * height as usize * 4;
            for (buffer, source) in execution.inputs.iter().zip(sources) {
                let pixels = std::slice::from_raw_parts(source.cast::<u8>(), floats * 4);
                gpu.queue.write_buffer(buffer, 0, pixels);
            }
            let mut encoder = gpu.device.create_command_encoder(&Default::default());
            execution.network.encode(&gpu.queue, &mut encoder);
            encoder.copy_buffer_to_buffer(
                execution.network.output_buffer(),
                0,
                &execution.readback,
                0,
                (floats * 4) as u64,
            );
            gpu.queue.submit([encoder.finish()]);
            let (send, receive) = std::sync::mpsc::channel();
            execution.readback.map_async(wgpu::MapMode::Read, .., move |result| {
                let _ = send.send(result);
            });
            gpu.device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: Some(Duration::from_secs(60)),
                })
                .map_err(|failure| Failure(OIDNW_ERROR_GPU, format!("GPU wait failed: {failure:?}")))?;
            receive
                .recv_timeout(Duration::from_secs(5))
                .map_err(|_| Failure(OIDNW_ERROR_GPU, "GPU readback did not complete".into()))?
                .map_err(|failure| Failure(OIDNW_ERROR_GPU, format!("GPU readback failed: {failure}")))?;
            {
                let mapped = execution.readback.get_mapped_range(..).unwrap();
                std::ptr::copy_nonoverlapping(mapped.as_ptr(), output.cast::<u8>(), floats * 4);
            }
            execution.readback.unmap();
            Ok(())
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn oidnw_filter_destroy(filter: *mut OidnwFilter) {
    if !filter.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| drop(unsafe { Box::from_raw(filter) })));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires a native GPU adapter"]
    fn explicit_prepare_reuses_resources_and_matches_execution() {
        unsafe {
            let mut error = [0i8; 1024];
            let mut device = std::ptr::null_mut();
            assert_eq!(oidnw_api_version(), 0x00020000);
            assert_eq!(
                oidnw_device_create(OIDNW_BACKEND_AUTO, &mut device, error.as_mut_ptr(), error.len()),
                OIDNW_OK
            );
            let bytes = std::fs::read(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../weights/rt_hdr_alb_nrm_small.tza"
            ))
            .unwrap();
            let mut desc = std::mem::zeroed();
            oidnw_filter_desc_default(&mut desc);
            desc.flags = OIDNW_FILTER_RUNTIME_CHECKS;
            desc.kernel = OIDNW_KERNEL_IMPLICIT_GEMM;
            let mut filter = std::ptr::null_mut();
            assert_eq!(
                oidnw_filter_create(
                    device,
                    bytes.as_ptr(),
                    bytes.len(),
                    &desc,
                    &mut filter,
                    error.as_mut_ptr(),
                    error.len()
                ),
                OIDNW_OK
            );
            assert!((*filter).execution.lock().unwrap().execution.is_none());
            assert_eq!(
                oidnw_filter_prepare(filter, 0, 16, error.as_mut_ptr(), error.len()),
                OIDNW_ERROR_INVALID_ARGUMENT
            );
            assert_eq!(
                oidnw_filter_prepare(filter, 16, 16, error.as_mut_ptr(), error.len()),
                OIDNW_OK
            );
            let (frame, weight) = {
                let state = (*filter).execution.lock().unwrap();
                let network = &state.execution.as_ref().unwrap().network;
                let meta = network.metadata.as_ref().unwrap();
                (
                    network.buffers[meta.frame.unwrap()].clone(),
                    network.buffers[meta.weights[0]].clone(),
                )
            };
            assert_eq!(
                oidnw_filter_prepare(filter, 16, 16, error.as_mut_ptr(), error.len()),
                OIDNW_OK
            );
            {
                let state = (*filter).execution.lock().unwrap();
                let network = &state.execution.as_ref().unwrap().network;
                assert_eq!(
                    network.buffers[network.metadata.as_ref().unwrap().frame.unwrap()],
                    frame
                );
                assert_eq!(
                    oidnw_filter_prepare(filter, 16, 16, error.as_mut_ptr(), error.len()),
                    OIDNW_ERROR_BUSY
                );
            }
            assert_eq!(
                oidnw_filter_prepare(filter, 17, 19, error.as_mut_ptr(), error.len()),
                OIDNW_OK
            );
            {
                let state = (*filter).execution.lock().unwrap();
                let network = &state.execution.as_ref().unwrap().network;
                assert_eq!(network.buffers[network.metadata.as_ref().unwrap().weights[0]], weight);
            }
            let input: Vec<f32> = (0..17 * 19).flat_map(|_| [0.5, 0.3, 0.1, 0.75]).collect();
            let mut output = vec![0.0; input.len()];
            assert_eq!(
                oidnw_filter_execute(
                    filter,
                    17,
                    19,
                    input.as_ptr(),
                    input.as_ptr(),
                    input.as_ptr(),
                    output.as_mut_ptr(),
                    error.as_mut_ptr(),
                    error.len()
                ),
                OIDNW_OK
            );
            assert!(output.iter().all(|v| v.is_finite()));
            assert!(output.chunks_exact(4).all(|pixel| pixel[3] == 0.75));
            oidnw_filter_destroy(filter);
            oidnw_device_destroy(device);
        }
    }
}
