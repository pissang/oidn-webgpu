//! Standalone Rust application; link a CI core with OIDN_WEBGPU_CORE_ARCHIVE.
//! --weights PATH [--output image.f32] [--width 256] [--height 256]
//! [--backend auto|vulkan|dx12|metal] [--matrix] [--unchecked]
//! [--tile 0] [--warmup 5] [--runs 21] [--input-dir DIR]
//! --inspect-model PATH validates the model without opening a GPU (used in CI).
//! GPU timings exclude uploads, readback, preparation and output validation.
use oidn_webgpu::{DenoiseBuffers, DenoiseParams, Denoiser, DenoiserOptions, Model, wgpu};
use std::time::{Duration, Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().is_some_and(|arg| arg == "--inspect-model") {
        if args.len() != 2 {
            return Err("usage: --inspect-model WEIGHTS.tza".into());
        }
        let model = Model::from_tza(&std::fs::read(&args[1])?)?;
        println!(
            "{{\"modelInputs\":{},\"gpuRequested\":false}}",
            model.input_count()
        );
        return Ok(());
    }
    if args.is_empty() || args.iter().any(|arg| arg == "--help") {
        println!(concat!(
            "--weights PATH [--output image.f32] [--width 256] [--height 256] ",
            "[--backend auto|vulkan|dx12|metal] [--matrix] [--unchecked] ",
            "[--tile 0] [--warmup 5] [--runs 21] [--input-dir DIR]"
        ));
        return Ok(());
    }
    let option = |name: &str, default: &str| -> Result<String, String> {
        match args.iter().position(|arg| arg == name) {
            Some(i) => args
                .get(i + 1)
                .filter(|s| !s.starts_with("--"))
                .cloned()
                .ok_or_else(|| format!("missing value for {name}")),
            None => Ok(default.into()),
        }
    };
    let width: u32 = option("--width", "256")?.parse()?;
    let height: u32 = option("--height", "256")?.parse()?;
    let warmup: usize = option("--warmup", "5")?.parse()?;
    let runs: usize = option("--runs", "21")?.parse()?;
    if !(1..=16384).contains(&width)
        || !(1..=16384).contains(&height)
        || !(1..=10000).contains(&runs)
        || warmup > 10000
    {
        return Err(
            "dimensions must be in 1..=16384, runs in 1..=10000 and warmup in 0..=10000".into(),
        );
    }
    let output = if args.iter().any(|arg| arg == "--output") {
        Some(option("--output", "")?)
    } else {
        None
    };
    let unchecked = args.iter().any(|arg| arg == "--unchecked");
    let model = Model::from_tza(&std::fs::read(option("--weights", "")?)?)?;
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = match option("--backend", "auto")?.as_str() {
        "auto" => descriptor.backends,
        "vulkan" => wgpu::Backends::VULKAN,
        "dx12" => wgpu::Backends::DX12,
        "metal" => wgpu::Backends::METAL,
        _ => return Err("backend must be auto, vulkan, dx12 or metal".into()),
    };
    let instance = wgpu::Instance::new(descriptor);
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))?;
    let matrix = wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX;
    let use_matrix = args.iter().any(|arg| arg == "--matrix");
    if use_matrix && !adapter.features().contains(matrix) {
        return Err(
            "adapter does not support cooperative matrices; on Windows use --backend vulkan".into(),
        );
    }
    let mut features = Denoiser::required_features() | Denoiser::recommended_features(&adapter);
    if use_matrix {
        features |= matrix;
    }
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features: features,
        required_limits: adapter.limits(),
        // SAFETY: --matrix explicitly opts into wgpu's experimental API.
        experimental_features: if use_matrix {
            unsafe { wgpu::ExperimentalFeatures::enabled() }
        } else {
            Default::default()
        },
        ..Default::default()
    }))?;
    let pixels = width as usize * height as usize;
    let size = pixels as u64 * 16;
    if size > device.limits().max_storage_buffer_binding_size {
        return Err("image exceeds the adapter's storage binding limit".into());
    }
    let input_dir = if args.iter().any(|arg| arg == "--input-dir") {
        Some(option("--input-dir", "")?)
    } else {
        None
    };
    let mut expected_alpha = Vec::with_capacity(pixels);
    let mut images = Vec::new();
    for source in 0..4 {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            mapped_at_creation: false,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        if source < 3 {
            let mut data = Vec::with_capacity(pixels * 16);
            if let Some(directory) = &input_dir {
                data = std::fs::read(
                    std::path::Path::new(directory)
                        .join(format!("{}.f32", ["color", "albedo", "normal"][source])),
                )?;
                if data.len() as u64 != size {
                    return Err("input file byte count mismatch".into());
                }
            } else {
                for y in 0..height {
                    for x in 0..width {
                        let fx = x as f32 / (width - 1).max(1) as f32;
                        let fy = y as f32 / (height - 1).max(1) as f32;
                        let rgba = match source {
                            0 => [
                                0.1 + 4.0 * fx,
                                0.05 + 2.0 * fy,
                                0.2 + ((x * 17 + y * 13) % 97) as f32 / 97.0,
                                1.0,
                            ],
                            1 => [0.2 + 0.7 * fx, 0.3 + 0.6 * fy, 0.55, 1.0],
                            _ => [0.0, 0.0, 1.0, 1.0], // Native API uses signed normals.
                        };
                        data.extend(rgba.into_iter().flat_map(f32::to_le_bytes));
                    }
                }
            }
            if source == 0 {
                expected_alpha.extend(
                    data.chunks_exact(16)
                        .map(|p| f32::from_le_bytes(p[12..16].try_into().unwrap())),
                );
            }
            queue.write_buffer(&buffer, 0, &data);
        }
        images.push(buffer);
    }
    let started = Instant::now();
    let mut options = DenoiserOptions::default();
    options.max_tile_size = option("--tile", "0")?.parse()?;
    let mut denoiser = if unchecked {
        // SAFETY: only the library's generated kernels are executed. They
        // have GPU regression coverage; all host buffer ranges are checked
        // by prepare. --unchecked explicitly opts out of shader checks.
        unsafe { Denoiser::new_unchecked(&device, &adapter, &model, options)? }
    } else {
        Denoiser::new(&device, &adapter, &model, options)?
    };
    let mut prepared = denoiser.prepare(
        [width, height],
        DenoiseBuffers {
            color: (&images[0]).into(),
            albedo: (model.input_count() >= 2).then_some((&images[1]).into()),
            normal: (model.input_count() >= 3).then_some((&images[2]).into()),
            output: (&images[3]).into(),
        },
    )?;
    let initialization_ms = started.elapsed().as_secs_f64() * 1000.0;
    let wait = || {
        device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(120)),
        })
    };
    queue.submit([]);
    wait()?;
    let mut execute = || -> Result<f64, Box<dyn std::error::Error>> {
        let started = Instant::now();
        let mut encoder = device.create_command_encoder(&Default::default());
        prepared.encode(&mut encoder, DenoiseParams::default())?;
        queue.submit([encoder.finish()]);
        wait()?;
        Ok(started.elapsed().as_secs_f64() * 1000.0)
    };
    for _ in 0..warmup {
        execute()?;
    }
    let times = (0..runs)
        .map(|_| execute())
        .collect::<Result<Vec<_>, _>>()?;
    // All readbacks and validation are outside the performance interval.
    let expected = read_output(&device, &queue, &images[3], size)?;
    validate(&expected, &expected_alpha)?;
    for _ in 0..3 {
        execute()?;
        let actual = read_output(&device, &queue, &images[3], size)?;
        validate(&actual, &expected_alpha)?;
        if actual != expected {
            return Err("repeated RGBA output differs".into());
        }
    }
    if let Some(path) = output {
        std::fs::write(path, &expected)?;
    }
    let mut sorted = times.clone();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    let median = if sorted.len() % 2 == 0 {
        (sorted[middle - 1] + sorted[middle]) * 0.5
    } else {
        sorted[middle]
    };
    let info = prepared.info();
    println!(
        "{{\"adapter\":{:?},\"backend\":\"{:?}\",\"width\":{},\"height\":{},\"runtimeChecks\":{},\"kernels\":\"{:?}\",\"tileCount\":{},\"workspaceBytes\":{},\"sharedWeightBytes\":{},\"initializationMs\":{},\"timesMs\":{:?},\"medianMs\":{},\"finiteNonnegative\":true,\"alphaPreserved\":true,\"repeatIdentical\":true}}",
        adapter.get_info().name,
        adapter.get_info().backend,
        width,
        height,
        !unchecked,
        info.kernels,
        info.tile_count,
        info.workspace_bytes,
        info.shared_weight_bytes,
        initialization_ms,
        times,
        median
    );
    Ok(())
}

fn read_output(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    output: &wgpu::Buffer,
    size: u64,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("validation readback"),
        size,
        mapped_at_creation: false,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(output, 0, &staging, 0, size);
    let submission = queue.submit([encoder.finish()]);
    let (send, receive) = std::sync::mpsc::channel();
    staging.map_async(wgpu::MapMode::Read, .., move |r| {
        let _ = send.send(r);
    });
    device.poll(wgpu::PollType::Wait {
        submission_index: Some(submission),
        timeout: Some(Duration::from_secs(120)),
    })?;
    receive.recv_timeout(Duration::from_secs(5))??;
    let bytes = staging.get_mapped_range(..)?.to_vec();
    staging.unmap();
    Ok(bytes)
}

fn validate(bytes: &[u8], alpha: &[f32]) -> Result<(), Box<dyn std::error::Error>> {
    if bytes.len() != alpha.len() * 16 {
        return Err("output byte count mismatch".into());
    }
    for (pixel, expected_alpha) in bytes.chunks_exact(16).zip(alpha) {
        for value in pixel[..12].chunks_exact(4) {
            let value = f32::from_le_bytes(value.try_into().unwrap());
            if !value.is_finite() || value < 0.0 {
                return Err("non-finite or negative RGB output".into());
            }
        }
        if f32::from_le_bytes(pixel[12..16].try_into().unwrap()) != *expected_alpha {
            return Err("output alpha changed".into());
        }
    }
    Ok(())
}
