//! Full-image GPU-buffer latency; preprocessing and output conversion included.
//! Upload/download and preparation are outside the measured executions.
//! --weights PATH --width 512 --height 512 [--warmup 3] [--runs 7]
//! [--kernel auto|implicit-gemm] [--runtime-checks] [--output image.rgba.f32]
use oidn_webgpu::{DenoiseBuffers, DenoiseParams, Denoiser, DenoiserOptions, Model};
use std::time::{Duration, Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
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
    let width: u32 = option("--width", "512")?.parse()?;
    let height: u32 = option("--height", "512")?.parse()?;
    let warmup: usize = option("--warmup", "3")?.parse()?;
    let runs: usize = option("--runs", "7")?.parse()?;
    if !(1..=16384).contains(&width) || !(1..=16384).contains(&height) || runs == 0 {
        return Err("dimensions must be in 1..=16384 and runs must be positive".into());
    }
    let output = if args.iter().any(|arg| arg == "--output") {
        Some(option("--output", "")?)
    } else {
        None
    };
    let kernel = option("--kernel", "auto")?;
    if !["auto", "implicit-gemm"].contains(&kernel.as_str()) {
        return Err("unsupported kernel".into());
    }
    let checks = args.iter().any(|arg| arg == "--runtime-checks");
    let model = Model::from_tza(&std::fs::read(option(
        "--weights",
        "weights/rt_hdr_alb_nrm.tza",
    )?)?)?;
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))?;
    let matrix = wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX;
    let mut features = Denoiser::recommended_features(&adapter);
    if kernel == "auto" {
        features |= adapter.features() & matrix;
    }
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features: features,
        required_limits: adapter.limits(),
        // SAFETY: this benchmark opts into wgpu's experimental matrix contract.
        experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
        ..Default::default()
    }))?;
    let pixels = width as usize * height as usize;
    let size = pixels as u64 * 16;
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
            queue.write_buffer(&buffer, 0, &data);
        }
        images.push(buffer);
    }
    let started = Instant::now();
    let mut denoiser = if checks {
        Denoiser::new(&device, &adapter, &model, DenoiserOptions::default())?
    } else {
        // SAFETY: only the repository's generated kernels are benchmarked;
        // their bounds and workgroup initialization are covered by GPU checks.
        unsafe { Denoiser::new_unchecked(&device, &adapter, &model, DenoiserOptions::default())? }
    };
    let mut prepared = denoiser.prepare(
        [width, height],
        DenoiseBuffers {
            color: (&images[0]).into(),
            albedo: Some((&images[1]).into()),
            normal: Some((&images[2]).into()),
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
    // Verify the final output outside the timed region.
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        mapped_at_creation: false,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(&images[3], 0, &readback, 0, size);
    queue.submit([encoder.finish()]);
    let (send, receive) = std::sync::mpsc::channel();
    readback.map_async(wgpu::MapMode::Read, .., move |r| {
        let _ = send.send(r);
    });
    wait()?;
    receive.recv()??;
    let mapped = readback.get_mapped_range(..)?;
    for pixel in mapped.chunks_exact(16) {
        if pixel[..12]
            .chunks_exact(4)
            .any(|v| !f32::from_le_bytes(v.try_into().unwrap()).is_finite())
        {
            return Err("non-finite output".into());
        }
    }
    if let Some(path) = output {
        // Save all RGBA32F pixels after verification, outside the timed region.
        std::fs::write(path, &mapped[..])?;
    }
    let info = prepared.info();
    println!(
        "{{\"label\":\"oidn-web(native wgpu)\",\"adapter\":{:?},\"backend\":\"{:?}\",\"width\":{},\"height\":{},\"runtimeChecks\":{},\"kernels\":\"{:?}\",\"tileCount\":{},\"workspaceBytes\":{},\"sharedWeightBytes\":{},\"initializationMs\":{},\"timesMs\":{:?},\"finiteOutput\":true}}",
        adapter.get_info().name,
        adapter.get_info().backend,
        width,
        height,
        checks,
        info.kernels,
        info.tile_count,
        info.workspace_bytes,
        info.shared_weight_bytes,
        initialization_ms,
        times
    );
    Ok(())
}
