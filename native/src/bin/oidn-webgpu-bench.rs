//! Benchmarks and checks the native oidn-webgpu network.
//!
//!   oidn-webgpu-bench --weights rt_hdr_alb_nrm.tza --width 512 --height 512
//!     [--kernel implicit-gemm|direct|auto] [--precision fp16|fp32]
//!     [--batch 50] [--rounds 5]
//!     [--input tensor.f32] [--output denoised.f32] [--unchecked]
//!     [--backend vulkan|metal|dx12] [--no-double-buffer|--double-buffer]
//!     [--f16-accumulation] [--no-sub-pixel]
//!
//! `--input` holds interleaved f32 pixels with the model's input channels in
//! the network's transfer space (as for the browser `executeCPU`); `--output`
//! receives interleaved RGB f32 pixels in the same space.
use oidn_webgpu::{Kernel, Network, NetworkConfig, Precision};
use std::time::{Duration, Instant};

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let option = |name: &str| {
        args.iter()
            .position(|arg| arg == name)
            .map(|index| args[index + 1].clone())
    };
    let weights = option("--weights").ok_or("--weights is required")?;
    let width: u32 = option("--width")
        .map_or(Ok(512), |value| value.parse())
        .map_err(|e| format!("{e}"))?;
    let height: u32 = option("--height")
        .map_or(Ok(512), |value| value.parse())
        .map_err(|e| format!("{e}"))?;
    let batch: u32 = option("--batch")
        .map_or(Ok(50), |value| value.parse())
        .map_err(|e| format!("{e}"))?;
    let rounds: u32 = option("--rounds")
        .map_or(Ok(5), |value| value.parse())
        .map_err(|e| format!("{e}"))?;
    if width == 0 || height == 0 || batch == 0 || rounds == 0 {
        return Err("width, height, batch and rounds must be positive".into());
    }
    let kernel = match option("--kernel").as_deref() {
        None | Some("implicit-gemm") => Kernel::ImplicitGemm,
        Some("direct") => Kernel::Direct,
        Some("auto") => Kernel::Auto,
        Some("subgroup-matrix") => Kernel::SubgroupMatrix,
        Some(other) => return Err(format!("unknown kernel {other}")),
    };
    let precision = match option("--precision").as_deref() {
        None | Some("fp16") => Precision::Fp16,
        Some("fp32") => Precision::Fp32,
        Some(other) => return Err(format!("unknown precision {other}")),
    };
    let tza = std::fs::read(&weights).map_err(|e| format!("{weights}: {e}"))?;

    // Cooperative matrices need Vulkan on Windows, where wgpu prefers DX12.
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    if let Some(backend) = option("--backend") {
        descriptor.backends = match backend.as_str() {
            "vulkan" => wgpu::Backends::VULKAN,
            "metal" => wgpu::Backends::METAL,
            "dx12" => wgpu::Backends::DX12,
            other => return Err(format!("unknown backend {other}")),
        };
    }
    let instance = wgpu::Instance::new(descriptor);
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .map_err(|e| format!("no adapter: {e}"))?;
    let info = adapter.get_info();
    let timestamp_features = wgpu::Features::TIMESTAMP_QUERY;
    let mut wanted = wgpu::Features::SHADER_F16 | wgpu::Features::SUBGROUP;
    if adapter.features().contains(timestamp_features) {
        wanted |= timestamp_features;
    }
    if kernel == Kernel::SubgroupMatrix {
        wanted |= wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX;
        for properties in adapter.cooperative_matrix_properties() {
            println!("cooperative matrix: {properties:?}");
        }
    }
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features: adapter.features() & wanted,
        required_limits: adapter.limits(),
        // SAFETY: cooperative matrices are an experimental wgpu feature.
        experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
        ..Default::default()
    }))
    .map_err(|e| format!("no device: {e}"))?;
    println!(
        "{} ({:?}), subgroup {}-{}, features {:?}",
        info.name,
        info.backend,
        info.subgroup_min_size,
        info.subgroup_max_size,
        device.features() & wanted
    );

    let flag = |name: &str| args.iter().any(|arg| arg == name);
    let config = NetworkConfig {
        precision,
        kernel,
        double_buffer: if flag("--double-buffer") {
            Some(true)
        } else if flag("--no-double-buffer") {
            Some(false)
        } else {
            None
        },
        f16_accumulation: flag("--f16-accumulation"),
        sub_pixel: !flag("--no-sub-pixel"),
        ..NetworkConfig::new(width, height)
    };
    let created = Instant::now();
    let network = if args.iter().any(|arg| arg == "--unchecked") {
        // SAFETY: benchmarking the generated kernels without runtime checks.
        unsafe { Network::new_unchecked(&device, &adapter, &tza, &config, None)? }
    } else {
        Network::new(&device, &adapter, &tza, &config, None)?
    };
    println!(
        "planned in {:.1} ms, resources created in {:.1} ms",
        network.planning_ms,
        created.elapsed().as_secs_f64() * 1e3 - network.planning_ms
    );

    let pixels = (width * height) as usize;
    let sources = network.input_buffers().count();
    let channels = sources * 3;
    let interleaved: Vec<f32> = match option("--input") {
        Some(path) => {
            let bytes = std::fs::read(&path).map_err(|e| format!("{path}: {e}"))?;
            if bytes.len() != pixels * channels * 4 {
                return Err(format!(
                    "{path} has {} bytes, expected {}",
                    bytes.len(),
                    pixels * channels * 4
                ));
            }
            bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect()
        }
        None => {
            // Deterministic values in the range of transfer-space inputs.
            let mut state = 1u32;
            (0..pixels * channels)
                .map(|_| {
                    state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                    (state >> 8) as f32 / (1u32 << 24) as f32
                })
                .collect()
        }
    };
    for (source, buffer) in network.input_buffers().enumerate() {
        let mut rgba = vec![0f32; pixels * 4];
        for pixel in 0..pixels {
            for channel in 0..3 {
                rgba[pixel * 4 + channel] = interleaved[pixel * channels + source * 3 + channel];
            }
        }
        let bytes: Vec<u8> = rgba.iter().flat_map(|v| v.to_le_bytes()).collect();
        queue.write_buffer(buffer, 0, &bytes);
    }

    let wait = || {
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(60)),
            })
            .map_err(|e| format!("{e:?}"))
    };
    // Timestamp the entire batch, including dispatch barriers. Keep wall time
    // separate: command encoding and host waits are not GPU execution time.
    let timestamp_available = device.features().contains(timestamp_features)
        && batch <= wgpu::QUERY_SET_MAX_QUERIES / 2;
    let run = |count: u32| -> Result<(f64, Option<f64>), String> {
        let timestamps = timestamp_available.then(|| {
            let queries = device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("oidn batch timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count: count * 2,
            });
            let resolve = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: u64::from(count) * 16,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: u64::from(count) * 16,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            (queries, resolve, readback)
        });
        let started = Instant::now();
        let mut encoder = device.create_command_encoder(&Default::default());
        for index in 0..count {
            if let Some((queries, _, _)) = &timestamps {
                // Metal stage-boundary sampling needs a begin/end pair on
                // each measured pass, including the first and last images.
                network.encode_with_timestamps(
                    &mut encoder,
                    queries,
                    Some(index * 2),
                    Some(index * 2 + 1),
                );
            } else {
                network.encode(&queue, &mut encoder);
            }
        }
        if let Some((queries, resolve, readback)) = &timestamps {
            encoder.resolve_query_set(queries, 0..count * 2, resolve, 0);
            encoder.copy_buffer_to_buffer(resolve, 0, readback, 0, u64::from(count) * 16);
        }
        queue.submit([encoder.finish()]);
        wait()?;
        let wall_ms = started.elapsed().as_secs_f64() * 1e3 / count as f64;
        let gpu_ms = if let Some((_, _, readback)) = &timestamps {
            let (send, receive) = std::sync::mpsc::channel();
            readback.map_async(wgpu::MapMode::Read, .., move |result| {
                let _ = send.send(result);
            });
            wait()?;
            receive
                .recv_timeout(Duration::from_secs(5))
                .map_err(|e| e.to_string())?
                .map_err(|e| e.to_string())?;
            let mapped = readback.get_mapped_range(..).map_err(|e| e.to_string())?;
            let begin = u64::from_le_bytes(mapped[0..8].try_into().unwrap());
            let offset = (count as usize - 1) * 16 + 8;
            let end = u64::from_le_bytes(mapped[offset..offset + 8].try_into().unwrap());
            drop(mapped);
            readback.unmap();
            let elapsed = end.saturating_sub(begin) as f64 * queue.get_timestamp_period() as f64
                / 1e6
                / count as f64;
            if elapsed > 0.0 && elapsed <= wall_ms * 1.1 {
                Some(elapsed)
            } else {
                None
            }
        } else {
            None
        };
        Ok((wall_ms, gpu_ms))
    };
    run(batch)?;
    let samples = (0..rounds)
        .map(|_| run(batch))
        .collect::<Result<Vec<_>, _>>()?;
    let mut times: Vec<_> = samples.iter().map(|sample| sample.0).collect();
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "{width}x{height} {kernel:?} {precision:?}: median {:.3} ms, min {:.3} ms per image ({batch} per batch, {rounds} rounds)",
        times[times.len() / 2],
        times[0]
    );
    let mut gpu_times: Vec<_> = samples.iter().filter_map(|sample| sample.1).collect();
    gpu_times.sort_by(f64::total_cmp);
    if gpu_times.len() == samples.len() {
        println!(
            "GPU timestamps: median {:.3} ms, min {:.3} ms per image",
            gpu_times[gpu_times.len() / 2],
            gpu_times[0]
        );
    }
    if gpu_times.len() != samples.len() && timestamp_available {
        println!(
            "GPU timestamps unavailable for some batches: invalid device counter results; wall times remain valid"
        );
    }

    if let Some(path) = option("--output") {
        let size = (pixels * 16) as u64;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("oidn-webgpu readback"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        network.encode(&queue, &mut encoder);
        encoder.copy_buffer_to_buffer(network.output_buffer(), 0, &readback, 0, size);
        queue.submit([encoder.finish()]);
        readback.map_async(wgpu::MapMode::Read, .., |_| {});
        wait()?;
        let mapped = readback.get_mapped_range(..).unwrap();
        let rgba: Vec<f32> = mapped
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let rgb: Vec<u8> = (0..pixels)
            .flat_map(|pixel| (0..3).map(move |channel| (pixel, channel)))
            .flat_map(|(pixel, channel)| rgba[pixel * 4 + channel].to_le_bytes())
            .collect();
        std::fs::write(&path, rgb).map_err(|e| format!("{path}: {e}"))?;
        println!("wrote {path}");
    }
    Ok(())
}
