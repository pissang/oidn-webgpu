//! A host creates GPU image buffers, produces inputs in a compute pass, then
//! denoises on the same encoder. Only the final file export reads back to CPU.
use oidn_webgpu::{DenoiseBuffers, DenoiseParams, Denoiser, DenoiserOptions, Model, wgpu};
use std::{error::Error, time::Duration};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() < 3 {
        return Err("usage: host_buffers WEIGHTS.tza OUTPUT.f32 [width=256] [height=192] [max_tile_size=0]".into());
    }
    let number = |index, default| -> Result<u32, Box<dyn Error>> {
        Ok(args
            .get(index)
            .map(|s: &String| s.parse())
            .transpose()?
            .unwrap_or(default))
    };
    let width = number(3, 256)?;
    let height = number(4, 192)?;
    if !(1..=16384).contains(&width) || !(1..=16384).contains(&height) {
        return Err("dimensions must be in 1..=16384".into());
    }
    let model = Model::from_tza(&std::fs::read(&args[1])?)?;
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))?;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("example host"),
        required_features: Denoiser::required_features() | Denoiser::recommended_features(&adapter),
        // Request the adapter's buffer limits for larger images. An application
        // can instead request only the limits its own workloads need.
        required_limits: adapter.limits(),
        ..Default::default()
    }))?;
    let bytes = width as u64 * height as u64 * 16;
    if bytes > device.limits().max_storage_buffer_binding_size {
        return Err("image exceeds the device's storage binding limit".into());
    }
    let storage = |label, extra| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | extra,
            mapped_at_creation: false,
        })
    };
    let color = storage("host color", wgpu::BufferUsages::empty());
    let albedo = storage("host albedo", wgpu::BufferUsages::empty());
    let normal = storage("host normal", wgpu::BufferUsages::empty());
    let output = storage("host denoised", wgpu::BufferUsages::COPY_SRC);

    // Replace this producer with your renderer's normalized, linear AOVs.
    let code = format!(
        r#"
        @group(0) @binding(0) var<storage, read_write> color: array<vec4f>;
        @group(0) @binding(1) var<storage, read_write> albedo: array<vec4f>;
        @group(0) @binding(2) var<storage, read_write> normal: array<vec4f>;
        @compute @workgroup_size(8, 8)
        fn main(@builtin(global_invocation_id) id: vec3u) {{
            if (id.x >= {width}u || id.y >= {height}u) {{ return; }}
            let i = id.y * {width}u + id.x;
            let noise = f32((i * 1664525u + 1013904223u) >> 24u) / 255.0 - 0.5;
            let signal = 0.5 + f32(id.x) / f32({width}u);
            color[i] = vec4f(max(vec3f(0.8, 0.6, 0.4) * signal + vec3f(noise * 0.3), vec3f(0.0)), 0.75);
            albedo[i] = vec4f(0.8, 0.6, 0.4, 0.0);
            normal[i] = vec4f(0.0, 0.0, 1.0, 0.0);
        }}
    "#
    );
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("host producer"),
        source: wgpu::ShaderSource::Wgsl(code.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("host producer"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("host images"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: color.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: albedo.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: normal.as_entire_binding(),
            },
        ],
    });
    let mut options = DenoiserOptions::default();
    options.max_tile_size = number(5, 0)?;
    let mut denoiser = Denoiser::new(&device, &adapter, &model, options)?;
    let mut prepared = denoiser.prepare(
        [width, height],
        DenoiseBuffers {
            color: (&color).into(),
            albedo: (model.input_count() >= 2).then_some((&albedo).into()),
            normal: (model.input_count() >= 3).then_some((&normal).into()),
            output: (&output).into(),
        },
    )?;
    println!("Denoise: {:?}", prepared.info());
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("file export"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    // Per frame: produce -> denoise -> consume, without an intermediate submit
    // or CPU readback. A display pass could consume output instead of export.
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bindings, &[]);
        pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
    }
    prepared.encode(&mut encoder, DenoiseParams::default().with_input_scale(1.0))?;
    encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, bytes);
    let submission = queue.submit([encoder.finish()]);

    // This wait and readback are only for saving the example's final image.
    let (send, receive) = std::sync::mpsc::channel();
    readback.map_async(wgpu::MapMode::Read, .., move |result| {
        let _ = send.send(result);
    });
    device.poll(wgpu::PollType::Wait {
        submission_index: Some(submission),
        timeout: Some(Duration::from_secs(60)),
    })?;
    receive.recv_timeout(Duration::from_secs(5))??;
    {
        let mapped = readback.get_mapped_range(..)?;
        for pixel in mapped.as_chunks::<16>().0 {
            if pixel
                .as_chunks::<4>()
                .0
                .iter()
                .any(|v| !f32::from_le_bytes(*v).is_finite())
                || f32::from_le_bytes(pixel[12..16].try_into().unwrap()) != 0.75
            {
                return Err("unexpected output: nonfinite pixel or changed alpha".into());
            }
        }
        std::fs::write(&args[2], &mapped)?;
    }
    readback.unmap();
    println!(
        "{}: saved {width}x{height} RGBA f32 to {}",
        adapter.get_info().name,
        args[2]
    );
    Ok(())
}
