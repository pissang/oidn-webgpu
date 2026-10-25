//! Regression coverage for matrix staging without shader runtime checks.
use oidn_webgpu::{DenoiseConfig, Denoiser, Kernel, Network, NetworkConfig, wgpu};
use std::time::Duration;

fn upload_gradient(queue: &wgpu::Queue, network: &Network, width: u32, height: u32) {
    for (source, buffer) in network.input_buffers().enumerate() {
        let mut data = Vec::new();
        for y in 0..height {
            for x in 0..width {
                let fx = x as f32 / (width - 1) as f32;
                let fy = y as f32 / (height - 1) as f32;
                let rgba = match source {
                    0 => [0.1 + 4.0 * fx, 0.05 + 2.0 * fy, 0.2 + ((x * 17 + y * 13) % 97) as f32 / 97.0, 1.0],
                    1 => [0.2 + 0.7 * fx, 0.3 + 0.6 * fy, 0.55, 1.0],
                    _ => [0.0, 0.0, 1.0, 1.0],
                };
                data.extend(rgba.into_iter().flat_map(f32::to_le_bytes));
            }
        }
        queue.write_buffer(buffer, 0, &data);
    }
}

fn execute(device: &wgpu::Device, queue: &wgpu::Queue, network: &Network) -> Vec<u8> {
    let output = network.output_buffer();
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("matrix regression readback"),
        size: output.size(),
        mapped_at_creation: false,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    network.encode(queue, &mut encoder);
    encoder.copy_buffer_to_buffer(output, 0, &staging, 0, output.size());
    let submission = queue.submit([encoder.finish()]);
    let (send, receive) = std::sync::mpsc::channel();
    staging.map_async(wgpu::MapMode::Read, .., move |result| { let _ = send.send(result); });
    device.poll(wgpu::PollType::Wait {
        submission_index: Some(submission),
        timeout: Some(Duration::from_secs(60)),
    }).unwrap();
    receive.recv_timeout(Duration::from_secs(5)).unwrap().unwrap();
    staging.get_mapped_range(..).unwrap().to_vec()
}

#[test]
#[ignore = "requires Vulkan FP16 cooperative matrices"]
fn large_unchecked_is_finite_and_matches_checked() {
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = wgpu::Backends::VULKAN;
    let instance = wgpu::Instance::new(descriptor);
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    })).unwrap();
    let required = wgpu::Features::SHADER_F16 | wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX;
    assert!(adapter.features().contains(required), "adapter lacks FP16 cooperative matrices");
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features: required | Denoiser::recommended_features(&adapter),
        required_limits: adapter.limits(),
        // This test deliberately exercises wgpu's experimental matrix API.
        experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
        ..Default::default()
    })).unwrap();
    eprintln!("matrix regression adapter: {:?}", adapter.get_info());
    let weights = std::fs::read(format!("{}/../weights/rt_hdr_calb_cnrm_large.tza", env!("CARGO_MANIFEST_DIR"))).unwrap();
    // Cover partial tiles as well as the size that reproduced NaN/Inf on RTX 4060.
    for (width, height) in [(256, 256), (257, 259)] {
        for sub_pixel in [true, false] {
            let config = NetworkConfig {
                kernel: Kernel::SubgroupMatrix,
                sub_pixel,
                denoise: Some(DenoiseConfig { hdr: true, log_transfer: false, max_tile_size: 0, input_scale: 1.0 }),
                ..NetworkConfig::new(width, height)
            };
            let checked = Network::new(&device, &adapter, &weights, &config, None).unwrap();
            // The regression verifies the generated kernels' unchecked execution.
            let unchecked = unsafe { Network::new_unchecked(&device, &adapter, &weights, &config, None) }.unwrap();
            upload_gradient(&queue, &checked, width, height);
            upload_gradient(&queue, &unchecked, width, height);
            let expected = execute(&device, &queue, &checked);
            assert!(expected.chunks_exact(4).all(|v| f32::from_le_bytes(v.try_into().unwrap()).is_finite()));
            for repeat in 0..4 {
                let actual = execute(&device, &queue, &unchecked);
                assert!(actual.chunks_exact(4).all(|v| f32::from_le_bytes(v.try_into().unwrap()).is_finite()),
                    "non-finite output: {width}x{height}, sub_pixel={sub_pixel}, repeat={repeat}");
                assert!(actual == expected,
                    "checked/unchecked output differs: {width}x{height}, sub_pixel={sub_pixel}, repeat={repeat}");
            }
        }
    }
}
