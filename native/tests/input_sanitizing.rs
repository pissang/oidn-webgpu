//! OIDN-style sanitizing of native denoiser inputs and outputs: images with
//! NaN, infinities and out-of-range values must denoise bit-identically to the
//! same images sanitized on the CPU, and outputs must be nonnegative (at most 1
//! for LDR). HDR outputs may be +inf where +inf input saturates the transfer.
//! Requires a GPU: cargo test --manifest-path native/Cargo.toml -- --ignored
use oidn_webgpu::{DenoiseBuffers, DenoiseParams, Denoiser, DenoiserOptions, Model, wgpu};
use std::time::Duration;

const WIDTH: u32 = 37;
const HEIGHT: u32 = 29;
const ALPHA: f32 = 0.75;

fn sanitize(value: f32, lo: f32, hi: f32) -> f32 {
    if value.is_nan() { 0.0 } else { value.clamp(lo, hi) }
}

/// A smooth RGBA image with a special value in every 7th channel value, and
/// the same image sanitized to [lo, hi].
fn image(lo: f32, hi: f32, base: impl Fn(u32, u32, u32) -> f32) -> (Vec<f32>, Vec<f32>) {
    let over = if hi == 1.0 { 1.5 } else { 7.0 };
    let special = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, lo - 2.0, over];
    let mut dirty = Vec::new();
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            for channel in 0..3 {
                let index = (y * WIDTH + x) * 3 + channel;
                dirty.push(if index % 7 == 0 {
                    special[(index / 7) as usize % special.len()]
                } else {
                    base(x, y, channel)
                });
            }
            dirty.push(ALPHA);
        }
    }
    let clean = dirty
        .chunks(4)
        .flat_map(|pixel| [sanitize(pixel[0], lo, hi), sanitize(pixel[1], lo, hi), sanitize(pixel[2], lo, hi), pixel[3]])
        .collect();
    (dirty, clean)
}

fn upload(device: &wgpu::Device, queue: &wgpu::Queue, data: &[f32], usage: wgpu::BufferUsages) -> wgpu::Buffer {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (data.len() * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | usage,
        mapped_at_creation: false,
    });
    let bytes: Vec<u8> = data.iter().flat_map(|value| value.to_le_bytes()).collect();
    queue.write_buffer(&buffer, 0, &bytes);
    buffer
}

fn download(device: &wgpu::Device, queue: &wgpu::Queue, buffer: &wgpu::Buffer) -> Vec<f32> {
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: buffer.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &readback, 0, buffer.size());
    let submission = queue.submit([encoder.finish()]);
    let (send, receive) = std::sync::mpsc::channel();
    readback.map_async(wgpu::MapMode::Read, .., move |result| {
        let _ = send.send(result);
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: Some(Duration::from_secs(60)),
        })
        .unwrap();
    receive.recv_timeout(Duration::from_secs(5)).unwrap().unwrap();
    let mapped = readback.get_mapped_range(..).unwrap();
    mapped.as_chunks::<4>().0.iter().map(|bytes| f32::from_le_bytes(*bytes)).collect()
}

#[test]
#[ignore = "requires a native GPU adapter"]
fn dirty_inputs_denoise_like_sanitized_inputs() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .unwrap();
    let recommended = Denoiser::required_features() | Denoiser::recommended_features(&adapter);
    let mut feature_sets = vec![Denoiser::required_features()];
    if recommended != feature_sets[0] {
        feature_sets.push(recommended);
    }
    for features in feature_sets {
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: features,
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .unwrap();
        for (weights, hdr) in [("rt_hdr_alb_nrm_small.tza", true), ("rt_ldr.tza", false)] {
            let path = format!("{}/../weights/{weights}", env!("CARGO_MANIFEST_DIR"));
            let model = Model::from_tza(&std::fs::read(path).unwrap()).unwrap();
            let color_hi = if hdr { f32::MAX } else { 1.0 };
            let images = [
                image(0.0, color_hi, |x, y, c| 0.2 + 0.7 * ((x * 3 + y * 5 + c * 7) % 17) as f32 / 17.0),
                image(0.0, 1.0, |x, _, c| 0.3 + 0.08 * ((x + c) % 5) as f32),
                image(-1.0, 1.0, |x, y, c| if c == 2 { 0.8 } else { 0.5 * ((x + y + c) as f32 * 0.3).sin() }),
            ];
            let mut options = DenoiserOptions::default();
            options.hdr = hdr;
            let mut denoiser = Denoiser::new(&device, &adapter, &model, options).unwrap();
            let mut run = |dirty: bool| {
                let inputs: Vec<_> = images[..model.input_count()]
                    .iter()
                    .map(|(d, c)| upload(&device, &queue, if dirty { d } else { c }, wgpu::BufferUsages::empty()))
                    .collect();
                let output = upload(&device, &queue, &vec![0.0; (WIDTH * HEIGHT * 4) as usize], wgpu::BufferUsages::COPY_SRC);
                let mut prepared = denoiser
                    .prepare(
                        [WIDTH, HEIGHT],
                        DenoiseBuffers {
                            color: (&inputs[0]).into(),
                            albedo: inputs.get(1).map(Into::into),
                            normal: inputs.get(2).map(Into::into),
                            output: (&output).into(),
                        },
                    )
                    .unwrap();
                let mut encoder = device.create_command_encoder(&Default::default());
                prepared.encode(&mut encoder, DenoiseParams::default()).unwrap();
                queue.submit([encoder.finish()]);
                download(&device, &queue, &output)
            };
            let dirty = run(true);
            let clean = run(false);
            for (index, (&a, &b)) in dirty.iter().zip(&clean).enumerate() {
                let what = format!("{weights} {features:?} value {index}");
                assert_eq!(a.to_bits(), b.to_bits(), "{what}: {a} vs {b}");
                if index % 4 == 3 {
                    assert_eq!(a, ALPHA, "{what}: alpha");
                } else {
                    assert!(a >= 0.0 && (hdr || a <= 1.0), "{what}: {a}");
                }
            }
        }
    }
}
