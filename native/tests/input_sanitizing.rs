//! OIDN-style sanitizing of native denoiser inputs and outputs: images with
//! NaN, infinities and out-of-range values must denoise bit-identically to the
//! same images sanitized on the CPU, and outputs must be nonnegative (at most 1
//! for LDR). HDR outputs may be +inf where +inf input saturates the transfer.
//! Alignment padding must also match an explicitly zero-padded network input.
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

#[test]
#[ignore = "requires a native GPU adapter"]
fn unaligned_images_denoise_like_explicit_zero_padding() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../weights/rt_hdr_alb_nrm_small.tza"
    );
    let model = Model::from_tza(&std::fs::read(path).unwrap()).unwrap();
    let required = Denoiser::required_features();
    let mut feature_sets = vec![required];
    if adapter.features().contains(wgpu::Features::SHADER_F16) {
        feature_sets.push(required | wgpu::Features::SHADER_F16);
    }
    for features in feature_sets {
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: features,
            required_limits: Denoiser::required_limits(),
            ..Default::default()
        }))
        .unwrap();
        for [width, height] in [[256_u32, 71], [71, 256]] {
            let run = |size: [u32; 2], max_tile_size| {
                let [w, h] = size;
                let mut images = [Vec::new(), Vec::new(), Vec::new()];
                for y in 0..h {
                    for x in 0..w {
                        let pixels = if x < width && y < height {
                            [
                                [0.25 + x as f32 / 256.0, 0.5 + y as f32 / 256.0, 0.75, ALPHA],
                                [0.3, 0.4, 0.5, 1.0],
                                [-0.3, 0.1, 0.8, 1.0],
                            ]
                        } else {
                            // Signed -1 maps to zero after the normal transfer.
                            [[0.0; 4], [0.0; 4], [-1.0, -1.0, -1.0, 0.0]]
                        };
                        for (image, pixel) in images.iter_mut().zip(pixels) {
                            image.extend(pixel);
                        }
                    }
                }
                let [color, albedo, normal] = images
                    .map(|image| upload(&device, &queue, &image, wgpu::BufferUsages::empty()));
                let output = upload(
                    &device,
                    &queue,
                    &vec![f32::NAN; (w * h * 4) as usize],
                    wgpu::BufferUsages::COPY_SRC,
                );
                let mut options = DenoiserOptions::default();
                options.max_tile_size = max_tile_size;
                let mut denoiser = Denoiser::new(&device, &adapter, &model, options).unwrap();
                let mut prepared = denoiser
                    .prepare(
                        size,
                        DenoiseBuffers {
                            color: (&color).into(),
                            albedo: Some((&albedo).into()),
                            normal: Some((&normal).into()),
                            output: (&output).into(),
                        },
                    )
                    .unwrap();
                assert_eq!(prepared.size(), size);
                assert_eq!(
                    prepared.info().tile_count,
                    if max_tile_size == 0 { 1 } else { 2 }
                );
                let mut encoder = device.create_command_encoder(&Default::default());
                prepared
                    .encode(&mut encoder, DenoiseParams::default().with_input_scale(0.5))
                    .unwrap();
                queue.submit([encoder.finish()]);
                download(&device, &queue, &output)
            };
            let aligned = [width.next_multiple_of(16), height.next_multiple_of(16)];
            let whole = run([width, height], 0);
            let tiled = run([width, height], 128);
            for (tile_size, actual) in [(0, &whole), (128, &tiled)] {
                let expected = run(aligned, tile_size);
                for y in 0..height {
                    for x in 0..width {
                        let i = ((y * width + x) * 4) as usize;
                        let j = ((y * aligned[0] + x) * 4) as usize;
                        let pixel = &actual[i..i + 4];
                        assert_eq!(
                            pixel,
                            &expected[j..j + 4],
                            "{width}x{height} tile {tile_size} pixel ({x}, {y})"
                        );
                        assert!(pixel.iter().all(|v| v.is_finite()));
                        assert_eq!(pixel[3], ALPHA);
                        let seam_axis = if width > height { x } else { y };
                        if tile_size != 0 && seam_axis.abs_diff(128) < 4 {
                            for (a, b) in whole[i..i + 3].iter().zip(&tiled[i..i + 3]) {
                                assert!(
                                    (a - b).abs() < 1e-5,
                                    "tile seam at ({x}, {y}): {a} vs {b}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}
