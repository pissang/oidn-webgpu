//! Metal interop feasibility gate, before freezing the native GPU C ABI.
//! Host Metal kernels produce inputs and consume outputs on independent queues.
//! Only validation flags are inspected on the CPU; image pixels stay on the GPU.

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("metal_interop requires macOS and a Metal GPU");
    std::process::exit(1);
}

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    objc2::rc::autoreleasepool(|_| metal::run())
}

#[cfg(target_os = "macos")]
mod metal {
    use objc2::{rc::Retained, runtime::ProtocolObject};
    use objc2_foundation::NSString;
    use objc2_metal::*;
    use oidn_webgpu::{
        DenoiseBuffers, DenoiseParams, Denoiser, DenoiserOptions, ImageBuffer, Model, wgpu,
    };
    use std::{
        error::Error,
        ptr::NonNull,
        sync::{Arc, Mutex},
        time::{Duration, Instant},
    };

    type Device = Retained<ProtocolObject<dyn MTLDevice>>;
    type Queue = Retained<ProtocolObject<dyn MTLCommandQueue>>;
    type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
    type Event = Retained<ProtocolObject<dyn MTLSharedEvent>>;
    type Command = Retained<ProtocolObject<dyn MTLCommandBuffer>>;
    type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
    type Result<T> = std::result::Result<T, Box<dyn Error>>;

    // Padding tests imported views with a nonzero, storage-aligned byte offset.
    const OFFSET: usize = 256;
    const SOURCE: &str = r#"
        #include <metal_stdlib>
        using namespace metal;
        kernel void produce(device float4* color [[buffer(0)]],
                            device float4* albedo [[buffer(1)]],
                            device float4* normal [[buffer(2)]],
                            constant uint4& p [[buffer(3)]],
                            uint i [[thread_position_in_grid]]) {
            if (i >= p.x * p.y) return;
            uint pattern = p.z;
            float noise = float(((i + pattern * 31u) * 1664525u + 1013904223u) >> 24u) / 255.0f - 0.5f;
            float signal = 0.5f + float(i % p.x) / float(p.x) + float(pattern) * 0.25f;
            color[i] = float4(max(float3(0.8f, 0.6f, 0.4f) * signal + noise * 0.3f, 0.0f),
                              0.25f + float(pattern) * 0.125f);
            albedo[i] = float4(0.8f, 0.6f, 0.4f, 0.0f);
            normal[i] = float4(0.0f, 0.0f, 1.0f, 0.0f);
        }
        kernel void consume(device const float4* actual [[buffer(0)]],
                            device const float4* expected [[buffer(1)]],
                            device uint* checks [[buffer(2)]],
                            constant uint4& p [[buffer(3)]],
                            uint i [[thread_position_in_grid]]) {
            uint count = p.x * p.y;
            if (i >= count) return;
            float4 a = actual[i], b = expected[i];
            uint failure = any(as_type<uint4>(a) != as_type<uint4>(b)) ? 1u : 0u;
            if (!all(isfinite(a)) || any(a.xyz < 0.0f)) failure |= 2u;
            if (a.w != 0.25f + float(p.z) * 0.125f) failure |= 4u;
            if (!any(a.xyz > 0.0f)) failure |= 8u;
            checks[p.w * count + i] = failure;
        }
    "#;

    struct Context {
        adapter: wgpu::Adapter,
        device: wgpu::Device,
        queue: wgpu::Queue,
    }

    impl Context {
        /// The caller's device remains owned by the host; both HAL wrappers
        /// retain Metal objects. Select an adapter for that exact physical GPU.
        unsafe fn import(raw: &Device, queue: &Queue, fp32: bool, strict: bool) -> Result<Self> {
            if queue.device().registryID() != raw.registryID() {
                return Err("queue belongs to another Metal device".into());
            }
            let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
            descriptor.backends = wgpu::Backends::METAL;
            let instance = wgpu::Instance::new(descriptor);
            let adapter = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::METAL))
                .into_iter()
                .find(|adapter| unsafe {
                    adapter
                        .as_hal::<wgpu::hal::api::Metal>()
                        .is_some_and(|hal| hal.raw_device().registryID() == raw.registryID())
                })
                .ok_or("host Metal device has no matching wgpu adapter")?;
            let mut features = Denoiser::recommended_features(&adapter);
            if fp32 {
                features.remove(wgpu::Features::SHADER_F16);
            }
            let limits = adapter.limits();
            let hal_device = unsafe {
                wgpu::hal::metal::Device::device_from_raw(raw.clone(), features, &limits)
            };
            let hal_queue = unsafe {
                // No timing queries are enabled here; Metal timestamp units
                // therefore do not affect validation on Intel or Apple GPUs.
                wgpu::hal::metal::Queue::queue_from_raw(queue.clone(), 1.0)
            };
            if strict {
                hal_queue.enable_strict_event_sync()?;
            }
            let (device, queue) = unsafe {
                adapter.create_device_from_hal::<wgpu::hal::api::Metal>(
                    wgpu::hal::OpenDevice {
                        device: hal_device,
                        queue: hal_queue,
                    },
                    &wgpu::DeviceDescriptor {
                        label: Some("imported host Metal device"),
                        required_features: features,
                        required_limits: limits,
                        ..Default::default()
                    },
                )?
            };
            Ok(Self {
                adapter,
                device,
                queue,
            })
        }

        unsafe fn buffer(&self, raw: &Buffer) -> Result<wgpu::Buffer> {
            let device = unsafe { self.device.as_hal::<wgpu::hal::api::Metal>() }
                .ok_or("not a Metal device")?;
            if device.raw_device().registryID() != raw.device().registryID() {
                return Err("buffer belongs to another Metal device".into());
            }
            let size = raw.length() as u64;
            let hal = unsafe { wgpu::hal::metal::Device::buffer_from_raw(raw.clone(), size) };
            Ok(unsafe {
                self.device.create_buffer_from_hal::<wgpu::hal::api::Metal>(
                    hal,
                    &wgpu::BufferDescriptor {
                        label: Some("borrowed host Metal image"),
                        size,
                        usage: wgpu::BufferUsages::STORAGE,
                        mapped_at_creation: false,
                    },
                )
            })
        }

        fn wait(&self, submission: wgpu::SubmissionIndex) -> Result<()> {
            self.device.poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: Some(Duration::from_secs(60)),
            })?;
            Ok(())
        }
    }

    struct Images {
        raw: Vec<Buffer>,
        imported: Vec<wgpu::Buffer>,
    }

    impl Images {
        fn new(raw: &Device, context: &Context, bytes: usize) -> Result<Self> {
            let raw = (0..4)
                .map(|_| {
                    raw.newBufferWithLength_options(
                        bytes + OFFSET,
                        MTLResourceOptions::StorageModePrivate,
                    )
                    .ok_or_else(|| "Metal buffer allocation failed".into())
                })
                .collect::<Result<Vec<_>>>()?;
            let imported = raw
                .iter()
                .map(|buffer| unsafe { context.buffer(buffer) })
                .collect::<Result<Vec<_>>>()?;
            Ok(Self { raw, imported })
        }

        fn bindings<'a>(&'a self, output: Option<&'a wgpu::Buffer>) -> DenoiseBuffers<'a> {
            let view = |buffer| ImageBuffer {
                buffer,
                offset: OFFSET as u64,
            };
            DenoiseBuffers {
                color: view(&self.imported[0]),
                albedo: Some(view(&self.imported[1])),
                normal: Some(view(&self.imported[2])),
                output: view(output.unwrap_or(&self.imported[3])),
            }
        }
    }

    fn pipeline(device: &Device, name: &str) -> Result<Pipeline> {
        let library =
            device.newLibraryWithSource_options_error(&NSString::from_str(SOURCE), None)?;
        let function = library
            .newFunctionWithName(&NSString::from_str(name))
            .ok_or("missing host kernel")?;
        Ok(device.newComputePipelineStateWithFunction_error(&function)?)
    }

    fn command(queue: &Queue) -> Result<Command> {
        queue
            .commandBuffer()
            .ok_or_else(|| "Metal command buffer allocation failed".into())
    }

    fn dispatch(
        command: &Command,
        pipeline: &Pipeline,
        buffers: &[(&Buffer, usize)],
        parameters: [u32; 4],
    ) -> Result<()> {
        let encoder = command
            .computeCommandEncoder()
            .ok_or("Metal encoder allocation failed")?;
        encoder.setComputePipelineState(pipeline);
        for (index, (buffer, offset)) in buffers.iter().enumerate() {
            // Buffers are retained by Images through GPU completion; every
            // binding's byte range and the kernel's grid size are checked here.
            unsafe { encoder.setBuffer_offset_atIndex(Some(buffer), *offset, index) };
        }
        unsafe {
            encoder.setBytes_length_atIndex(NonNull::from(&parameters).cast(), 16, buffers.len());
        }
        encoder.dispatchThreads_threadsPerThreadgroup(
            MTLSize {
                width: parameters[0] as usize * parameters[1] as usize,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 64,
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();
        Ok(())
    }

    fn wait_command(command: &Command) -> Result<()> {
        let deadline = Instant::now() + Duration::from_secs(60);
        loop {
            match command.status() {
                MTLCommandBufferStatus::Completed => return Ok(()),
                MTLCommandBufferStatus::Error => {
                    return Err(format!("host Metal command failed: {:?}", command.error()).into());
                }
                _ if Instant::now() >= deadline => {
                    return Err("host Metal command timed out".into());
                }
                _ => std::thread::sleep(Duration::from_millis(1)),
            }
        }
    }

    fn event(device: &Device) -> Result<Event> {
        device
            .newSharedEvent()
            .ok_or_else(|| "Metal shared events unavailable".into())
    }

    pub fn run() -> Result<()> {
        let args: Vec<String> = std::env::args().skip(1).collect();
        if args.iter().any(|a| a == "--help") {
            println!(
                "metal_interop [--frames N] [--size N] [--weights PATH] [--shared-queue] [--fp32] [--tile N] [--staged-waits] [--trace]\nDefaults: 1000 frames, 16x16, Small HDR model, dedicated queue, FP16 if available, whole image\n--staged-waits disables strict gating for diagnostics; --trace prints CPU submission timestamps"
            );
            return Ok(());
        }
        let mut index = 0;
        while index < args.len() {
            match args[index].as_str() {
                "--frames" | "--size" | "--weights" | "--tile" => index += 2,
                "--shared-queue" | "--fp32" | "--staged-waits" | "--trace" => index += 1,
                other => return Err(format!("unknown argument: {other}").into()),
            }
        }
        let option = |name: &str| -> Result<Option<&str>> {
            args.iter()
                .position(|a| a == name)
                .map(|index| {
                    args.get(index + 1)
                        .filter(|v| !v.starts_with("--"))
                        .map(String::as_str)
                        .ok_or_else(|| format!("{name} requires a value").into())
                })
                .transpose()
        };
        let frames = option("--frames")?.unwrap_or("1000").parse::<usize>()?;
        let size = option("--size")?.unwrap_or("16").parse::<u32>()?;
        let tile = option("--tile")?.unwrap_or("0").parse::<u32>()?;
        if frames == 0 || frames > 100_000 || !(1..=1024).contains(&size) {
            return Err("frames must be 1..=100000; size must be 1..=1024".into());
        }
        let count = size as usize * size as usize;
        let report_bytes = frames
            .checked_mul(count)
            .and_then(|n| n.checked_mul(4))
            .filter(|n| *n <= 256 * 1024 * 1024)
            .ok_or("validation flags exceed 256 MiB")?;
        let model_path = option("--weights")?.unwrap_or(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../weights/rt_hdr_alb_nrm_small.tza"
        ));
        let model = Model::from_tza(&std::fs::read(model_path)?)?;
        if model.input_count() != 3 {
            return Err("this test requires a color/albedo/normal model".into());
        }
        let raw = MTLCreateSystemDefaultDevice().ok_or("no Metal GPU")?;
        // Match wgpu-hal 30.0.1's MAX_COMMAND_BUFFERS. Metal's default queue
        // capacity can block submit while the first producer is gated, before
        // enough work has been enqueued to release that gate (notably tiling).
        let producer = raw
            .newCommandQueueWithMaxCommandBufferCount(4096)
            .ok_or("producer queue allocation failed")?;
        let consumer = raw
            .newCommandQueueWithMaxCommandBufferCount(4096)
            .ok_or("consumer queue allocation failed")?;
        let shared = args.iter().any(|a| a == "--shared-queue");
        let library_queue = if shared {
            producer.clone()
        } else {
            raw.newCommandQueueWithMaxCommandBufferCount(4096)
                .ok_or("library queue allocation failed")?
        };
        // The example has one submitting CPU thread. Shared-queue integrations
        // must serialize host submissions with library queue access and teardown.
        let context = unsafe {
            Context::import(
                &raw,
                &library_queue,
                args.iter().any(|a| a == "--fp32"),
                !args.iter().any(|a| a == "--staged-waits"),
            )?
        };
        let errors = Arc::new(Mutex::new(Vec::new()));
        let callback_errors = errors.clone();
        context.device.on_uncaptured_error(Arc::new(move |error| {
            callback_errors.lock().unwrap().push(error.to_string())
        }));
        let lost_errors = errors.clone();
        context
            .device
            .set_device_lost_callback(move |reason, message| {
                lost_errors
                    .lock()
                    .unwrap()
                    .push(format!("device lost ({reason:?}): {message}"));
            });
        let produce = pipeline(&raw, "produce")?;
        let consume = pipeline(&raw, "consume")?;
        let ring = (0..3)
            .map(|_| Images::new(&raw, &context, count * 16))
            .collect::<Result<Vec<_>>>()?;
        let references = Images::new(&raw, &context, count * 16)?;
        let ready = event(&raw)?;
        let done = event(&raw)?;
        let consumed = event(&raw)?;
        let gate = event(&raw)?;
        let checks = raw
            .newBufferWithLength_options(report_bytes, MTLResourceOptions::StorageModeShared)
            .ok_or("validation buffer allocation failed")?;
        // Validation flags only; image allocations are private and never CPU mapped.
        unsafe { checks.contents().as_ptr().write_bytes(0xff, report_bytes) };
        let mut options = DenoiserOptions::default();
        options.max_tile_size = tile;
        let mut denoiser = Denoiser::new(&context.device, &context.adapter, &model, options)?;
        let mut prepared = denoiser.prepare([size, size], ring[0].bindings(None))?;
        let scales = [1.0, 2.0, 0.5];

        // Serialized references use the existing Rust API. Output remains in
        // three host GPU buffers; no reference image is downloaded to the CPU.
        for pattern in 0..3 {
            let cb = command(&producer)?;
            dispatch(
                &cb,
                &produce,
                &[
                    (&ring[0].raw[0], OFFSET),
                    (&ring[0].raw[1], OFFSET),
                    (&ring[0].raw[2], OFFSET),
                ],
                [size, size, pattern as u32, 0],
            )?;
            cb.commit();
            wait_command(&cb).map_err(|error| format!("reference producer {pattern}: {error}"))?;
            prepared.rebind(ring[0].bindings(Some(&references.imported[pattern])))?;
            let mut encoder = context.device.create_command_encoder(&Default::default());
            prepared.encode(
                &mut encoder,
                DenoiseParams {
                    input_scale: scales[pattern],
                },
            )?;
            context.wait(context.queue.submit([encoder.finish()]))?;
            let observed = errors.lock().unwrap();
            if !observed.is_empty() {
                return Err(format!("reference {pattern} failed: {observed:?}").into());
            }
        }

        // Block the first native producer until several denoise executions
        // have been enqueued. Strict HAL waits must protect untracked foreign
        // buffers even while native queues execute independent workloads.
        let release_gate = gate.clone();
        let release_done = done.clone();
        let (gate_send, gate_receive) = std::sync::mpsc::channel();
        let release = std::thread::spawn(move || {
            let submitted = gate_receive.recv_timeout(Duration::from_secs(10)).is_ok();
            std::thread::sleep(Duration::from_millis(100));
            let completed_early = release_done.signaledValue();
            release_gate.setSignaledValue(1);
            (submitted, completed_early)
        });
        let started = Instant::now();
        let mut last_consumer = None;
        let mut last_submit = None;
        for frame in 0..frames {
            objc2::rc::autoreleasepool(|_| -> Result<()> {
                context.device.poll(wgpu::PollType::Poll)?;
                let slot = frame % 3;
                let sequence = frame as u64 + 1;
                let images = &ring[slot];
                let cb = command(&producer)?;
                // A wait in just the first CB does not gate independent later
                // CBs on Metal. Chain the host timeline explicitly as well as
                // gating every producer, so ready=N implies all earlier writes.
                cb.encodeWaitForEvent_value(ProtocolObject::from_ref(&*gate), 1);
                if frame != 0 {
                    cb.encodeWaitForEvent_value(ProtocolObject::from_ref(&*ready), sequence - 1);
                }
                if frame >= 3 {
                    cb.encodeWaitForEvent_value(ProtocolObject::from_ref(&*consumed), sequence - 3);
                }
                dispatch(
                    &cb,
                    &produce,
                    &[
                        (&images.raw[0], OFFSET),
                        (&images.raw[1], OFFSET),
                        (&images.raw[2], OFFSET),
                    ],
                    [size, size, slot as u32, 0],
                )?;
                cb.encodeSignalEvent_value(ProtocolObject::from_ref(&*ready), sequence);
                cb.commit();

                if args.iter().any(|a| a == "--trace") {
                    eprintln!("frame {frame} before rebind at {:?}", started.elapsed());
                }
                prepared.rebind(images.bindings(None))?;
                if args.iter().any(|a| a == "--trace") {
                    eprintln!("frame {frame} after rebind at {:?}", started.elapsed());
                }
                let mut encoder = context.device.create_command_encoder(&Default::default());
                prepared.encode(
                    &mut encoder,
                    DenoiseParams {
                        input_scale: scales[slot],
                    },
                )?;
                let hal = unsafe { context.queue.as_hal::<wgpu::hal::api::Metal>() }
                    .ok_or("not a Metal queue")?;
                hal.add_wait_event(ready.clone(), sequence);
                hal.add_signal_event(done.clone(), sequence);
                last_submit = Some(context.queue.submit([encoder.finish()]));
                if args.iter().any(|a| a == "--trace") {
                    eprintln!("frame {frame} submitted at {:?}", started.elapsed());
                }
                if frame + 1 == frames.min(3) {
                    gate_send.send(())?;
                }

                let cb = command(&consumer)?;
                if frame != 0 {
                    cb.encodeWaitForEvent_value(ProtocolObject::from_ref(&*consumed), sequence - 1);
                }
                cb.encodeWaitForEvent_value(ProtocolObject::from_ref(&*done), sequence);
                dispatch(
                    &cb,
                    &consume,
                    &[
                        (&images.raw[3], OFFSET),
                        (&references.raw[slot], OFFSET),
                        (&checks, 0),
                    ],
                    [size, size, slot as u32, frame as u32],
                )?;
                cb.encodeSignalEvent_value(ProtocolObject::from_ref(&*consumed), sequence);
                cb.commit();
                last_consumer = Some(cb);
                Ok(())
            })?;
        }
        let (submitted, completed_early) =
            release.join().map_err(|_| "gate release thread failed")?;
        if !submitted {
            return Err("GPU submission stalled before the producer gate was released".into());
        }
        if completed_early != 0 {
            return Err(format!(
                "denoising signalled {completed_early} before its producer was released"
            )
            .into());
        }
        context.wait(last_submit.unwrap())?;
        wait_command(last_consumer.as_ref().unwrap()).map_err(|error| {
            format!(
                "async consumer: {error}; ready={}, done={}, consumed={}, gate={}, errors={:?}",
                ready.signaledValue(),
                done.signaledValue(),
                consumed.signaledValue(),
                gate.signaledValue(),
                errors.lock().unwrap()
            )
        })?;
        let errors = errors.lock().unwrap();
        if !errors.is_empty() {
            return Err(format!("wgpu errors: {errors:?}").into());
        }
        let report = unsafe {
            std::slice::from_raw_parts(checks.contents().as_ptr().cast::<u32>(), frames * count)
        };
        if let Some((index, value)) = report.iter().enumerate().find(|(_, value)| **value != 0) {
            return Err(format!(
                "frame {} pixel {} failed (flags {value:#x})",
                index / count,
                index % count
            )
            .into());
        }
        println!(
            "{}",
            serde_json::json!({
                "backend": "Metal", "adapter": context.adapter.get_info().name,
                "queue": if shared { "shared" } else { "dedicated" },
                "size": size, "frames": frames, "tile": tile,
                "precision": format!("{:?}", prepared.info().precision),
                "runtime_checks": prepared.info().runtime_checks,
                "wait_mode": if args.iter().any(|a| a == "--staged-waits") { "staged" } else { "hal-relay" },
            "strict_event_sync": !args.iter().any(|a| a == "--staged-waits"), "output_byte_equal": true,
            "completed_before_producer_release": completed_early,
                "cpu_image_uploads": 0, "cpu_image_readbacks": 0,
                "elapsed_ms": started.elapsed().as_secs_f64() * 1000.0
            })
        );
        Ok(())
    }
}
