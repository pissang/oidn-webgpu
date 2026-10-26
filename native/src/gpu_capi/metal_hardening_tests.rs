use super::*;
use objc2::rc::{Retained, autoreleasepool};
use objc2_metal::{MTLBuffer, MTLCreateSystemDefaultDevice, MTLDevice, MTLResourceOptions};

#[test]
#[ignore = "requires an Apple silicon Metal GPU"]
fn metal_matrix_defaults_info_scale_and_cpu_equivalence() {
    autoreleasepool(|_| unsafe {
        let weights = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../weights/rt_hdr_alb_nrm_small.tza"
        ))
        .unwrap();
        let raw = MTLCreateSystemDefaultDevice().expect("Metal GPU required");
        let (width, height) = (257u32, 259u32);
        let count = (width * height * 4) as usize;
        let bytes = (count * 4) as u64;
        let inputs: Vec<Vec<f32>> = (0..3)
            .map(|kind| {
                (0..count)
                    .map(|i| {
                        if i % 4 == 3 {
                            0.75
                        } else {
                            let value = ((i * 17 + kind * 31) % 251) as f32 / 251.0;
                            if kind == 2 { value * 2.0 - 1.0 } else { value }
                        }
                    })
                    .collect()
            })
            .collect();
        let mut cpu_device = std::ptr::null_mut();
        let mut message = [0i8; 1024];
        let message_ptr = message.as_ptr();
        let ok = |code| {
            assert_eq!(
                code,
                OIDNW_OK,
                "{}",
                std::ffi::CStr::from_ptr(message_ptr).to_string_lossy()
            )
        };
        ok(oidnw_device_create(
            OIDNW_BACKEND_METAL,
            &mut cpu_device,
            std::ptr::null_mut(),
            0,
        ));
        for (device_flags, kernel, checks, quality) in [
            (0, OIDNW_KERNEL_AUTO, true, OIDNW_QUALITY_HIGH),
            (0, OIDNW_KERNEL_AUTO, false, OIDNW_QUALITY_HIGH),
            (4, OIDNW_KERNEL_AUTO, true, OIDNW_QUALITY_HIGH),
            (0, OIDNW_KERNEL_IMPLICIT_GEMM, true, OIDNW_QUALITY_HIGH),
            (0, OIDNW_KERNEL_AUTO, true, OIDNW_QUALITY_BALANCED),
            (0, OIDNW_KERNEL_AUTO, true, OIDNW_QUALITY_FAST),
            (0, OIDNW_KERNEL_SUBGROUP_MATRIX, true, OIDNW_QUALITY_HIGH),
            (4, OIDNW_KERNEL_DIRECT, true, OIDNW_QUALITY_HIGH),
        ] {
            let mut desc: MetalDeviceDesc = std::mem::zeroed();
            oidnw_gpu_metal_device_desc_default(&mut desc);
            desc.device = Retained::as_ptr(&raw).cast_mut().cast();
            desc.flags = device_flags;
            let mut device = std::ptr::null_mut();
            ok(oidnw_gpu_device_import_metal(
                &desc,
                &mut device,
                std::ptr::null_mut(),
                0,
            ));
            let mut enabled = 0;
            ok(oidnw_gpu_device_info(
                device,
                std::ptr::null_mut(),
                0,
                std::ptr::null_mut(),
                &mut enabled,
            ));
            assert_eq!(
                enabled & OIDNW_FEATURE_COOPERATIVE_MATRIX != 0,
                device_flags & 4 == 0
            );
            let raw_buffers: Vec<_> = (0..4)
                .map(|_| {
                    raw.newBufferWithLength_options(
                        bytes as usize,
                        MTLResourceOptions::StorageModeShared,
                    )
                    .unwrap()
                })
                .collect();
            let mut handles = Vec::new();
            for (i, buffer) in raw_buffers.iter().enumerate() {
                if i < 3 {
                    std::ptr::copy_nonoverlapping(
                        inputs[i].as_ptr(),
                        buffer.contents().as_ptr().cast::<f32>(),
                        count,
                    );
                }
                let mut handle = std::ptr::null_mut();
                ok(oidnw_gpu_buffer_import_metal(
                    device,
                    Retained::as_ptr(buffer).cast_mut().cast(),
                    bytes,
                    &mut handle,
                    std::ptr::null_mut(),
                    0,
                ));
                handles.push(handle);
            }
            let images = Images {
                color: Image {
                    buffer: handles[0],
                    offset: 0,
                },
                albedo: Image {
                    buffer: handles[1],
                    offset: 0,
                },
                normal: Image {
                    buffer: handles[2],
                    offset: 0,
                },
                output: Image {
                    buffer: handles[3],
                    offset: 0,
                },
            };
            let mut model = std::ptr::null_mut();
            ok(oidnw_gpu_model_create(
                weights.as_ptr().cast(),
                weights.len(),
                &mut model,
                std::ptr::null_mut(),
                0,
            ));
            let mut options: FilterDesc = std::mem::zeroed();
            oidnw_gpu_filter_desc_default(&mut options);
            options.flags = if checks {
                OIDNW_FILTER_RUNTIME_CHECKS
            } else {
                0
            };
            options.kernel = kernel;
            options.quality = quality;
            let mut filter = std::ptr::null_mut();
            ok(oidnw_gpu_filter_create(
                device,
                model,
                &options,
                &mut filter,
                std::ptr::null_mut(),
                0,
            ));
            if device_flags == 4 {
                let mut unavailable = std::ptr::null_mut();
                options.kernel = OIDNW_KERNEL_SUBGROUP_MATRIX;
                assert_eq!(
                    oidnw_gpu_filter_create(
                        device,
                        model,
                        &options,
                        &mut unavailable,
                        std::ptr::null_mut(),
                        0
                    ),
                    OIDNW_ERROR_UNAVAILABLE
                );
                assert!(unavailable.is_null());
            }
            assert_eq!(
                oidnw_gpu_filter_warm_up(filter, 0, height, std::ptr::null_mut(), 0),
                OIDNW_ERROR_INVALID_ARGUMENT
            );
            {
                let _held = (&(*filter).gate).lock().unwrap();
                assert_eq!(
                    oidnw_gpu_filter_warm_up(filter, width, height, std::ptr::null_mut(), 0),
                    OIDNW_ERROR_BUSY
                );
            }
            ok(oidnw_gpu_filter_warm_up(
                filter,
                width,
                height,
                std::ptr::null_mut(),
                0,
            ));
            let warmed_count = (&(*filter).gate).lock().unwrap().cached_pipeline_count();
            assert!(warmed_count > 0);
            assert!((&(*filter).gate).lock().unwrap().weights_are_unallocated());
            let mut prepared = std::ptr::null_mut();
            ok(oidnw_gpu_prepare(
                filter,
                width,
                height,
                &images,
                &mut prepared,
                message.as_mut_ptr(),
                message.len(),
            ));
            assert_eq!(
                (&(*filter).gate).lock().unwrap().cached_pipeline_count(),
                warmed_count
            );
            ok(oidnw_gpu_filter_warm_up(
                filter,
                width,
                height,
                std::ptr::null_mut(),
                0,
            ));
            assert_eq!(
                (&(*filter).gate).lock().unwrap().cached_pipeline_count(),
                warmed_count
            );
            let mut info: PreparedInfo = std::mem::zeroed();
            // Holding the submit gate must not affect immutable queries.
            {
                let _held = (&(*prepared).inner).gate.lock().unwrap();
                ok(oidnw_gpu_prepared_get_info(
                    prepared,
                    &mut info,
                    std::ptr::null_mut(),
                    0,
                ));
            }
            let matrix = device_flags == 0
                && (kernel == OIDNW_KERNEL_AUTO || kernel == OIDNW_KERNEL_SUBGROUP_MATRIX);
            assert_eq!(info.kernels & 4 != 0, matrix);
            assert_eq!(info.runtime_checks != 0, checks);
            assert!(info.workspace_bytes > 0 && info.shared_weight_bytes > 0);
            let mut cpu_desc: OidnwFilterDesc = std::mem::zeroed();
            oidnw_filter_desc_default(&mut cpu_desc);
            cpu_desc.kernel = if kernel != OIDNW_KERNEL_AUTO {
                kernel
            } else if matrix {
                OIDNW_KERNEL_SUBGROUP_MATRIX
            } else {
                OIDNW_KERNEL_IMPLICIT_GEMM
            };
            cpu_desc.quality = quality;
            cpu_desc.flags = options.flags;
            let mut cpu = std::ptr::null_mut();
            ok(oidnw_filter_create(
                cpu_device,
                weights.as_ptr(),
                weights.len(),
                &cpu_desc,
                &mut cpu,
                std::ptr::null_mut(),
                0,
            ));
            ok(oidnw_filter_prepare(
                cpu,
                width,
                height,
                std::ptr::null_mut(),
                0,
            ));
            let mut expected = vec![0.0f32; count];
            let mut cpu_ms = Vec::new();
            let mut gpu_ms = Vec::new();
            let mut request: SubmitDesc = std::mem::zeroed();
            oidnw_gpu_submit_desc_default(&mut request);
            request.images = images;
            for iteration in 0..12 {
                let scale = if iteration % 2 == 0 { 0.5 } else { 2.0 };
                ok(oidnw_filter_set_input_scale(
                    cpu,
                    scale,
                    std::ptr::null_mut(),
                    0,
                ));
                request.input_scale = scale;
                let started = Instant::now();
                ok(oidnw_filter_execute(
                    cpu,
                    width,
                    height,
                    inputs[0].as_ptr(),
                    inputs[1].as_ptr(),
                    inputs[2].as_ptr(),
                    expected.as_mut_ptr(),
                    std::ptr::null_mut(),
                    0,
                ));
                if iteration >= 5 {
                    cpu_ms.push(started.elapsed().as_secs_f64() * 1000.0);
                }
                let started = Instant::now();
                let mut sequence = 0;
                ok(oidnw_gpu_submit(
                    prepared,
                    &request,
                    &mut sequence,
                    std::ptr::null_mut(),
                    0,
                ));
                let mut complete = 0;
                ok(oidnw_gpu_wait(
                    device,
                    sequence,
                    u64::MAX,
                    &mut complete,
                    std::ptr::null_mut(),
                    0,
                ));
                assert_eq!(complete, 1);
                if iteration >= 5 {
                    gpu_ms.push(started.elapsed().as_secs_f64() * 1000.0);
                }
                let actual = std::slice::from_raw_parts(
                    raw_buffers[3].contents().as_ptr().cast::<f32>(),
                    count,
                );
                for (i, (&a, &b)) in actual.iter().zip(&expected).enumerate() {
                    assert!(
                        a.is_finite() && b.is_finite() && (a - b).abs() <= 1e-5 + b.abs() * 1e-4,
                        "pixel {i}: GPU {a}, CPU {b}, flags={device_flags}, checks={checks}"
                    );
                }
            }
            let (rust_device, adapter, queue) = with_runtime(&(*device).state, |runtime| {
                Ok((
                    runtime.context.device.clone(),
                    runtime.context.adapter.clone(),
                    runtime.context.queue.clone(),
                ))
            })
            .unwrap();
            let rust_model = Model::from_tza(&weights).unwrap();
            let mut rust_options = DenoiserOptions::default();
            rust_options.kernel = match kernel {
                OIDNW_KERNEL_IMPLICIT_GEMM => crate::Kernel::ImplicitGemm,
                OIDNW_KERNEL_DIRECT => crate::Kernel::Direct,
                OIDNW_KERNEL_SUBGROUP_MATRIX => crate::Kernel::SubgroupMatrix,
                _ => crate::Kernel::Auto,
            };
            rust_options.quality = match quality {
                OIDNW_QUALITY_FAST => crate::Quality::Fast,
                OIDNW_QUALITY_BALANCED => crate::Quality::Balanced,
                _ => crate::Quality::High,
            };
            if device_flags == 4 {
                let mut unsupported = rust_options.clone();
                unsupported.kernel = crate::Kernel::SubgroupMatrix;
                assert!(matches!(
                    Denoiser::new(&rust_device, &adapter, &rust_model, unsupported),
                    Err(crate::Error::UnsupportedDevice(_))
                ));
            }
            let mut denoiser = if checks {
                Denoiser::new(&rust_device, &adapter, &rust_model, rust_options)
            } else {
                Denoiser::new_unchecked(&rust_device, &adapter, &rust_model, rust_options)
            }
            .unwrap();
            denoiser.warm_up([width, height]).unwrap();
            let pipeline_count = denoiser.cached_pipeline_count();
            let buffers: Vec<_> = handles
                .iter()
                .map(|handle| (&(**handle).inner).buffer.clone())
                .collect();
            let mut rust_prepared = denoiser
                .prepare(
                    [width, height],
                    DenoiseBuffers {
                        color: (&buffers[0]).into(),
                        albedo: Some((&buffers[1]).into()),
                        normal: Some((&buffers[2]).into()),
                        output: (&buffers[3]).into(),
                    },
                )
                .unwrap();
            assert_eq!(pipeline_count, denoiser.cached_pipeline_count());
            assert_eq!(
                rust_prepared
                    .info()
                    .kernels
                    .contains(&crate::KernelPath::SubgroupMatrix),
                matrix
            );
            let mut encoder = rust_device.create_command_encoder(&Default::default());
            rust_prepared
                .encode(&mut encoder, DenoiseParams::default().with_input_scale(2.0))
                .unwrap();
            let submission = queue.submit([encoder.finish()]);
            rust_device
                .poll(wgpu::PollType::Wait {
                    submission_index: Some(submission),
                    timeout: Some(Duration::from_secs(60)),
                })
                .unwrap();
            let actual =
                std::slice::from_raw_parts(raw_buffers[3].contents().as_ptr().cast::<f32>(), count);
            for (&a, &b) in actual.iter().zip(&expected) {
                assert!(
                    a.is_finite() && (a - b).abs() <= 1e-5 + b.abs() * 1e-4,
                    "Rust {a}, CPU {b}; kernel={kernel}, quality={quality}"
                );
            }
            drop(rust_prepared);
            drop(denoiser);
            drop(buffers);
            for scale in [0.0, -1.0, f32::NAN, f32::INFINITY] {
                assert_eq!(
                    oidnw_filter_set_input_scale(cpu, scale, std::ptr::null_mut(), 0),
                    OIDNW_ERROR_INVALID_ARGUMENT
                );
                cpu_desc.input_scale = scale;
                let mut invalid = std::ptr::null_mut();
                assert_eq!(
                    oidnw_filter_create(
                        cpu_device,
                        weights.as_ptr(),
                        weights.len(),
                        &cpu_desc,
                        &mut invalid,
                        std::ptr::null_mut(),
                        0
                    ),
                    OIDNW_ERROR_INVALID_ARGUMENT
                );
            }
            let address = prepared as usize;
            std::thread::scope(|scope| {
                let worker = scope.spawn(move || {
                    for _ in 0..100 {
                        let mut queried: PreparedInfo = std::mem::zeroed();
                        assert_eq!(
                            oidnw_gpu_prepared_get_info(
                                address as *const GpuPrepared,
                                &mut queried,
                                std::ptr::null_mut(),
                                0
                            ),
                            OIDNW_OK
                        );
                        assert_eq!(queried.kernels & 4 != 0, matrix);
                    }
                });
                let mut sequence = 0;
                ok(oidnw_gpu_submit(
                    prepared,
                    &request,
                    &mut sequence,
                    std::ptr::null_mut(),
                    0,
                ));
                let mut complete = 0;
                ok(oidnw_gpu_wait(
                    device,
                    sequence,
                    u64::MAX,
                    &mut complete,
                    std::ptr::null_mut(),
                    0,
                ));
                worker.join().unwrap();
            });
            cpu_ms.sort_by(f64::total_cmp);
            gpu_ms.sort_by(f64::total_cmp);
            eprintln!(
                "flags={device_flags} kernel={kernel} quality={quality} checks={checks} precision={} kernels={} CPU_median_ms={:.3} GPU_median_ms={:.3}",
                info.precision,
                info.kernels,
                cpu_ms[cpu_ms.len() / 2],
                gpu_ms[gpu_ms.len() / 2]
            );
            oidnw_filter_destroy(cpu);
            oidnw_gpu_prepared_destroy(prepared);
            oidnw_gpu_filter_destroy(filter);
            oidnw_gpu_model_destroy(model);
            for handle in handles {
                oidnw_gpu_buffer_destroy(handle);
            }
            ok(oidnw_gpu_device_destroy(device, std::ptr::null_mut(), 0));
        }
        oidnw_device_destroy(cpu_device);
    });
}
