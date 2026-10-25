use super::*;

#[test]
fn invalid_calls_clear_outputs_and_defaults_match_contract() {
    unsafe {
        let mut desc: SubmitDesc = std::mem::zeroed();
        oidnw_gpu_submit_desc_default(&mut desc);
        assert_eq!(desc.input_scale, 1.0);
        assert!(desc.images.color.buffer.is_null());
        let mut seq = 99;
        let mut message = [1i8; 1];
        assert_eq!(
            oidnw_gpu_submit(
                std::ptr::null_mut(),
                &desc,
                &mut seq,
                message.as_mut_ptr(),
                1
            ),
            OIDNW_ERROR_INVALID_ARGUMENT
        );
        assert_eq!(seq, 0);
        assert_eq!(message, [0]);
        let mut complete = 9;
        assert_eq!(
            oidnw_gpu_wait(
                std::ptr::null_mut(),
                1,
                0,
                &mut complete,
                std::ptr::null_mut(),
                0
            ),
            OIDNW_ERROR_INVALID_ARGUMENT
        );
        assert_eq!(complete, 0);
        let mut model = std::ptr::dangling_mut::<GpuModel>();
        assert_eq!(
            oidnw_gpu_model_create(std::ptr::null(), 0, &mut model, std::ptr::null_mut(), 0),
            OIDNW_ERROR_INVALID_ARGUMENT
        );
        assert!(model.is_null());
        assert_eq!(
            oidnw_gpu_device_destroy(std::ptr::null_mut(), std::ptr::null_mut(), 0),
            OIDNW_OK
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires a Metal GPU; tests post-acceptance failure without a driver fault"]
fn metal_backpressure_and_failed_batch_preserve_external_completion() {
    use objc2::rc::Retained;
    use objc2_metal::{
        MTLCreateSystemDefaultDevice, MTLDevice, MTLResourceOptions, MTLSharedEvent,
    };
    struct QueueAccess {
        held: std::sync::atomic::AtomicBool,
        locks: AtomicUsize,
        unlocks: AtomicUsize,
    }
    unsafe extern "C" fn lock(user: *mut c_void) {
        let access = unsafe { &*user.cast::<QueueAccess>() };
        while access.held.swap(true, Ordering::Acquire) {
            std::thread::yield_now();
        }
        access.locks.fetch_add(1, Ordering::Relaxed);
    }
    unsafe extern "C" fn unlock(user: *mut c_void) {
        let access = unsafe { &*user.cast::<QueueAccess>() };
        access.unlocks.fetch_add(1, Ordering::Relaxed);
        access.held.store(false, Ordering::Release);
    }
    objc2::rc::autoreleasepool(|_| unsafe {
        let mut error = [0i8; 1024];
        let text = |error: &[i8]| {
            std::ffi::CStr::from_ptr(error.as_ptr())
                .to_string_lossy()
                .into_owned()
        };
        let raw = MTLCreateSystemDefaultDevice().expect("Metal GPU required");
        let queue = raw.newCommandQueueWithMaxCommandBufferCount(4096).unwrap();
        let access = Box::new(QueueAccess {
            held: std::sync::atomic::AtomicBool::new(false),
            locks: AtomicUsize::new(0),
            unlocks: AtomicUsize::new(0),
        });
        let ready = raw.newSharedEvent().unwrap();
        let done = raw.newSharedEvent().unwrap();
        let mut desc: MetalDeviceDesc = std::mem::zeroed();
        oidnw_gpu_metal_device_desc_default(&mut desc);
        desc.device = Retained::as_ptr(&raw).cast_mut().cast();
        desc.queue = Retained::as_ptr(&queue).cast_mut().cast();
        desc.lock = Some(lock);
        desc.unlock = Some(unlock);
        desc.user_data = (&*access as *const QueueAccess).cast_mut().cast();
        desc.max_pending_submissions = 1;
        let mut device = std::ptr::null_mut();
        assert_eq!(
            oidnw_gpu_device_import_metal(&desc, &mut device, error.as_mut_ptr(), error.len()),
            0,
            "{}",
            text(&error)
        );
        let mut imported = Vec::new();
        let raw_buffers: Vec<_> = (0..4)
            .map(|_| {
                raw.newBufferWithLength_options(4096, MTLResourceOptions::StorageModePrivate)
                    .unwrap()
            })
            .collect();
        for buffer in &raw_buffers {
            let mut handle = std::ptr::null_mut();
            assert_eq!(
                oidnw_gpu_buffer_import_metal(
                    device,
                    Retained::as_ptr(buffer).cast_mut().cast(),
                    4096,
                    &mut handle,
                    error.as_mut_ptr(),
                    error.len()
                ),
                0,
                "{}",
                text(&error)
            );
            imported.push(handle);
        }
        let mut duplicate = std::ptr::null_mut();
        assert_eq!(
            oidnw_gpu_buffer_import_metal(
                device,
                Retained::as_ptr(&raw_buffers[0]).cast_mut().cast(),
                4096,
                &mut duplicate,
                error.as_mut_ptr(),
                error.len()
            ),
            OIDNW_ERROR_INVALID_ARGUMENT
        );
        assert!(duplicate.is_null());
        let mut ready_handle = std::ptr::null_mut();
        let mut done_handle = std::ptr::null_mut();
        assert_eq!(
            oidnw_gpu_sync_import_metal(
                device,
                Retained::as_ptr(&ready).cast_mut().cast(),
                &mut ready_handle,
                error.as_mut_ptr(),
                error.len()
            ),
            0,
            "{}",
            text(&error)
        );
        assert_eq!(
            oidnw_gpu_sync_import_metal(
                device,
                Retained::as_ptr(&done).cast_mut().cast(),
                &mut done_handle,
                error.as_mut_ptr(),
                error.len()
            ),
            0,
            "{}",
            text(&error)
        );
        let weights = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../weights/rt_hdr_alb_nrm_small.tza"
        ))
        .unwrap();
        let mut model = std::ptr::null_mut();
        assert_eq!(
            oidnw_gpu_model_create(
                weights.as_ptr().cast(),
                weights.len(),
                &mut model,
                error.as_mut_ptr(),
                error.len()
            ),
            0
        );
        let mut filter = std::ptr::null_mut();
        let mut filter_desc: FilterDesc = std::mem::zeroed();
        oidnw_gpu_filter_desc_default(&mut filter_desc);
        assert_eq!(
            oidnw_gpu_filter_create(
                device,
                model,
                std::ptr::null(),
                &mut filter,
                error.as_mut_ptr(),
                error.len()
            ),
            OIDNW_ERROR_INVALID_ARGUMENT
        );
        assert!(filter.is_null());
        assert_eq!(
            oidnw_gpu_filter_create(
                device,
                model,
                &filter_desc,
                &mut filter,
                error.as_mut_ptr(),
                error.len()
            ),
            0,
            "{}",
            text(&error)
        );
        oidnw_gpu_model_destroy(model); // The filter owns its Model clone.
        let images = Images {
            color: Image {
                buffer: imported[0],
                offset: 0,
            },
            albedo: Image {
                buffer: imported[1],
                offset: 0,
            },
            normal: Image {
                buffer: imported[2],
                offset: 0,
            },
            output: Image {
                buffer: imported[3],
                offset: 0,
            },
        };
        let mut prepared = std::ptr::null_mut();
        assert_eq!(
            oidnw_gpu_prepare(
                filter,
                16,
                16,
                &images,
                &mut prepared,
                error.as_mut_ptr(),
                error.len()
            ),
            0,
            "{}",
            text(&error)
        );
        let mut second = std::ptr::null_mut();
        assert_eq!(
            oidnw_gpu_prepare(
                filter,
                16,
                16,
                &images,
                &mut second,
                error.as_mut_ptr(),
                error.len()
            ),
            OIDNW_OK,
            "{}",
            text(&error)
        );
        oidnw_gpu_filter_destroy(filter); // A prepared workspace is independent.
        let mut submit: SubmitDesc = std::mem::zeroed();
        oidnw_gpu_submit_desc_default(&mut submit);
        submit.images = images;
        let mut wait = TimelinePoint {
            sync: ready_handle,
            value: 1,
        };
        let mut signal = TimelinePoint {
            sync: done_handle,
            value: 1,
        };
        submit.waits = &wait;
        submit.wait_count = 1;
        submit.signals = &signal;
        submit.signal_count = 1;
        let mut seq = 0;
        assert_eq!(
            oidnw_gpu_submit(prepared, &submit, &mut seq, error.as_mut_ptr(), error.len()),
            0,
            "{}",
            text(&error)
        );
        assert_eq!(seq, 1);
        let mut rejected = 42;
        assert_eq!(
            oidnw_gpu_submit(
                prepared,
                &submit,
                &mut rejected,
                error.as_mut_ptr(),
                error.len()
            ),
            OIDNW_ERROR_QUEUE_FULL
        );
        assert_eq!(rejected, 0);
        let mut complete = 1;
        assert_eq!(
            oidnw_gpu_wait(
                device,
                seq,
                0,
                &mut complete,
                error.as_mut_ptr(),
                error.len()
            ),
            OIDNW_ERROR_TIMEOUT
        );
        assert_eq!(complete, 0);
        assert_eq!(done.signaledValue(), 0);
        assert_eq!(
            oidnw_gpu_device_destroy(device, error.as_mut_ptr(), error.len()),
            OIDNW_ERROR_BUSY
        );
        ready.setSignaledValue(1);
        assert_eq!(
            oidnw_gpu_wait(
                device,
                seq,
                5_000_000_000,
                &mut complete,
                error.as_mut_ptr(),
                error.len()
            ),
            0,
            "{}",
            text(&error)
        );
        assert_eq!(complete, 1);
        assert!(done.signaledValue() >= 1);
        // Hold one workspace's CPU gate to make BUSY deterministic. Other
        // workspaces must still be usable, and rejected calls issue no sequence.
        let held = (&(*prepared).inner).gate.lock().unwrap();
        let handles = [prepared as usize, second as usize];
        let buffers = imported.iter().map(|p| *p as usize).collect::<Vec<_>>();
        let barrier = std::sync::Barrier::new(8);
        let outcomes = std::thread::scope(|scope| {
            let threads = (0..8)
                .map(|i| {
                    let barrier = &barrier;
                    let buffers = &buffers;
                    scope.spawn(move || {
                        let image = |index| Image {
                            buffer: buffers[index] as *mut GpuBuffer,
                            offset: 0,
                        };
                        let mut request: SubmitDesc = std::mem::zeroed();
                        oidnw_gpu_submit_desc_default(&mut request);
                        request.images = Images {
                            color: image(0),
                            albedo: image(1),
                            normal: image(2),
                            output: image(3),
                        };
                        let mut sequence = 99;
                        barrier.wait();
                        let status = oidnw_gpu_submit(
                            handles[i % 2] as *mut GpuPrepared,
                            &request,
                            &mut sequence,
                            std::ptr::null_mut(),
                            0,
                        );
                        (i % 2, status, sequence)
                    })
                })
                .collect::<Vec<_>>();
            threads
                .into_iter()
                .map(|t| t.join().unwrap())
                .collect::<Vec<_>>()
        });
        drop(held);
        let mut accepted = Vec::new();
        for (workspace, status, sequence) in outcomes {
            if workspace == 0 {
                assert_eq!((status, sequence), (OIDNW_ERROR_BUSY, 0));
            } else if status == OIDNW_OK {
                assert!(sequence > 1);
                accepted.push(sequence);
            } else {
                assert!(matches!(status, OIDNW_ERROR_BUSY | OIDNW_ERROR_QUEUE_FULL));
                assert_eq!(sequence, 0);
            }
        }
        assert!(!accepted.is_empty());
        accepted.sort_unstable();
        assert!(accepted.windows(2).all(|pair| pair[0] < pair[1]));
        let latest = *accepted.last().unwrap();
        assert_eq!(
            oidnw_gpu_wait(
                device,
                latest,
                5_000_000_000,
                &mut complete,
                error.as_mut_ptr(),
                error.len()
            ),
            OIDNW_OK,
            "{}",
            text(&error)
        );
        assert_eq!(complete, 1);
        oidnw_gpu_prepared_destroy(second);
        with_runtime(&(*device).state, |runtime| {
            runtime.fail_next_encoding = true;
            Ok(())
        })
        .unwrap();
        wait.value = 2;
        signal.value = 2;
        submit.waits = &wait;
        submit.signals = &signal;
        assert_eq!(
            oidnw_gpu_submit(prepared, &submit, &mut seq, error.as_mut_ptr(), error.len()),
            OIDNW_ERROR_GPU
        );
        assert_eq!(seq, latest + 1);
        assert_eq!(
            oidnw_gpu_wait(
                device,
                seq,
                0,
                &mut complete,
                error.as_mut_ptr(),
                error.len()
            ),
            OIDNW_ERROR_TIMEOUT
        );
        assert_eq!(complete, 0);
        assert_eq!(done.signaledValue(), 1);
        // Public wrappers can be released while the failed batch is gated.
        let saved = (*prepared).inner.clone();
        oidnw_gpu_prepared_destroy(prepared);
        for buffer in imported {
            oidnw_gpu_buffer_destroy(buffer);
        }
        oidnw_gpu_sync_destroy(ready_handle);
        oidnw_gpu_sync_destroy(done_handle);
        assert_eq!(
            oidnw_gpu_device_destroy(device, error.as_mut_ptr(), error.len()),
            OIDNW_ERROR_BUSY
        );
        ready.setSignaledValue(2);
        assert_eq!(
            oidnw_gpu_wait(
                device,
                seq,
                5_000_000_000,
                &mut complete,
                error.as_mut_ptr(),
                error.len()
            ),
            OIDNW_ERROR_GPU,
            "{}",
            text(&error)
        );
        assert_eq!(complete, 1);
        assert!(done.signaledValue() >= 2);
        // Simulate an uncertain accepted submission with no actual GPU work.
        // Ordinary destruction must retain its resources rather than assuming
        // a wgpu error proves the host GPU has stopped accessing them.
        with_runtime(&(*device).state, |runtime| {
            runtime.lost("injected uncertain submission; host GPU is already idle");
            let uncertain = runtime
                .ledger
                .accept(Pending {
                    _prepared: saved,
                    _buffers: Vec::new(),
                    _waits: Vec::new(),
                    signals: Vec::new(),
                })
                .unwrap();
            // A software completion notification after loss must not retire
            // unknown work or allow ordinary destruction to free resources.
            runtime
                .internal_completed
                .store(uncertain, Ordering::Release);
            Ok(())
        })
        .unwrap();
        assert_eq!(
            oidnw_gpu_device_destroy(device, error.as_mut_ptr(), error.len()),
            OIDNW_ERROR_BUSY
        );
        assert_eq!(
            oidnw_gpu_device_destroy_quiescent(device, error.as_mut_ptr(), error.len()),
            0,
            "{}",
            text(&error)
        );
        assert!(!access.held.load(Ordering::Acquire));
        assert!(access.locks.load(Ordering::Relaxed) > 8);
        assert_eq!(
            access.locks.load(Ordering::Relaxed),
            access.unlocks.load(Ordering::Relaxed)
        );
        // Host resources are still usable after library teardown.
        ready.setSignaledValue(3);
        assert_eq!(ready.signaledValue(), 3);
    });
}
