//! Drives the exported C entry points from the existing native-Metal oracle.
use oidn_webgpu::{capi::*, gpu_capi::*};
use std::{
    ffi::{CStr, c_void},
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
#[derive(Default)]
struct QueueLockState {
    held: AtomicBool,
    locks: AtomicUsize,
    unlocks: AtomicUsize,
}
unsafe extern "C" fn lock(user: *mut c_void) {
    let state = unsafe { &*user.cast::<QueueLockState>() };
    while state.held.swap(true, Ordering::Acquire) {
        std::thread::yield_now();
    }
    state.locks.fetch_add(1, Ordering::Relaxed);
}
unsafe extern "C" fn unlock(user: *mut c_void) {
    let state = unsafe { &*user.cast::<QueueLockState>() };
    state.unlocks.fetch_add(1, Ordering::Relaxed);
    state.held.store(false, Ordering::Release);
}

pub struct HostQueueGuard<'a>(&'a QueueLockState);
impl Drop for HostQueueGuard<'_> {
    fn drop(&mut self) {
        unsafe { unlock((self.0 as *const QueueLockState).cast_mut().cast()) };
    }
}

pub struct MetalCApi {
    device: *mut GpuDevice,
    prepared: *mut GpuPrepared,
    buffers: Vec<*mut GpuBuffer>,
    ready: *mut GpuSync,
    done: *mut GpuSync,
    images: Vec<Images>,
    last: u64,
    pub queue_full: usize,
    callbacks: Option<Box<QueueLockState>>,
}

impl MetalCApi {
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn new(
        raw_device: *mut c_void,
        raw_queue: *mut c_void,
        raw_buffers: &[[*mut c_void; 4]],
        length: u64,
        ready: *mut c_void,
        done: *mut c_void,
        weights: &[u8],
        size: u32,
        tile: u32,
        fp32: bool,
    ) -> Result<Self> {
        unsafe {
            let mut api = Self {
                device: std::ptr::null_mut(),
                prepared: std::ptr::null_mut(),
                buffers: Vec::new(),
                ready: std::ptr::null_mut(),
                done: std::ptr::null_mut(),
                images: Vec::new(),
                last: 0,
                queue_full: 0,
                callbacks: Some(Box::default()),
            };
            let mut error = [0i8; 1024];
            let mut desc: MetalDeviceDesc = std::mem::zeroed();
            oidnw_gpu_metal_device_desc_default(&mut desc);
            desc.device = raw_device;
            desc.queue = raw_queue;
            desc.flags = u32::from(fp32) | 4; // Match the stable-feature Rust reference device.
            desc.max_pending_submissions = 8; // Exercise bounded admission/backpressure.
            desc.lock = Some(lock);
            desc.unlock = Some(unlock);
            desc.user_data = (&mut **api.callbacks.as_mut().unwrap() as *mut QueueLockState).cast();
            check(
                oidnw_gpu_device_import_metal(
                    &desc,
                    &mut api.device,
                    error.as_mut_ptr(),
                    error.len(),
                ),
                &error,
            )?;
            for group in raw_buffers {
                let mut handles = Vec::new();
                for raw in group {
                    let mut handle = std::ptr::null_mut();
                    check(
                        oidnw_gpu_buffer_import_metal(
                            api.device,
                            *raw,
                            length,
                            &mut handle,
                            error.as_mut_ptr(),
                            error.len(),
                        ),
                        &error,
                    )?;
                    api.buffers.push(handle);
                    handles.push(handle);
                }
                let image = |i| Image {
                    buffer: handles[i],
                    offset: 256,
                };
                api.images.push(Images {
                    color: image(0),
                    albedo: image(1),
                    normal: image(2),
                    output: image(3),
                });
            }
            check(
                oidnw_gpu_sync_import_metal(
                    api.device,
                    ready,
                    &mut api.ready,
                    error.as_mut_ptr(),
                    error.len(),
                ),
                &error,
            )?;
            check(
                oidnw_gpu_sync_import_metal(
                    api.device,
                    done,
                    &mut api.done,
                    error.as_mut_ptr(),
                    error.len(),
                ),
                &error,
            )?;
            let mut model = std::ptr::null_mut();
            check(
                oidnw_gpu_model_create(
                    weights.as_ptr().cast(),
                    weights.len(),
                    &mut model,
                    error.as_mut_ptr(),
                    error.len(),
                ),
                &error,
            )?;
            let mut filter_desc: FilterDesc = std::mem::zeroed();
            oidnw_gpu_filter_desc_default(&mut filter_desc);
            filter_desc.max_tile_size = tile;
            filter_desc.binding_cache_capacity = 3;
            let mut filter = std::ptr::null_mut();
            let status = oidnw_gpu_filter_create(
                api.device,
                model,
                &filter_desc,
                &mut filter,
                error.as_mut_ptr(),
                error.len(),
            );
            oidnw_gpu_model_destroy(model);
            check(status, &error)?;
            let status = oidnw_gpu_prepare(
                filter,
                size,
                size,
                &api.images[0],
                &mut api.prepared,
                error.as_mut_ptr(),
                error.len(),
            );
            oidnw_gpu_filter_destroy(filter);
            check(status, &error)?;
            Ok(api)
        }
    }
    pub fn lock_host_queue(&self) -> HostQueueGuard<'_> {
        let state = self.callbacks.as_deref().unwrap();
        unsafe { lock((state as *const QueueLockState).cast_mut().cast()) };
        HostQueueGuard(state)
    }
    pub fn submit(&mut self, slot: usize, value: u64, scale: f32) -> Result<()> {
        unsafe {
            let mut desc: SubmitDesc = std::mem::zeroed();
            oidnw_gpu_submit_desc_default(&mut desc);
            desc.images = self.images[slot];
            desc.input_scale = scale;
            let wait = TimelinePoint {
                sync: self.ready,
                value,
            };
            let signal = TimelinePoint {
                sync: self.done,
                value,
            };
            desc.waits = &wait;
            desc.wait_count = 1;
            desc.signals = &signal;
            desc.signal_count = 1;
            loop {
                let mut sequence = 0;
                let mut error = [0i8; 1024];
                let status = oidnw_gpu_submit(
                    self.prepared,
                    &desc,
                    &mut sequence,
                    error.as_mut_ptr(),
                    error.len(),
                );
                if status == OIDNW_ERROR_QUEUE_FULL {
                    if sequence != 0 {
                        return Err("QUEUE_FULL accepted synchronization".into());
                    }
                    self.queue_full += 1;
                    self.wait()?;
                } else {
                    check(status, &error)?;
                    if sequence <= self.last {
                        return Err("C submission sequence did not increase".into());
                    }
                    self.last = sequence;
                    return Ok(());
                }
            }
        }
    }
    pub fn wait(&self) -> Result<()> {
        if self.last == 0 {
            return Ok(());
        }
        let mut complete = 0;
        let mut error = [0i8; 1024];
        let status = unsafe {
            oidnw_gpu_wait(
                self.device,
                self.last,
                60_000_000_000,
                &mut complete,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        check(status, &error)?;
        if complete != 1 {
            return Err("wait returned without external completion".into());
        }
        Ok(())
    }
    pub fn release_handles(&mut self) {
        unsafe {
            oidnw_gpu_prepared_destroy(self.prepared);
            self.prepared = std::ptr::null_mut();
            for buffer in self.buffers.drain(..) {
                oidnw_gpu_buffer_destroy(buffer);
            }
            oidnw_gpu_sync_destroy(self.ready);
            self.ready = std::ptr::null_mut();
            oidnw_gpu_sync_destroy(self.done);
            self.done = std::ptr::null_mut();
        }
    }
    pub fn finish(mut self) -> Result<()> {
        self.release_handles();
        self.wait()?;
        let mut error = [0i8; 1024];
        let status =
            unsafe { oidnw_gpu_device_destroy(self.device, error.as_mut_ptr(), error.len()) };
        check(status, &error)?;
        self.device = std::ptr::null_mut();
        let callbacks = self.callbacks.as_ref().unwrap();
        if callbacks.held.load(Ordering::Acquire)
            || callbacks.locks.load(Ordering::Relaxed) == 0
            || callbacks.locks.load(Ordering::Relaxed) != callbacks.unlocks.load(Ordering::Relaxed)
        {
            return Err("queue callbacks were unbalanced or not invoked".into());
        }
        Ok(())
    }
}
fn check(status: i32, error: &[i8]) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(format!(
            "GPU C API status {status}: {}",
            unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy()
        )
        .into())
    }
}
impl Drop for MetalCApi {
    fn drop(&mut self) {
        if self.device.is_null() {
            return;
        }
        self.release_handles();
        let _ = self.wait();
        let status = unsafe { oidnw_gpu_device_destroy(self.device, std::ptr::null_mut(), 0) };
        if status != 0 {
            // Diagnostics may exit with an unresolved producer gate. Keep
            // callback userdata alive with the intentionally retained device.
            std::mem::forget(self.callbacks.take());
            eprintln!("GPU C API cleanup retained device, status {status}");
        }
    }
}
