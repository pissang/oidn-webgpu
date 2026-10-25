//! Submission client shared by the native host oracle examples.
use oidn_webgpu::{capi::*, gpu_capi::*};
use std::ffi::CStr;
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
pub struct GpuClient {
    device: *mut GpuDevice,
    prepared: *mut GpuPrepared,
    buffers: Vec<*mut GpuBuffer>,
    ready: *mut GpuSync,
    done: *mut GpuSync,
    images: Vec<Images>,
    last: u64,
    pub queue_full: usize,
}
impl GpuClient {
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn new(
        device: *mut GpuDevice,
        images: Vec<Images>,
        buffers: Vec<*mut GpuBuffer>,
        ready: *mut GpuSync,
        done: *mut GpuSync,
        weights: &[u8],
        size: u32,
        tile: u32,
        unchecked: bool,
    ) -> Result<Self> {
        unsafe {
            let mut api = Self {
                device,
                images,
                buffers,
                ready,
                done,
                prepared: std::ptr::null_mut(),
                last: 0,
                queue_full: 0,
            };
            let mut error = [0i8; 1024];
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
            let mut desc: FilterDesc = std::mem::zeroed();
            oidnw_gpu_filter_desc_default(&mut desc);
            desc.max_tile_size = tile;
            desc.binding_cache_capacity = 3;
            desc.flags = if unchecked {
                0
            } else {
                OIDNW_FILTER_RUNTIME_CHECKS
            };
            let mut filter = std::ptr::null_mut();
            let status = oidnw_gpu_filter_create(
                device,
                model,
                &desc,
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
impl Drop for GpuClient {
    fn drop(&mut self) {
        if !self.device.is_null() {
            self.release_handles();
            let _ = self.wait();
            let status = unsafe { oidnw_gpu_device_destroy(self.device, std::ptr::null_mut(), 0) };
            if status != 0 {
                eprintln!("retaining uncertain GPU C context: {status}");
            }
        }
    }
}
