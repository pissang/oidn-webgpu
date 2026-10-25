//! GPU-resident C API. Platform objects are isolated behind Backend/Timeline;
//! the shared state machine owns numbering, completion, limits and lifetimes.
mod dx12;
mod ledger;
mod metal;
pub use dx12::{
    Dx12DeviceDesc, oidnw_gpu_buffer_import_dx12, oidnw_gpu_device_create_dx12,
    oidnw_gpu_dx12_device_desc_default, oidnw_gpu_sync_import_dx12,
};
mod vulkan;
pub use vulkan::{
    VulkanDeviceDesc, VulkanRequirements, VulkanRequirementsDesc, oidnw_gpu_buffer_import_vulkan,
    oidnw_gpu_device_import_vulkan, oidnw_gpu_sync_import_vulkan,
    oidnw_gpu_vulkan_device_desc_default, oidnw_gpu_vulkan_instance_extensions,
    oidnw_gpu_vulkan_requirements_create, oidnw_gpu_vulkan_requirements_desc_default,
    oidnw_gpu_vulkan_requirements_destroy, oidnw_gpu_vulkan_requirements_get,
};
#[cfg(all(test, target_os = "windows"))]
mod dx12_tests;
#[cfg(test)]
mod tests;
#[cfg(all(test, any(target_os = "windows", target_os = "linux")))]
mod vulkan_tests;

use crate::capi::*;
use crate::denoiser::ImageBindings;
use crate::{
    DenoiseBuffers, DenoiseParams, Denoiser, DenoiserOptions, ImageBuffer, Model, PreparedDenoise,
    Transfer,
};
use ledger::Ledger;
use std::{
    collections::{HashMap, VecDeque},
    ffi::{c_char, c_void},
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Arc, Mutex, Weak,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

pub type QueueLock = Option<unsafe extern "C" fn(*mut c_void)>;
#[repr(C)]
#[derive(Clone, Copy)]
pub struct MetalDeviceDesc {
    pub flags: u32,
    pub device: *mut c_void,
    pub queue: *mut c_void,
    pub lock: QueueLock,
    pub unlock: QueueLock,
    pub user_data: *mut c_void,
    pub max_pending_submissions: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FilterDesc {
    pub hdr: u32,
    pub transfer: u32,
    pub max_tile_size: u32,
    pub flags: u32,
    pub binding_cache_capacity: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Image {
    pub buffer: *mut GpuBuffer,
    pub offset: u64,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Images {
    pub color: Image,
    pub albedo: Image,
    pub normal: Image,
    pub output: Image,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct TimelinePoint {
    pub sync: *mut GpuSync,
    pub value: u64,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SubmitDesc {
    pub input_scale: f32,
    pub images: Images,
    pub waits: *const TimelinePoint,
    pub wait_count: usize,
    pub signals: *const TimelinePoint,
    pub signal_count: usize,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct DeviceLimits {
    pub image_offset_alignment: u32,
    pub max_buffer_size: u64,
    pub max_image_bytes: u64,
    pub max_pending_submissions: u32,
}

pub struct GpuDevice {
    state: Arc<DeviceState>,
    gate: Mutex<()>,
}
pub struct GpuModel {
    model: Model,
}
pub struct GpuFilter {
    _lease: Lease,
    state: Arc<DeviceState>,
    model: Model,
    gate: Mutex<Denoiser>,
    cache_capacity: usize,
}
pub struct GpuPrepared {
    _lease: Lease,
    inner: Arc<PreparedInner>,
}
pub struct GpuBuffer {
    _lease: Lease,
    inner: Arc<BufferInner>,
    token: Arc<()>,
}
pub struct GpuSync {
    _lease: Lease,
    inner: Arc<SyncInner>,
}

trait Timeline: Send + Sync {
    fn value(&self) -> Outcome<u64>;
    fn stage_wait(&self, queue: &wgpu::Queue, value: u64) -> Outcome<()>;
    fn stage_signal(&self, queue: &wgpu::Queue, value: u64) -> Outcome<()>;
    fn unstage(&self, queue: &wgpu::Queue);
}
trait Backend: Send + Sync {
    fn check_health(&self) -> Outcome<()> {
        Ok(())
    }
    unsafe fn identity(&self, raw: *mut c_void) -> Outcome<usize> {
        Ok(raw as usize)
    }
    unsafe fn retain_buffer(&self, _raw: *mut c_void) -> Outcome<Option<Box<dyn Send + Sync>>> {
        Ok(None)
    }
    fn check_admission(&self, _ledger: &Ledger<Pending>) -> Outcome<()> {
        Ok(())
    }
    unsafe fn buffer(
        &self,
        device: &wgpu::Device,
        raw: *mut c_void,
        size: u64,
    ) -> Outcome<wgpu::Buffer>;
    unsafe fn timeline(&self, raw: *mut c_void) -> Outcome<Box<dyn Timeline>>;
}
struct Context {
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    backend: Box<dyn Backend>,
}
#[derive(Clone, Copy)]
struct Callbacks {
    lock: QueueLock,
    unlock: QueueLock,
    user: usize,
}
struct QueueGuard(Callbacks);
impl Callbacks {
    fn enter(self) -> QueueGuard {
        if let Some(lock) = self.lock {
            unsafe { lock(self.user as *mut c_void) };
        }
        QueueGuard(self)
    }
}
impl Drop for QueueGuard {
    fn drop(&mut self) {
        if let Some(unlock) = self.0.unlock {
            unsafe { unlock(self.0.user as *mut c_void) };
        }
    }
}
#[derive(Default)]
struct Health {
    failure: Option<Failure>,
}
struct DeviceState {
    runtime: Mutex<Option<Runtime>>,
    callbacks: Callbacks,
    children: AtomicUsize,
}
struct Runtime {
    context: Context,
    health: Arc<Mutex<Health>>,
    internal_completed: Arc<AtomicU64>,
    ledger: Ledger<Pending>,
    buffers: HashMap<usize, Weak<BufferInner>>,
    syncs: HashMap<usize, Weak<SyncInner>>,
    prepared: Vec<Weak<PreparedInner>>,
    next_buffer: u64,
    failed: Option<(u64, Failure)>,
    failure_marker: wgpu::Buffer,
    #[cfg(test)]
    fail_next_encoding: bool,
}
// A lease counts only public handles, not references retained by in-flight work.
struct Lease(Arc<DeviceState>);
impl Lease {
    fn new(state: &Arc<DeviceState>) -> Self {
        state.children.fetch_add(1, Ordering::Relaxed);
        Self(state.clone())
    }
}
impl Drop for Lease {
    fn drop(&mut self) {
        self.0.children.fetch_sub(1, Ordering::Release);
    }
}
struct BufferInner {
    _native_owner: Option<Box<dyn Send + Sync>>,
    state: Arc<DeviceState>,
    buffer: wgpu::Buffer,
    id: u64,
    role: AtomicUsize,
}
struct SyncInner {
    state: Arc<DeviceState>,
    native: Box<dyn Timeline>,
    last_signal: AtomicU64,
}
#[derive(Clone)]
struct Point {
    sync: Arc<SyncInner>,
    value: u64,
}
struct View {
    buffer: Arc<BufferInner>,
    token: Weak<()>,
    offset: u64,
}
struct Bindings {
    views: Vec<View>,
    input_count: usize,
}
impl Bindings {
    fn key(&self) -> Vec<(u64, u64)> {
        self.views.iter().map(|v| (v.buffer.id, v.offset)).collect()
    }
    fn images(&self) -> DenoiseBuffers<'_> {
        let view = |i: usize| ImageBuffer {
            buffer: &self.views[i].buffer.buffer,
            offset: self.views[i].offset,
        };
        DenoiseBuffers {
            color: view(0),
            albedo: (self.input_count >= 2).then(|| view(1)),
            normal: (self.input_count >= 3).then(|| view(2)),
            output: view(self.input_count),
        }
    }
    fn roles(&self, set: bool) -> Outcome<()> {
        for (index, view) in self.views.iter().enumerate() {
            let wanted = if index == self.input_count { 2 } else { 1 };
            let role = view.buffer.role.load(Ordering::Relaxed);
            if role != 0 && role != wanted {
                return Err(invalid("buffer input/output role cannot change"));
            }
        }
        if set {
            for (index, view) in self.views.iter().enumerate() {
                view.buffer.role.store(
                    if index == self.input_count { 2 } else { 1 },
                    Ordering::Relaxed,
                );
            }
        }
        Ok(())
    }
}
struct CachedBindings {
    key: Vec<(u64, u64)>,
    _buffers: Vec<Arc<BufferInner>>,
    tokens: Vec<Weak<()>>,
    snapshot: ImageBindings,
}
struct PreparedData {
    prepared: PreparedDenoise,
    cache: VecDeque<CachedBindings>,
    cache_capacity: usize,
    // Current bindings outlive cache eviction and preserve duplicate-import identity.
    current: Vec<Arc<BufferInner>>,
}
struct PreparedInner {
    state: Arc<DeviceState>,
    inputs: usize,
    bytes: u64,
    gate: Mutex<PreparedData>,
}
struct Pending {
    _prepared: Arc<PreparedInner>,
    _buffers: Vec<Arc<BufferInner>>,
    _waits: Vec<Point>,
    signals: Vec<Point>,
}
impl PreparedData {
    fn prune(&mut self) {
        self.cache
            .retain(|c| c.tokens.iter().all(|t| t.strong_count() != 0));
    }
    fn remember(&mut self, bindings: &Bindings) {
        if self.cache.len() == self.cache_capacity {
            self.cache.pop_front();
        }
        self.cache.push_back(CachedBindings {
            key: bindings.key(),
            _buffers: bindings.views.iter().map(|v| v.buffer.clone()).collect(),
            tokens: bindings.views.iter().map(|v| v.token.clone()).collect(),
            snapshot: self.prepared.image_bindings(),
        });
    }
    fn bind(&mut self, bindings: &Bindings) -> Outcome<()> {
        self.prune();
        let key = bindings.key();
        if let Some(index) = self.cache.iter().position(|c| c.key == key) {
            let cached = self.cache.remove(index).unwrap();
            self.prepared.restore_image_bindings(&cached.snapshot);
            self.cache.push_back(cached);
        } else {
            self.prepared
                .rebind(bindings.images())
                .map_err(denoise_error)?;
            self.remember(bindings);
        }
        self.current = bindings.views.iter().map(|v| v.buffer.clone()).collect();
        Ok(())
    }
}
impl Runtime {
    fn healthy(&self) -> Outcome<()> {
        if let Some(failure) = &self
            .health
            .lock()
            .map_err(|_| gpu("health lock poisoned"))?
            .failure
        {
            return Err(failure.clone());
        }
        if let Some((_, failure)) = &self.failed {
            return Err(failure.clone());
        }
        Ok(())
    }
    fn lost(&self, message: impl Into<String>) -> Failure {
        let failure = Failure(OIDNW_ERROR_DEVICE_LOST, message.into());
        if let Ok(mut health) = self.health.lock() {
            health.failure = Some(failure.clone());
        }
        failure
    }
    fn maintain(&mut self) -> Outcome<()> {
        self.context
            .backend
            .check_health()
            .map_err(|e| self.lost(e.1))?;
        self.context
            .device
            .poll(wgpu::PollType::Poll)
            .map_err(|e| self.lost(format!("GPU poll failed: {e}")))?;
        self.context
            .backend
            .check_health()
            .map_err(|e| self.lost(e.1))?;
        // Loss/error callbacks may run while polling. Once submission outcome
        // is uncertain, even completed software fences or CPU-advanced events
        // cannot establish physical GPU quiescence. Retain pending resources
        // until the host explicitly guarantees quiescence.
        let health = self
            .health
            .lock()
            .map_err(|_| gpu("health lock poisoned"))?;
        if health.failure.is_none() {
            let internal = self.internal_completed.load(Ordering::Acquire);
            // Serialize retirement with asynchronous loss/error callbacks.
            // Drop this guard before marking a native query error as uncertain.
            let retirement = self.ledger.try_retire(internal, |pending| {
                let mut complete = true;
                for point in &pending.signals {
                    let value = point.sync.native.value()?;
                    complete &= value >= point.value;
                }
                Ok::<_, Failure>(complete)
            });
            if let Err(error) = retirement {
                drop(health);
                return Err(self.lost(error.1));
            }
        }
        drop(health);
        self.buffers.retain(|_, b| b.strong_count() != 0);
        self.syncs.retain(|_, s| s.strong_count() != 0);
        self.prepared.retain(|p| p.strong_count() != 0);
        for prepared in self.prepared.iter().filter_map(Weak::upgrade) {
            if let Ok(mut data) = prepared.gate.try_lock() {
                data.prune();
            }
        }
        Ok(())
    }
    fn status(&self, sequence: u64) -> Outcome<bool> {
        if sequence == 0 || sequence > self.ledger.issued {
            return Err(invalid("unknown submission sequence"));
        }
        if let Some(failure) = &self
            .health
            .lock()
            .map_err(|_| gpu("health lock poisoned"))?
            .failure
        {
            return Err(failure.clone());
        }
        if sequence <= self.ledger.completed {
            if let Some((failed, failure)) = &self.failed {
                if sequence == *failed {
                    return Err(failure.clone());
                }
            }
            Ok(true)
        } else {
            Ok(false)
        }
    }
}
impl Drop for DeviceState {
    fn drop(&mut self) {
        // Normal destruction takes Runtime explicitly while holding this guard.
        // This path also protects final queue destruction after partial creation.
        let runtime = self
            .runtime
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if runtime.is_some() {
            let _queue = self.callbacks.enter();
            drop(runtime.take());
        }
    }
}
fn exclusive<'a, T>(mutex: &'a Mutex<T>, name: &str) -> Outcome<std::sync::MutexGuard<'a, T>> {
    match mutex.try_lock() {
        Ok(guard) => Ok(guard),
        Err(std::sync::TryLockError::WouldBlock) => Err(Failure(OIDNW_ERROR_BUSY, name.into())),
        Err(std::sync::TryLockError::Poisoned(_)) => Err(gpu(
            "handle operation previously panicked; release the handle",
        )),
    }
}
fn gpu(message: impl Into<String>) -> Failure {
    Failure(OIDNW_ERROR_GPU, message.into())
}
fn unavailable() -> Failure {
    Failure(
        OIDNW_ERROR_UNAVAILABLE,
        "Metal interop requires macOS".into(),
    )
}
fn denoise_error(error: crate::Error) -> Failure {
    let status = match error {
        crate::Error::InvalidArgument(_)
        | crate::Error::BufferTooSmall { .. }
        | crate::Error::LimitExceeded { .. } => OIDNW_ERROR_INVALID_ARGUMENT,
        crate::Error::Model(_) => OIDNW_ERROR_MODEL,
        crate::Error::UnsupportedDevice(_) => OIDNW_ERROR_UNAVAILABLE,
        _ => OIDNW_ERROR_GPU,
    };
    Failure(status, error.to_string())
}
fn with_runtime<T>(
    state: &Arc<DeviceState>,
    body: impl FnOnce(&mut Runtime) -> Outcome<T>,
) -> Outcome<T> {
    let mut runtime = match state.runtime.lock() {
        Ok(runtime) => runtime,
        Err(poisoned) => {
            let runtime = poisoned.into_inner();
            if let Some(runtime) = runtime.as_ref() {
                runtime.lost("previous GPU operation panicked; device is quarantined");
            }
            runtime
        }
    };
    let _queue = state.callbacks.enter();
    body(
        runtime
            .as_mut()
            .ok_or_else(|| invalid("device was destroyed"))?,
    )
}
unsafe fn required<'a, T>(pointer: *const T, name: &str) -> Outcome<&'a T> {
    if pointer.is_null() || !pointer.is_aligned() {
        Err(invalid(format!("{name} is NULL or unaligned")))
    } else {
        Ok(unsafe { &*pointer })
    }
}
fn check_pointer<T>(pointer: *const T, name: &str) -> Outcome<()> {
    if pointer.is_null() || !pointer.is_aligned() {
        Err(invalid(format!("{name} is NULL or unaligned")))
    } else {
        Ok(())
    }
}
unsafe fn descriptor<T: Copy>(pointer: *const T) -> Outcome<T> {
    Ok(*unsafe { required(pointer, "description")? })
}
unsafe fn output<T>(pointer: *mut *mut T) -> Outcome<()> {
    if pointer.is_null() || !pointer.is_aligned() {
        return Err(invalid("output handle is NULL or unaligned"));
    }
    unsafe {
        pointer.write(std::ptr::null_mut());
    }
    Ok(())
}
unsafe fn bindings(
    state: &Arc<DeviceState>,
    count: usize,
    bytes: u64,
    alignment: u64,
    images: *const Images,
) -> Outcome<Bindings> {
    let images = unsafe { required(images, "images")? };
    let source = [images.color, images.albedo, images.normal, images.output];
    if source[count..3]
        .iter()
        .any(|v| !v.buffer.is_null() || v.offset != 0)
    {
        return Err(invalid("unused auxiliary images must be {NULL,0}"));
    }
    let mut views = Vec::new();
    for index in (0..count).chain(std::iter::once(3)) {
        let image = source[index];
        let buffer = unsafe { required(image.buffer, "image buffer")? };
        if !Arc::ptr_eq(&buffer.inner.state, state) {
            return Err(invalid("image belongs to another GPU device"));
        }
        if !image.offset.is_multiple_of(alignment)
            || image
                .offset
                .checked_add(bytes)
                .is_none_or(|end| end > buffer.inner.buffer.size())
        {
            return Err(invalid(
                "image offset is misaligned or its range exceeds the buffer",
            ));
        }
        views.push(View {
            buffer: buffer.inner.clone(),
            token: Arc::downgrade(&buffer.token),
            offset: image.offset,
        });
    }
    for i in 0..views.len() {
        for previous in &views[..i] {
            if Arc::ptr_eq(&views[i].buffer, &previous.buffer)
                && (i == count
                    || (views[i].offset < previous.offset + bytes
                        && previous.offset < views[i].offset + bytes))
            {
                return Err(invalid(
                    "image ranges overlap or output shares an input buffer",
                ));
            }
        }
    }
    let bindings = Bindings {
        views,
        input_count: count,
    };
    bindings.roles(false)?;
    Ok(bindings)
}
unsafe fn points(
    state: &Arc<DeviceState>,
    pointer: *const TimelinePoint,
    count: usize,
) -> Outcome<Vec<Point>> {
    if count > 64 || (count != 0 && (pointer.is_null() || !pointer.is_aligned())) {
        return Err(invalid(
            "timeline points need a valid array of at most 64 entries",
        ));
    }
    if count == 0 {
        return Ok(Vec::new());
    }
    unsafe { std::slice::from_raw_parts(pointer, count) }
        .iter()
        .map(|point| {
            if point.value == u64::MAX {
                return Err(invalid(
                    "UINT64_MAX is reserved; timeline values must be smaller",
                ));
            }
            let sync = unsafe { required(point.sync, "sync")? };
            if !Arc::ptr_eq(&sync.inner.state, state) {
                return Err(invalid("sync belongs to another GPU device"));
            }
            Ok(Point {
                sync: sync.inner.clone(),
                value: point.value,
            })
        })
        .collect()
}
fn validate_signals(waits: &[Point], signals: &[Point]) -> Outcome<()> {
    for (i, wait) in waits.iter().enumerate() {
        if waits[..i].iter().any(|p| Arc::ptr_eq(&p.sync, &wait.sync)) {
            return Err(invalid("duplicate wait object"));
        }
    }
    for (i, signal) in signals.iter().enumerate() {
        if signal.value
            <= signal
                .sync
                .last_signal
                .load(Ordering::Relaxed)
                .max(signal.sync.native.value()?)
        {
            return Err(invalid(
                "signal value must increase beyond current and accepted values",
            ));
        }
        if signals[..i]
            .iter()
            .any(|p| Arc::ptr_eq(&p.sync, &signal.sync))
        {
            return Err(invalid("duplicate signal object"));
        }
        if waits
            .iter()
            .any(|p| Arc::ptr_eq(&p.sync, &signal.sync) && p.value >= signal.value)
        {
            return Err(invalid(
                "submission cannot wait for its own or a later signal",
            ));
        }
    }
    Ok(())
}
fn context_state(context: Context, callbacks: Callbacks, capacity: usize) -> Arc<DeviceState> {
    let health = Arc::new(Mutex::new(Health::default()));
    let errors = health.clone();
    context.device.on_uncaptured_error(Arc::new(move |error| {
        if let Ok(mut health) = errors.lock() {
            if health.failure.is_none() {
                health.failure = Some(Failure(
                    OIDNW_ERROR_DEVICE_LOST,
                    format!("uncaptured GPU error; submission outcome may be uncertain: {error}"),
                ));
            }
        }
    }));
    let lost = health.clone();
    context
        .device
        .set_device_lost_callback(move |reason, message| {
            if reason != wgpu::DeviceLostReason::Destroyed {
                if let Ok(mut health) = lost.lock() {
                    health.failure = Some(Failure(
                        OIDNW_ERROR_DEVICE_LOST,
                        format!("device lost ({reason:?}): {message}"),
                    ));
                }
            }
        });
    let failure_marker = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("oidn GPU failure completion marker"),
        size: 4,
        usage: wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    Arc::new(DeviceState {
        runtime: Mutex::new(Some(Runtime {
            context,
            health,
            internal_completed: Arc::new(AtomicU64::new(0)),
            ledger: Ledger::new(capacity),
            buffers: HashMap::new(),
            syncs: HashMap::new(),
            prepared: Vec::new(),
            next_buffer: 0,
            failed: None,
            failure_marker,
            #[cfg(test)]
            fail_next_encoding: false,
        })),
        callbacks,
        children: AtomicUsize::new(0),
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn oidnw_gpu_metal_device_desc_default(desc: *mut MetalDeviceDesc) {
    if !desc.is_null() && desc.is_aligned() {
        unsafe {
            desc.write(MetalDeviceDesc {
                flags: 0,
                device: std::ptr::null_mut(),
                queue: std::ptr::null_mut(),
                lock: None,
                unlock: None,
                user_data: std::ptr::null_mut(),
                max_pending_submissions: 64,
            });
        }
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oidnw_gpu_filter_desc_default(desc: *mut FilterDesc) {
    if !desc.is_null() && desc.is_aligned() {
        unsafe {
            desc.write(FilterDesc {
                hdr: 1,
                transfer: 0,
                max_tile_size: 0,
                flags: OIDNW_FILTER_RUNTIME_CHECKS,
                binding_cache_capacity: 8,
            });
        }
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oidnw_gpu_submit_desc_default(desc: *mut SubmitDesc) {
    if !desc.is_null() && desc.is_aligned() {
        let mut value: SubmitDesc = unsafe { std::mem::zeroed() };
        value.input_scale = 1.0;
        unsafe {
            desc.write(value);
        }
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oidnw_gpu_device_import_metal(
    desc: *const MetalDeviceDesc,
    result: *mut *mut GpuDevice,
    error: *mut c_char,
    capacity: usize,
) -> i32 {
    unsafe {
        guard(error, capacity, || {
            output(result)?;
            let desc = descriptor(desc)?;
            if desc.device.is_null()
                || desc.flags & !3 != 0
                || !(1..=4096).contains(&desc.max_pending_submissions)
                || desc.lock.is_some() != desc.unlock.is_some()
                || (!desc.queue.is_null() && desc.lock.is_none())
            {
                return Err(invalid(
                    "invalid Metal device description or missing shared-queue callbacks",
                ));
            }
            let callbacks = Callbacks {
                lock: desc.lock,
                unlock: desc.unlock,
                user: desc.user_data as usize,
            };
            let _queue = callbacks.enter();
            let context = metal::import(&desc)?;
            result.write(Box::into_raw(Box::new(GpuDevice {
                state: context_state(context, callbacks, desc.max_pending_submissions as usize),
                gate: Mutex::new(()),
            })));
            Ok(())
        })
    }
}
fn ensure_backend(runtime: &Runtime, backend: wgpu::Backend) -> Outcome<()> {
    if runtime.context.adapter.get_info().backend != backend {
        return Err(invalid("platform import does not match the device backend"));
    }
    Ok(())
}

/// Source-only oracle access. The caller must serialize these handles with all
/// C API calls and stop using them before native host teardown. They exist only
/// to compare the Rust denoiser on the exact device owned by the GPU C API.
#[cfg(feature = "interop-validation")]
#[doc(hidden)]
pub unsafe fn reference_context(
    device: *mut GpuDevice,
) -> Result<(wgpu::Adapter, wgpu::Device, wgpu::Queue), String> {
    unsafe { required(device, "device") }
        .and_then(|device| {
            with_runtime(&device.state, |r| {
                r.healthy()?;
                Ok((
                    r.context.adapter.clone(),
                    r.context.device.clone(),
                    r.context.queue.clone(),
                ))
            })
        })
        .map_err(|e| e.1)
}
unsafe fn import_buffer(
    device: &GpuDevice,
    raw: *mut c_void,
    length: u64,
    result: *mut *mut GpuBuffer,
    backend: wgpu::Backend,
) -> Outcome<()> {
    unsafe {
        let _call = exclusive(&device.gate, "device handle is busy")?;
        if raw.is_null() || length == 0 {
            return Err(invalid("buffer and length are required"));
        }
        with_runtime(&device.state, |runtime| {
            ensure_backend(runtime, backend)?;
            runtime.maintain()?;
            runtime.healthy()?;
            if length > runtime.context.device.limits().max_buffer_size {
                return Err(invalid("buffer exceeds device size limit"));
            }
            let identity = runtime.context.backend.identity(raw)?;
            if runtime
                .buffers
                .get(&identity)
                .is_some_and(|b| b.strong_count() != 0)
            {
                return Err(invalid("buffer is already imported"));
            }
            let buffer = runtime
                .context
                .backend
                .buffer(&runtime.context.device, raw, length)?;
            runtime.next_buffer = runtime
                .next_buffer
                .checked_add(1)
                .ok_or_else(|| gpu("buffer identity exhausted"))?;
            let inner = Arc::new(BufferInner {
                _native_owner: runtime.context.backend.retain_buffer(raw)?,
                state: device.state.clone(),
                buffer,
                id: runtime.next_buffer,
                role: AtomicUsize::new(0),
            });
            runtime.buffers.insert(identity, Arc::downgrade(&inner));
            result.write(Box::into_raw(Box::new(GpuBuffer {
                _lease: Lease::new(&device.state),
                inner,
                token: Arc::new(()),
            })));
            Ok(())
        })
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oidnw_gpu_buffer_import_metal(
    device: *mut GpuDevice,
    raw: *mut c_void,
    length: u64,
    result: *mut *mut GpuBuffer,
    error: *mut c_char,
    capacity: usize,
) -> i32 {
    unsafe {
        guard(error, capacity, || {
            output(result)?;
            import_buffer(
                required(device, "device")?,
                raw,
                length,
                result,
                wgpu::Backend::Metal,
            )
        })
    }
}
unsafe fn import_sync(
    device: &GpuDevice,
    raw: *mut c_void,
    result: *mut *mut GpuSync,
    backend: wgpu::Backend,
) -> Outcome<()> {
    unsafe {
        let _call = exclusive(&device.gate, "device handle is busy")?;
        if raw.is_null() {
            return Err(invalid("shared event is NULL"));
        }
        with_runtime(&device.state, |runtime| {
            ensure_backend(runtime, backend)?;
            runtime.maintain()?;
            runtime.healthy()?;
            let identity = runtime.context.backend.identity(raw)?;
            if runtime
                .syncs
                .get(&identity)
                .is_some_and(|s| s.strong_count() != 0)
            {
                return Err(invalid("event is already imported"));
            }
            let native = runtime.context.backend.timeline(raw).map_err(|error| {
                if error.0 == OIDNW_ERROR_DEVICE_LOST {
                    runtime.lost(error.1)
                } else {
                    error
                }
            })?;
            let inner = Arc::new(SyncInner {
                state: device.state.clone(),
                native,
                last_signal: AtomicU64::new(0),
            });
            runtime.syncs.insert(identity, Arc::downgrade(&inner));
            result.write(Box::into_raw(Box::new(GpuSync {
                _lease: Lease::new(&device.state),
                inner,
            })));
            Ok(())
        })
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oidnw_gpu_sync_import_metal(
    device: *mut GpuDevice,
    raw: *mut c_void,
    result: *mut *mut GpuSync,
    error: *mut c_char,
    capacity: usize,
) -> i32 {
    unsafe {
        guard(error, capacity, || {
            output(result)?;
            import_sync(
                required(device, "device")?,
                raw,
                result,
                wgpu::Backend::Metal,
            )
        })
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oidnw_gpu_model_create(
    weights: *const c_void,
    weights_size: usize,
    result: *mut *mut GpuModel,
    error: *mut c_char,
    capacity: usize,
) -> i32 {
    unsafe {
        guard(error, capacity, || {
            output(result)?;
            if weights.is_null() || weights_size == 0 || weights_size > u32::MAX as usize {
                return Err(invalid("weights need 1..=UINT32_MAX bytes"));
            }
            let model = Model::from_tza(std::slice::from_raw_parts(weights.cast(), weights_size))
                .map_err(denoise_error)?;
            result.write(Box::into_raw(Box::new(GpuModel { model })));
            Ok(())
        })
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oidnw_gpu_model_input_count(model: *const GpuModel) -> u32 {
    unsafe { model.as_ref() }.map_or(0, |m| m.model.input_count() as u32)
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oidnw_gpu_filter_create(
    device: *mut GpuDevice,
    model: *const GpuModel,
    desc: *const FilterDesc,
    result: *mut *mut GpuFilter,
    error: *mut c_char,
    capacity: usize,
) -> i32 {
    unsafe {
        guard(error, capacity, || {
            output(result)?;
            let device = required(device, "device")?;
            let _call = exclusive(&device.gate, "device handle is busy")?;
            let model = required(model, "model")?;
            let desc = descriptor(desc)?;
            if desc.hdr > 1
                || desc.transfer > 1
                || desc.flags & !OIDNW_FILTER_RUNTIME_CHECKS != 0
                || !(1..=64).contains(&desc.binding_cache_capacity)
            {
                return Err(invalid("invalid GPU filter description"));
            }
            with_runtime(&device.state, |runtime| {
                runtime.maintain()?;
                runtime.healthy()?;
                let mut options = DenoiserOptions::default();
                options.hdr = desc.hdr != 0;
                options.transfer = if desc.transfer == 0 {
                    Transfer::Pu
                } else {
                    Transfer::Log
                };
                options.max_tile_size = desc.max_tile_size;
                let denoiser = if desc.flags & OIDNW_FILTER_RUNTIME_CHECKS != 0 {
                    Denoiser::new(
                        &runtime.context.device,
                        &runtime.context.adapter,
                        &model.model,
                        options,
                    )
                } else {
                    // The same unchecked-kernel contract as the existing C API.
                    Denoiser::new_unchecked(
                        &runtime.context.device,
                        &runtime.context.adapter,
                        &model.model,
                        options,
                    )
                }
                .map_err(denoise_error)?;
                result.write(Box::into_raw(Box::new(GpuFilter {
                    _lease: Lease::new(&device.state),
                    state: device.state.clone(),
                    model: model.model.clone(),
                    gate: Mutex::new(denoiser),
                    cache_capacity: desc.binding_cache_capacity as usize,
                })));
                Ok(())
            })
        })
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oidnw_gpu_prepare(
    filter: *mut GpuFilter,
    width: u32,
    height: u32,
    images: *const Images,
    result: *mut *mut GpuPrepared,
    error: *mut c_char,
    capacity: usize,
) -> i32 {
    unsafe {
        guard(error, capacity, || {
            output(result)?;
            let filter = required(filter, "filter")?;
            let mut denoiser = exclusive(&filter.gate, "filter handle is busy")?;
            if !(1..=16384).contains(&width) || !(1..=16384).contains(&height) {
                return Err(invalid("image dimensions must be 1..=16384"));
            }
            with_runtime(&filter.state, |runtime| {
                runtime.maintain()?;
                runtime.healthy()?;
                let bytes = u64::from(width) * u64::from(height) * 16;
                let views = bindings(
                    &filter.state,
                    filter.model.input_count(),
                    bytes,
                    runtime
                        .context
                        .device
                        .limits()
                        .min_storage_buffer_offset_alignment as u64,
                    images,
                )?;
                let prepared = denoiser
                    .prepare([width, height], views.images())
                    .map_err(denoise_error)?;
                views.roles(true)?;
                let mut data = PreparedData {
                    prepared,
                    cache: VecDeque::new(),
                    cache_capacity: filter.cache_capacity,
                    current: views.views.iter().map(|v| v.buffer.clone()).collect(),
                };
                data.remember(&views);
                let inner = Arc::new(PreparedInner {
                    state: filter.state.clone(),
                    inputs: filter.model.input_count(),
                    bytes,
                    gate: Mutex::new(data),
                });
                runtime.prepared.push(Arc::downgrade(&inner));
                result.write(Box::into_raw(Box::new(GpuPrepared {
                    _lease: Lease::new(&filter.state),
                    inner,
                })));
                Ok(())
            })
        })
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oidnw_gpu_submit(
    prepared: *mut GpuPrepared,
    desc: *const SubmitDesc,
    sequence: *mut u64,
    error: *mut c_char,
    capacity: usize,
) -> i32 {
    unsafe {
        guard(error, capacity, || {
            check_pointer(sequence, "sequence output")?;
            sequence.write(0);
            let prepared = required(prepared, "prepared")?;
            let desc = descriptor(desc)?;
            if !desc.input_scale.is_finite() || desc.input_scale <= 0.0 {
                return Err(invalid("input_scale must be positive and finite"));
            }
            let mut data = exclusive(&prepared.inner.gate, "prepared handle is busy")?;
            let state = &prepared.inner.state;
            with_runtime(state, |runtime| {
                runtime.maintain()?;
                runtime.healthy()?;
                runtime.ledger.next().ok_or_else(|| {
                    Failure(
                        OIDNW_ERROR_QUEUE_FULL,
                        "submission queue is full or sequence numbers exhausted".into(),
                    )
                })?;
                runtime
                    .context
                    .backend
                    .check_admission(&runtime.ledger)
                    .map_err(|error| {
                        if error.0 == OIDNW_ERROR_DEVICE_LOST {
                            runtime.lost(error.1)
                        } else {
                            error
                        }
                    })?;
                let views = bindings(
                    state,
                    prepared.inner.inputs,
                    prepared.inner.bytes,
                    runtime
                        .context
                        .device
                        .limits()
                        .min_storage_buffer_offset_alignment as u64,
                    &desc.images,
                )?;
                let waits = points(state, desc.waits, desc.wait_count)?;
                let signals = points(state, desc.signals, desc.signal_count)?;
                validate_signals(&waits, &signals).map_err(|error| {
                    if error.0 == OIDNW_ERROR_DEVICE_LOST {
                        runtime.lost(error.1)
                    } else {
                        error
                    }
                })?;
                // No synchronization is staged until all CPU checks above pass.
                // Retain everything before any operation that could submit work.
                let seq = runtime
                    .ledger
                    .accept(Pending {
                        _prepared: prepared.inner.clone(),
                        _buffers: views.views.iter().map(|v| v.buffer.clone()).collect(),
                        _waits: waits.clone(),
                        signals: signals.clone(),
                    })
                    .unwrap();
                sequence.write(seq);
                for signal in &signals {
                    signal
                        .sync
                        .last_signal
                        .store(signal.value, Ordering::Relaxed);
                }
                views.roles(true)?;
                let encoding =
                    catch_unwind(AssertUnwindSafe(|| -> Outcome<wgpu::CommandBuffer> {
                        #[cfg(test)]
                        if runtime.fail_next_encoding {
                            runtime.fail_next_encoding = false;
                            return Err(gpu("injected post-acceptance encoding failure"));
                        }
                        data.bind(&views)?;
                        let mut encoder = runtime
                            .context
                            .device
                            .create_command_encoder(&Default::default());
                        data.prepared
                            .encode(
                                &mut encoder,
                                DenoiseParams {
                                    input_scale: desc.input_scale,
                                },
                            )
                            .map_err(denoise_error)?;
                        Ok(encoder.finish())
                    }))
                    .unwrap_or_else(|_| Err(gpu("panic while encoding GPU work")));
                let encoding_error = encoding.as_ref().err().cloned();
                if let Some(error) = &encoding_error {
                    runtime.failed = Some((seq, error.clone()));
                }
                let submitted = catch_unwind(AssertUnwindSafe(|| -> Outcome<()> {
                    let command = match encoding {
                        Ok(command) => command,
                        Err(_) => {
                            // HAL's empty-submit command buffer is not relay-gated on
                            // Metal. Record a private 4-byte clear to force a normal
                            // gated CB. No imported images are accessed on this path.
                            let mut encoder = runtime
                                .context
                                .device
                                .create_command_encoder(&Default::default());
                            encoder.clear_buffer(&runtime.failure_marker, 0, None);
                            encoder.finish()
                        }
                    };
                    for point in &waits {
                        point
                            .sync
                            .native
                            .stage_wait(&runtime.context.queue, point.value)?;
                    }
                    for point in &signals {
                        point
                            .sync
                            .native
                            .stage_signal(&runtime.context.queue, point.value)?;
                    }
                    runtime.context.queue.submit([command]);
                    let completed = runtime.internal_completed.clone();
                    runtime.context.queue.on_submitted_work_done(move || {
                        completed.fetch_max(seq, Ordering::Release);
                    });
                    Ok(())
                }));
                if !matches!(submitted, Ok(Ok(()))) {
                    // A HAL submit may have consumed some or all staged objects.
                    // Keep the Pending record alive and quarantine instead of
                    // guessing whether an external completion will ever arrive.
                    for point in waits.iter().chain(&signals) {
                        point.sync.native.unstage(&runtime.context.queue);
                    }
                    return Err(runtime
                        .lost("submission outcome is uncertain; external signals may not fire"));
                }
                if let Some(error) = encoding_error {
                    return Err(error);
                }
                runtime.healthy()
            })
        })
    }
}

unsafe fn poll_once(device: &GpuDevice, sequence: u64, completed: *mut u32) -> Outcome<()> {
    with_runtime(&device.state, |runtime| {
        runtime.maintain()?;
        match runtime.status(sequence) {
            Ok(value) => {
                unsafe {
                    completed.write(value as u32);
                }
                Ok(())
            }
            Err(error) => {
                // A controlled failed batch can complete its external signals
                // while returning GPU. A lost/uncertain device never says done.
                if error.0 != OIDNW_ERROR_DEVICE_LOST
                    && sequence <= runtime.ledger.completed
                    && runtime
                        .failed
                        .as_ref()
                        .is_some_and(|(failed, _)| *failed == sequence)
                {
                    unsafe {
                        completed.write(1);
                    }
                }
                Err(error)
            }
        }
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oidnw_gpu_poll(
    device: *mut GpuDevice,
    sequence: u64,
    completed: *mut u32,
    error: *mut c_char,
    capacity: usize,
) -> i32 {
    unsafe {
        guard(error, capacity, || {
            check_pointer(completed, "completed output")?;
            completed.write(0);
            let device = required(device, "device")?;
            let _call = exclusive(&device.gate, "device handle is busy")?;
            poll_once(device, sequence, completed)
        })
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oidnw_gpu_wait(
    device: *mut GpuDevice,
    sequence: u64,
    timeout_ns: u64,
    completed: *mut u32,
    error: *mut c_char,
    capacity: usize,
) -> i32 {
    unsafe {
        guard(error, capacity, || {
            check_pointer(completed, "completed output")?;
            completed.write(0);
            let device = required(device, "device")?;
            let _call = exclusive(&device.gate, "device handle is busy")?;
            let started = Instant::now();
            loop {
                poll_once(device, sequence, completed)?;
                if completed.read() != 0 {
                    return Ok(());
                }
                if timeout_ns != u64::MAX && started.elapsed() >= Duration::from_nanos(timeout_ns) {
                    return Err(Failure(OIDNW_ERROR_TIMEOUT, "GPU wait timed out".into()));
                }
                std::thread::sleep(Duration::from_micros(100));
            }
        })
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oidnw_gpu_device_info(
    device: *const GpuDevice,
    name: *mut c_char,
    name_capacity: usize,
    backend: *mut u32,
    features: *mut u32,
) -> i32 {
    unsafe {
        guard(std::ptr::null_mut(), 0, || {
            let device = required(device, "device")?;
            let _call = exclusive(&device.gate, "device handle is busy")?;
            with_runtime(&device.state, |runtime| {
                write_error(
                    name,
                    name_capacity,
                    &runtime.context.adapter.get_info().name,
                );
                if !backend.is_null() {
                    check_pointer(backend, "backend output")?;
                    backend.write(match runtime.context.adapter.get_info().backend {
                        wgpu::Backend::Metal => OIDNW_BACKEND_METAL,
                        wgpu::Backend::Vulkan => OIDNW_BACKEND_VULKAN,
                        wgpu::Backend::Dx12 => OIDNW_BACKEND_DX12,
                        _ => OIDNW_BACKEND_AUTO,
                    });
                }
                if !features.is_null() {
                    check_pointer(features, "features output")?;
                    let enabled = runtime.context.device.features();
                    features.write(
                        u32::from(enabled.contains(wgpu::Features::SHADER_F16))
                            | (u32::from(enabled.contains(wgpu::Features::SUBGROUP)) << 1),
                    );
                }
                Ok(())
            })
        })
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oidnw_gpu_get_limits(
    device: *const GpuDevice,
    limits: *mut DeviceLimits,
    error: *mut c_char,
    capacity: usize,
) -> i32 {
    unsafe {
        guard(error, capacity, || {
            // The result is output-only; do not read its uninitialized body.
            check_pointer(limits, "limits output")?;
            let device = required(device, "device")?;
            let _call = exclusive(&device.gate, "device handle is busy")?;
            with_runtime(&device.state, |runtime| {
                let supported = runtime.context.device.limits();
                limits.write(DeviceLimits {
                    image_offset_alignment: supported.min_storage_buffer_offset_alignment,
                    max_buffer_size: supported.max_buffer_size,
                    max_image_bytes: supported
                        .max_buffer_size
                        .min(supported.max_storage_buffer_binding_size)
                        .min(u32::MAX as u64),
                    max_pending_submissions: runtime.ledger.capacity as u32,
                });
                Ok(())
            })
        })
    }
}
unsafe fn destroy_device(device: *mut GpuDevice, quiescent: bool) -> Outcome<()> {
    unsafe {
        if device.is_null() {
            return Ok(());
        }
        let device_ref = required(device, "device")?;
        let (call, poisoned) = match device_ref.gate.try_lock() {
            Ok(call) => (call, false),
            Err(std::sync::TryLockError::WouldBlock) => {
                return Err(Failure(OIDNW_ERROR_BUSY, "device handle is busy".into()));
            }
            Err(std::sync::TryLockError::Poisoned(error)) => (error.into_inner(), true),
        };
        let state = device_ref.state.clone();
        {
            let mut slot = state
                .runtime
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let _queue = state.callbacks.enter();
            let runtime = slot
                .as_mut()
                .ok_or_else(|| invalid("device was destroyed"))?;
            if poisoned || state.runtime.is_poisoned() {
                runtime.lost("previous operation panicked; submission outcome may be uncertain");
            }
            let _maintenance = runtime.maintain();
            if state.children.load(Ordering::Acquire) != 0 {
                return Err(Failure(
                    OIDNW_ERROR_BUSY,
                    "destroy all device-owned handles before the device".into(),
                ));
            }
            if quiescent {
                let uncertain = runtime
                    .health
                    .lock()
                    .map_err(|_| gpu("health lock poisoned"))?
                    .failure
                    .as_ref()
                    .is_some_and(|f| f.0 == OIDNW_ERROR_DEVICE_LOST);
                if !uncertain {
                    return Err(invalid(
                        "quiescent teardown requires a lost or uncertain device",
                    ));
                }
                // The caller guarantees GPU access has ceased. Never infer
                // physical queue quiescence from a wgpu error alone.
                runtime.ledger.pending.clear();
            } else if !runtime.ledger.pending.is_empty() {
                return Err(Failure(OIDNW_ERROR_BUSY, "submissions are in flight or uncertain; poll/wait, or establish host GPU quiescence before quiescent teardown".into()));
            }
            drop(slot.take());
        }
        drop(call);
        drop(Box::from_raw(device));
        Ok(())
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oidnw_gpu_device_destroy(
    device: *mut GpuDevice,
    error: *mut c_char,
    capacity: usize,
) -> i32 {
    unsafe { guard(error, capacity, || destroy_device(device, false)) }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oidnw_gpu_device_destroy_quiescent(
    device: *mut GpuDevice,
    error: *mut c_char,
    capacity: usize,
) -> i32 {
    unsafe { guard(error, capacity, || destroy_device(device, true)) }
}
macro_rules! destroy_handle {
    ($name:ident, $ty:ty) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name(pointer: *mut $ty) {
            if !pointer.is_null() {
                let _ = catch_unwind(AssertUnwindSafe(|| unsafe {
                    drop(Box::from_raw(pointer));
                }));
            }
        }
    };
}
destroy_handle!(oidnw_gpu_model_destroy, GpuModel);
destroy_handle!(oidnw_gpu_filter_destroy, GpuFilter);
destroy_handle!(oidnw_gpu_prepared_destroy, GpuPrepared);
destroy_handle!(oidnw_gpu_buffer_destroy, GpuBuffer);
destroy_handle!(oidnw_gpu_sync_destroy, GpuSync);
