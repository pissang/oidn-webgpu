use super::*;
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Dx12DeviceDesc {
    pub next: *const std::ffi::c_void,
    pub flags: u32,
    pub host_device: *mut c_void,
    pub agility_sdk_version: u32,
    pub agility_sdk_path: *const c_char,
    pub max_pending_submissions: u32,
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oidnw_gpu_dx12_device_desc_default(desc: *mut Dx12DeviceDesc) {
    if !desc.is_null() && desc.is_aligned() {
        let mut value: Dx12DeviceDesc = unsafe { std::mem::zeroed() };
        value.max_pending_submissions = 64;
        unsafe {
            desc.write(value);
        }
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oidnw_gpu_device_create_dx12(
    desc: *const Dx12DeviceDesc,
    result: *mut *mut GpuDevice,
    error: *mut c_char,
    capacity: usize,
) -> i32 {
    unsafe {
        guard(error, capacity, || {
            output(result)?;
            let desc = descriptor(desc)?;
            if desc.host_device.is_null()
                || desc.agility_sdk_path.is_null()
                || desc.agility_sdk_version == 0
                || desc.flags & !15 != 0
                || (desc.flags & 8 != 0 && desc.flags & 4 == 0)
                || !(1..=4096).contains(&desc.max_pending_submissions)
            {
                return Err(invalid("invalid DX12 device description"));
            }
            let callbacks = Callbacks {
                lock: None,
                unlock: None,
                user: 0,
            };
            let context = platform_import(&desc)?;
            result.write(Box::into_raw(Box::new(GpuDevice {
                state: context_state(context, callbacks, desc.max_pending_submissions as usize),
                gate: Mutex::new(()),
            })));
            Ok(())
        })
    }
}
#[cfg(not(target_os = "windows"))]
unsafe fn platform_import(_: &Dx12DeviceDesc) -> Outcome<Context> {
    Err(Failure(
        OIDNW_ERROR_UNAVAILABLE,
        "DX12 requires Windows".into(),
    ))
}
#[cfg(not(target_os = "windows"))]
unsafe fn buffer_size(_: *mut c_void) -> Outcome<u64> {
    Err(Failure(
        OIDNW_ERROR_UNAVAILABLE,
        "DX12 requires Windows".into(),
    ))
}

// Kept independently testable: this sentinel is not a successful fence value.
#[cfg(any(target_os = "windows", test))]
fn fence_value(value: u64) -> Outcome<u64> {
    if value == u64::MAX {
        Err(Failure(
            OIDNW_ERROR_DEVICE_LOST,
            "DX12 fence reports device removal (UINT64_MAX)".into(),
        ))
    } else {
        Ok(value)
    }
}
#[cfg(target_os = "windows")]
mod platform {
    use super::*;
    use windows::{
        Win32::{
            Foundation::{CloseHandle, HANDLE},
            Graphics::Direct3D12::*,
        },
        core::{IUnknown, Interface, PCWSTR},
    };
    type Api = wgpu::hal::api::Dx12;
    struct Dx12 {
        host: ID3D12Device,
        raw: ID3D12Device,
        diagnostics: Vec<(&'static str, ID3D12InfoQueue)>,
        _layers: Option<libloading::Library>,
    }
    struct Fence {
        opened: ID3D12Fence,
        _host: ID3D12Fence,
    }
    struct Handle(HANDLE);
    impl Drop for Handle {
        fn drop(&mut self) {
            unsafe {
                let _ = CloseHandle(self.0);
            }
        }
    }
    fn native_error(error: windows::core::Error) -> Failure {
        gpu(format!("DX12: {error}"))
    }
    unsafe fn borrowed<T: Interface + Clone>(raw: *mut c_void) -> Outcome<T> {
        let unknown = unsafe { IUnknown::from_raw_borrowed(&raw) }
            .ok_or_else(|| invalid("COM object is NULL"))?;
        unknown
            .cast::<T>()
            .map_err(|e| invalid(format!("incompatible COM interface: {e}")))
    }
    pub(super) unsafe fn import(desc: &Dx12DeviceDesc) -> Outcome<Context> {
        unsafe {
            let host = borrowed::<ID3D12Device>(desc.host_device)?;
            let path = std::ffi::CStr::from_ptr(desc.agility_sdk_path)
                .to_str()
                .map_err(|_| invalid("SDK path must be UTF-8"))?;
            if !std::path::Path::new(path).is_absolute() || !path.ends_with(['/', '\\']) {
                return Err(invalid(
                    "SDK path must be an absolute directory ending with a slash",
                ));
            }
            let host_config = host
                .cast::<ID3D12DeviceConfiguration>()
                .map_err(native_error)?
                .GetDesc();
            if host_config.SDKVersion != desc.agility_sdk_version {
                return Err(invalid(
                    "host device and requested Agility SDK versions differ",
                ));
            }
            let layers = if desc.flags & 4 != 0 {
                Some(
                    libloading::Library::new(std::path::Path::new(path).join("d3d12SDKLayers.dll"))
                        .map_err(|e| {
                            Failure(
                                OIDNW_ERROR_UNAVAILABLE,
                                format!("Agility debug layers: {e}"),
                            )
                        })?,
                )
            } else {
                None
            };
            let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
            descriptor.backends = wgpu::Backends::DX12;
            descriptor.flags = wgpu::InstanceFlags::empty();
            if desc.flags & 4 != 0 {
                descriptor.flags |= wgpu::InstanceFlags::DEBUG | wgpu::InstanceFlags::VALIDATION;
            }
            if desc.flags & 8 != 0 {
                descriptor.flags |= wgpu::InstanceFlags::GPU_BASED_VALIDATION;
            }
            descriptor.backend_options.dx12.agility_sdk = Some(wgpu::wgt::Dx12AgilitySDK {
                sdk_version: desc.agility_sdk_version,
                sdk_path: path.into(),
                on_load_failure: wgpu::wgt::Dx12AgilitySDKLoadFailure::Error,
            });
            let instance = wgpu::Instance::new(descriptor);
            let luid = host.GetAdapterLuid();
            let adapter = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::DX12))
                .into_iter()
                .find(|a| {
                    a.as_hal::<Api>().is_some_and(|hal| {
                        hal.raw_adapter().GetDesc().is_ok_and(|d| {
                            d.AdapterLuid.LowPart == luid.LowPart
                                && d.AdapterLuid.HighPart == luid.HighPart
                        })
                    })
                })
                .ok_or_else(|| {
                    Failure(
                        OIDNW_ERROR_UNAVAILABLE,
                        "no DX12 adapter matches host LUID".into(),
                    )
                })?;
            let mut features = Denoiser::recommended_features(&adapter);
            if desc.flags & 1 != 0 {
                features.remove(wgpu::Features::SHADER_F16);
            }
            if desc.flags & 2 != 0 {
                features.remove(wgpu::Features::SUBGROUP);
            }
            let (device, queue) =
                pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                    label: Some("oidn independent DX12 device"),
                    required_features: features,
                    required_limits: Denoiser::required_limits(),
                    ..Default::default()
                }))
                .map_err(|e| gpu(e.to_string()))?;
            let raw = device
                .as_hal::<Api>()
                .ok_or_else(|| gpu("not DX12"))?
                .raw_device()
                .clone();
            let imported_luid = raw.GetAdapterLuid();
            if imported_luid.LowPart != luid.LowPart
                || imported_luid.HighPart != luid.HighPart
                || raw.as_raw() == host.as_raw()
            {
                return Err(invalid(
                    "expected distinct DX12 devices on the same adapter LUID",
                ));
            }
            for (name, d) in [("host", &host), ("library", &raw)] {
                let config = d
                    .cast::<ID3D12DeviceConfiguration>()
                    .map_err(native_error)?
                    .GetDesc();
                if config.SDKVersion != desc.agility_sdk_version {
                    return Err(invalid("library Agility SDK mismatch"));
                }
                if desc.flags & 4 != 0
                    && !config.Flags.contains(D3D12_DEVICE_FLAG_DEBUG_LAYER_ENABLED)
                {
                    return Err(invalid(format!("{name} device did not enable debug layer")));
                }
                if desc.flags & 8 != 0
                    && !config
                        .Flags
                        .contains(D3D12_DEVICE_FLAG_GPU_BASED_VALIDATION_ENABLED)
                {
                    return Err(invalid(format!(
                        "{name} device did not enable GPU validation"
                    )));
                }
            }
            let diagnostics = if desc.flags & 4 != 0 {
                vec![
                    (
                        "host",
                        host.cast::<ID3D12InfoQueue>().map_err(native_error)?,
                    ),
                    (
                        "library",
                        raw.cast::<ID3D12InfoQueue>().map_err(native_error)?,
                    ),
                ]
            } else {
                Vec::new()
            };
            Ok(Context {
                adapter,
                device,
                queue,
                backend: Box::new(Dx12 {
                    host,
                    raw,
                    diagnostics,
                    _layers: layers,
                }),
            })
        }
    }
    impl Dx12 {
        unsafe fn owner(&self, child: &impl Interface) -> Outcome<()> {
            // Both resources and fences inherit ID3D12DeviceChild.
            let child = child.cast::<ID3D12DeviceChild>().map_err(native_error)?;
            let mut owner = None;
            unsafe { child.GetDevice::<ID3D12Device>(&mut owner) }.map_err(native_error)?;
            if owner
                .as_ref()
                .is_none_or(|d| d.as_raw() != self.host.as_raw())
            {
                return Err(invalid(
                    "object does not belong to the exporting host device",
                ));
            }
            Ok(())
        }
        unsafe fn open<T: Interface>(&self, child: &impl Interface) -> Outcome<T> {
            unsafe {
                self.owner(child)?;
            }
            let child = child.cast::<ID3D12DeviceChild>().map_err(native_error)?;
            let handle = Handle(
                unsafe {
                    self.host
                        .CreateSharedHandle(&child, None, 0x10000000, PCWSTR::null())
                }
                .map_err(native_error)?,
            );
            let mut result = None;
            unsafe { self.raw.OpenSharedHandle::<T>(handle.0, &mut result) }
                .map_err(native_error)?;
            result.ok_or_else(|| gpu("OpenSharedHandle returned NULL"))
        }
    }
    impl Backend for Dx12 {
        fn check_health(&self) -> Outcome<()> {
            unsafe {
                self.host.GetDeviceRemovedReason().map_err(native_error)?;
                self.raw.GetDeviceRemovedReason().map_err(native_error)?;
                for (name, info) in &self.diagnostics {
                    for index in 0..info.GetNumStoredMessagesAllowedByRetrievalFilter() {
                        let mut bytes = 0;
                        info.GetMessage(index, None, &mut bytes)
                            .map_err(native_error)?;
                        let mut storage =
                            vec![0usize; bytes.div_ceil(std::mem::size_of::<usize>())];
                        let message = storage.as_mut_ptr().cast::<D3D12_MESSAGE>();
                        info.GetMessage(index, Some(message), &mut bytes)
                            .map_err(native_error)?;
                        if (*message).Severity == D3D12_MESSAGE_SEVERITY_ERROR
                            || (*message).Severity == D3D12_MESSAGE_SEVERITY_CORRUPTION
                        {
                            let description =
                                std::ffi::CStr::from_ptr((*message).pDescription.cast())
                                    .to_string_lossy();
                            return Err(gpu(format!("{name} DX12 validation: {description}")));
                        }
                    }
                }
                Ok(())
            }
        }
        unsafe fn identity(&self, raw: *mut c_void) -> Outcome<usize> {
            Ok(unsafe { borrowed::<IUnknown>(raw)? }.as_raw() as usize)
        }
        unsafe fn retain_buffer(&self, raw: *mut c_void) -> Outcome<Option<Box<dyn Send + Sync>>> {
            Ok(Some(Box::new(unsafe { borrowed::<ID3D12Resource>(raw)? })))
        }
        unsafe fn buffer(
            &self,
            device: &wgpu::Device,
            raw: *mut c_void,
            size: u64,
        ) -> Outcome<wgpu::Buffer> {
            unsafe {
                let host = borrowed::<ID3D12Resource>(raw)?;
                self.owner(&host)?;
                validate_buffer(&host)?;
                if host.GetDesc().Width != size {
                    return Err(invalid("DX12 buffer size mismatch"));
                }
                let opened = self.open::<ID3D12Resource>(&host)?;
                crate::denoiser::gpu_checked(device, || {
                    Ok(device.create_buffer_from_hal::<Api>(
                        wgpu::hal::dx12::Device::buffer_from_raw(opened, size),
                        &wgpu::BufferDescriptor {
                            label: Some("oidn shared host DX12 buffer"),
                            size,
                            usage: wgpu::BufferUsages::STORAGE,
                            mapped_at_creation: false,
                        },
                    ))
                })
                .map_err(denoise_error)
            }
        }
        unsafe fn timeline(&self, raw: *mut c_void) -> Outcome<Box<dyn Timeline>> {
            let host = unsafe { borrowed::<ID3D12Fence>(raw)? };
            let flags = unsafe {
                host.cast::<ID3D12Fence1>()
                    .map_err(native_error)?
                    .GetCreationFlags()
            };
            if !flags.contains(D3D12_FENCE_FLAG_SHARED)
                || flags.contains(D3D12_FENCE_FLAG_SHARED_CROSS_ADAPTER)
            {
                return Err(invalid("expected a shared fence on the same adapter"));
            }
            let opened = unsafe { self.open::<ID3D12Fence>(&host)? };
            let fence = Fence {
                opened,
                _host: host,
            };
            fence.value()?;
            Ok(Box::new(fence))
        }
    }
    unsafe fn validate_buffer(host: &ID3D12Resource) -> Outcome<()> {
        let desc = unsafe { host.GetDesc() };
        let mut heap = D3D12_HEAP_PROPERTIES::default();
        let mut flags = D3D12_HEAP_FLAGS::default();
        unsafe { host.GetHeapProperties(Some(&mut heap), Some(&mut flags)) }
            .map_err(native_error)?;
        if desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER
            || desc.Layout != D3D12_TEXTURE_LAYOUT_ROW_MAJOR
            || desc.Flags != D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
            || heap.Type != D3D12_HEAP_TYPE_DEFAULT
            || !flags.contains(D3D12_HEAP_FLAG_SHARED)
            || flags.contains(D3D12_HEAP_FLAG_SHARED_CROSS_ADAPTER)
        {
            return Err(invalid(
                "expected shared committed DEFAULT UAV buffer on the same adapter",
            ));
        }
        // CreateSharedHandle below rejects placed/reserved resources; successful
        // export of a resource handle proves that this is shareable as a resource.
        Ok(())
    }
    pub(super) unsafe fn size(raw: *mut c_void) -> Outcome<u64> {
        let resource = unsafe { borrowed::<ID3D12Resource>(raw)? };
        unsafe {
            validate_buffer(&resource)?;
        }
        Ok(unsafe { resource.GetDesc() }.Width)
    }
    impl Timeline for Fence {
        fn value(&self) -> Outcome<u64> {
            fence_value(unsafe { self.opened.GetCompletedValue() })
        }
        fn stage_wait(&self, queue: &wgpu::Queue, value: u64) -> Outcome<()> {
            let hal = unsafe { queue.as_hal::<Api>() }.ok_or_else(|| gpu("not DX12"))?;
            hal.add_wait_fence(self.opened.clone(), value);
            Ok(())
        }
        fn stage_signal(&self, queue: &wgpu::Queue, value: u64) -> Outcome<()> {
            let hal = unsafe { queue.as_hal::<Api>() }.ok_or_else(|| gpu("not DX12"))?;
            hal.add_signal_fence(self.opened.clone(), value);
            Ok(())
        }
        fn unstage(&self, queue: &wgpu::Queue) {
            if let Some(hal) = unsafe { queue.as_hal::<Api>() } {
                hal.remove_wait_fence(&self.opened);
                hal.remove_signal_fence(&self.opened);
            }
        }
    }
}
#[cfg(target_os = "windows")]
unsafe fn platform_import(desc: &Dx12DeviceDesc) -> Outcome<Context> {
    unsafe { platform::import(desc) }
}
#[cfg(target_os = "windows")]
unsafe fn buffer_size(raw: *mut c_void) -> Outcome<u64> {
    unsafe { platform::size(raw) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn oidnw_gpu_buffer_import_dx12(
    device: *mut GpuDevice,
    raw: *mut c_void,
    result: *mut *mut GpuBuffer,
    error: *mut c_char,
    capacity: usize,
) -> i32 {
    unsafe {
        guard(error, capacity, || {
            output(result)?;
            let device = required(device, "device")?;
            with_runtime(&device.state, |r| ensure_backend(r, wgpu::Backend::Dx12))?;
            let size = buffer_size(raw)?;
            import_buffer(device, raw, size, result, wgpu::Backend::Dx12)
        })
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oidnw_gpu_sync_import_dx12(
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
                wgpu::Backend::Dx12,
            )
        })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn removed_fence_is_never_complete() {
        assert_eq!(
            fence_value(u64::MAX).unwrap_err().0,
            OIDNW_ERROR_DEVICE_LOST
        );
        assert_eq!(fence_value(u64::MAX - 1).unwrap(), u64::MAX - 1);
        assert_eq!(fence_value(0).unwrap(), 0);
    }
}
