use super::*;
use ash::{vk, vk::Handle};

#[repr(C)]
#[derive(Clone, Copy)]
pub struct VulkanDeviceDesc {
    pub flags: u32,
    pub get_instance_proc_addr: Option<vk::PFN_vkGetInstanceProcAddr>,
    pub instance: vk::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: vk::Device,
    pub queue: vk::Queue,
    pub queue_family_index: u32,
    pub queue_index: u32,
    pub instance_create_info: *const vk::InstanceCreateInfo<'static>,
    pub device_create_info: *const vk::DeviceCreateInfo<'static>,
    pub lock: QueueLock,
    pub unlock: QueueLock,
    pub user_data: *mut c_void,
    pub max_pending_submissions: u32,
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn oidnw_gpu_vulkan_device_desc_default(desc: *mut VulkanDeviceDesc) {
    if !desc.is_null() && desc.is_aligned() {
        let mut value: VulkanDeviceDesc = unsafe { std::mem::zeroed() };
        value.max_pending_submissions = 64;
        unsafe {
            desc.write(value);
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn oidnw_gpu_device_import_vulkan(
    desc: *const VulkanDeviceDesc,
    result: *mut *mut GpuDevice,
    error: *mut c_char,
    capacity: usize,
) -> i32 {
    unsafe {
        guard(error, capacity, || {
            output(result)?;
            let desc = descriptor(desc)?;
            if desc.flags & !3 != 0
                || !(1..=4096).contains(&desc.max_pending_submissions)
                || desc.lock.is_none()
                || desc.unlock.is_none()
                || desc.instance == vk::Instance::null()
                || desc.device == vk::Device::null()
                || desc.physical_device == vk::PhysicalDevice::null()
                || desc.queue == vk::Queue::null()
                || desc.get_instance_proc_addr.is_none()
            {
                return Err(invalid(
                    "invalid Vulkan description; queue lock callbacks are required",
                ));
            }
            let callbacks = Callbacks {
                lock: desc.lock,
                unlock: desc.unlock,
                user: desc.user_data as usize,
            };
            let _queue = callbacks.enter();
            let context = platform_import(&desc)?;
            result.write(Box::into_raw(Box::new(GpuDevice {
                state: context_state(context, callbacks, desc.max_pending_submissions as usize),
                gate: Mutex::new(()),
            })));
            Ok(())
        })
    }
}

#[cfg(not(any(target_os = "windows", target_os = "linux")))]
unsafe fn platform_import(_: &VulkanDeviceDesc) -> Outcome<Context> {
    Err(Failure(
        OIDNW_ERROR_UNAVAILABLE,
        "Vulkan interop requires Windows or Linux".into(),
    ))
}

#[cfg(any(target_os = "windows", target_os = "linux"))]
#[path = "vulkan_features.rs"]
mod features;

#[cfg(any(target_os = "windows", target_os = "linux"))]
unsafe fn platform_import(desc: &VulkanDeviceDesc) -> Outcome<Context> {
    unsafe {
        let create = required(desc.instance_create_info, "instance create info")?;
        let app = required(create.p_application_info, "application info")?;
        if create.s_type != vk::StructureType::INSTANCE_CREATE_INFO
            || app.api_version < vk::API_VERSION_1_2
        {
            return Err(invalid("Vulkan 1.2 or newer instance is required"));
        }
        let entry = ash::Entry::from_static_fn(ash::StaticFn {
            get_instance_proc_addr: desc.get_instance_proc_addr.unwrap(),
        });
        let instance = ash::Instance::load(entry.static_fn(), desc.instance);
        if !instance
            .enumerate_physical_devices()
            .map_err(vk_error)?
            .contains(&desc.physical_device)
        {
            return Err(invalid("physical device does not belong to instance"));
        }
        if instance
            .get_physical_device_properties(desc.physical_device)
            .api_version
            < vk::API_VERSION_1_2
        {
            return Err(invalid("physical device must support Vulkan 1.2 or newer"));
        }
        let flags = wgpu::InstanceFlags::empty();
        let extensions =
            wgpu::hal::vulkan::Instance::desired_extensions(&entry, app.api_version, flags)
                .map_err(|e| gpu(e.to_string()))?;
        let actual = checked_slice(
            create.pp_enabled_extension_names,
            create.enabled_extension_count as usize,
        )?;
        for extension in &extensions {
            if !actual
                .iter()
                .any(|p| !p.is_null() && std::ffi::CStr::from_ptr(*p) == *extension)
            {
                return Err(invalid(format!(
                    "missing HAL-required instance extension {extension:?}"
                )));
            }
        }
        let families = instance.get_physical_device_queue_family_properties(desc.physical_device);
        let family = families
            .get(desc.queue_family_index as usize)
            .ok_or_else(|| invalid("invalid queue family"))?;
        if !family
            .queue_flags
            .contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
            || desc.queue_index >= family.queue_count
        {
            return Err(invalid("queue must support graphics and compute"));
        }
        let native = ash::Device::load(instance.fp_v1_0(), desc.device);
        let hal_instance = wgpu::hal::vulkan::Instance::from_raw(
            entry,
            instance,
            app.api_version,
            0,
            None,
            extensions,
            flags,
            Default::default(),
            false,
            Some(Box::new(|| {})),
        )
        .map_err(|e| gpu(e.to_string()))?;
        let exposed = hal_instance
            .expose_adapter(desc.physical_device)
            .ok_or_else(|| invalid("HAL cannot expose physical device"))?;
        let wg_instance = wgpu::Instance::from_hal::<wgpu::hal::api::Vulkan>(hal_instance);
        let adapter = wg_instance.create_adapter_from_hal(exposed);
        let mut requested = Denoiser::recommended_features(&adapter);
        if desc.flags & 1 != 0 {
            requested.remove(wgpu::Features::SHADER_F16);
        }
        if desc.flags & 2 != 0 {
            requested.remove(wgpu::Features::SUBGROUP);
        }
        let mut limits = adapter.limits();
        limits.max_buffer_size = limits.max_buffer_size.min(u32::MAX as u64);
        let hal = adapter
            .as_hal::<wgpu::hal::api::Vulkan>()
            .ok_or_else(|| gpu("not Vulkan"))?;
        let device_extensions = hal.required_device_extensions(requested);
        let mut requirements = hal.physical_device_features(&device_extensions, requested);
        let required_create = requirements.add_to_device_create(vk::DeviceCreateInfo::default());
        let required_features =
            features::snapshot(&required_create).map_err(|e| invalid(e.to_string()))?;
        let device_create = required(desc.device_create_info, "device create info")?;
        if device_create.s_type != vk::StructureType::DEVICE_CREATE_INFO {
            return Err(invalid("invalid device create info"));
        }
        checked_slice(
            device_create.pp_enabled_extension_names,
            device_create.enabled_extension_count as usize,
        )?;
        checked_slice(
            device_create.p_queue_create_infos,
            device_create.queue_create_info_count as usize,
        )?;
        features::validate_create(
            &required_features,
            &device_extensions,
            device_create,
            desc.queue_family_index,
            desc.queue_index,
        )
        .map_err(|e| invalid(e.to_string()))?;
        let actual = features::snapshot(device_create).map_err(|e| invalid(e.to_string()))?;
        if actual
            .get("PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES")
            .and_then(|f| f.first())
            != Some(&1)
        {
            return Err(invalid(
                "timelineSemaphore must be enabled with an individual feature struct",
            ));
        }
        if native.get_device_queue(desc.queue_family_index, desc.queue_index) != desc.queue {
            return Err(invalid("queue does not match device/family/index"));
        }
        let open = hal
            .device_from_raw(
                native.clone(),
                Some(Box::new(|| {})),
                &device_extensions,
                requested,
                &limits,
                &Default::default(),
                desc.queue_family_index,
                desc.queue_index,
            )
            .map_err(|e| gpu(e.to_string()))?;
        drop(hal);
        let (device, queue) = adapter
            .create_device_from_hal::<wgpu::hal::api::Vulkan>(
                open,
                &wgpu::DeviceDescriptor {
                    label: Some("oidn host Vulkan device"),
                    required_features: requested,
                    required_limits: limits,
                    ..Default::default()
                },
            )
            .map_err(|e| gpu(e.to_string()))?;
        Ok(Context {
            adapter,
            device,
            queue,
            backend: Box::new(Vulkan { raw: native }),
        })
    }
}

unsafe fn checked_slice<'a, T>(ptr: *const T, count: usize) -> Outcome<&'a [T]> {
    if count == 0 {
        return Ok(&[]);
    }
    if count > 4096 {
        return Err(invalid("Vulkan declaration array exceeds 4096 entries"));
    }
    check_pointer(ptr, "Vulkan declaration array")?;
    Ok(unsafe { std::slice::from_raw_parts(ptr, count) })
}
fn vk_error(error: vk::Result) -> Failure {
    Failure(
        OIDNW_ERROR_DEVICE_LOST,
        format!("Vulkan operation failed: {error:?}"),
    )
}

#[cfg(any(target_os = "windows", target_os = "linux"))]
struct Vulkan {
    raw: ash::Device,
}
#[cfg(any(target_os = "windows", target_os = "linux"))]
struct Semaphore {
    raw: ash::Device,
    semaphore: vk::Semaphore,
}
#[cfg(any(target_os = "windows", target_os = "linux"))]
impl Backend for Vulkan {
    fn check_admission(&self, ledger: &Ledger<Pending>) -> Outcome<()> {
        // HAL 30 chains submissions using binary relay semaphores. Chaining
        // behind a not-yet-submitted transitive CPU timeline signal violates
        // Vulkan 03238. Vulkan cannot query whether a future signal has been
        // submitted, so conservatively require prior waits to have completed.
        for (_, pending) in &ledger.pending {
            for point in &pending._waits {
                if point.sync.native.value()? < point.value {
                    return Err(Failure(OIDNW_ERROR_QUEUE_FULL,
                        "Vulkan relay admission: a previous submission still has unresolved external waits".into()));
                }
            }
        }
        Ok(())
    }
    unsafe fn buffer(
        &self,
        device: &wgpu::Device,
        raw: *mut c_void,
        size: u64,
    ) -> Outcome<wgpu::Buffer> {
        crate::denoiser::gpu_checked(device, || {
            Ok(unsafe {
                device.create_buffer_from_hal::<wgpu::hal::api::Vulkan>(
                    wgpu::hal::vulkan::Buffer::from_raw_externally_owned(
                        vk::Buffer::from_raw(raw as u64),
                        Box::new(|| {}),
                    ),
                    &wgpu::BufferDescriptor {
                        label: Some("oidn host VkBuffer"),
                        size,
                        usage: wgpu::BufferUsages::STORAGE,
                        mapped_at_creation: false,
                    },
                )
            })
        })
        .map_err(denoise_error)
    }
    unsafe fn timeline(&self, raw: *mut c_void) -> Outcome<Box<dyn Timeline>> {
        let semaphore = Semaphore {
            raw: self.raw.clone(),
            semaphore: vk::Semaphore::from_raw(raw as u64),
        };
        semaphore.value()?;
        Ok(Box::new(semaphore))
    }
}
#[cfg(any(target_os = "windows", target_os = "linux"))]
impl Timeline for Semaphore {
    fn value(&self) -> Outcome<u64> {
        unsafe { self.raw.get_semaphore_counter_value(self.semaphore) }.map_err(vk_error)
    }
    fn stage_wait(&self, queue: &wgpu::Queue, value: u64) -> Outcome<()> {
        let hal =
            unsafe { queue.as_hal::<wgpu::hal::api::Vulkan>() }.ok_or_else(|| gpu("not Vulkan"))?;
        hal.add_wait_semaphore(
            self.semaphore,
            Some(value),
            vk::PipelineStageFlags::ALL_COMMANDS,
        );
        Ok(())
    }
    fn stage_signal(&self, queue: &wgpu::Queue, value: u64) -> Outcome<()> {
        let hal =
            unsafe { queue.as_hal::<wgpu::hal::api::Vulkan>() }.ok_or_else(|| gpu("not Vulkan"))?;
        hal.add_signal_semaphore(self.semaphore, Some(value));
        Ok(())
    }
    fn unstage(&self, queue: &wgpu::Queue) {
        if let Some(hal) = unsafe { queue.as_hal::<wgpu::hal::api::Vulkan>() } {
            hal.remove_wait_semaphore(self.semaphore);
            hal.remove_signal_semaphore(self.semaphore);
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn oidnw_gpu_buffer_import_vulkan(
    device: *mut GpuDevice,
    buffer: vk::Buffer,
    create: *const vk::BufferCreateInfo<'_>,
    result: *mut *mut GpuBuffer,
    error: *mut c_char,
    capacity: usize,
) -> i32 {
    unsafe {
        guard(error, capacity, || {
            output(result)?;
            let desc = required(create, "buffer create info")?;
            if desc.s_type != vk::StructureType::BUFFER_CREATE_INFO
                || !desc.flags.is_empty()
                || !desc.p_next.is_null()
                || !desc.usage.contains(vk::BufferUsageFlags::STORAGE_BUFFER)
            {
                return Err(invalid(
                    "expected an unprotected, non-sparse storage buffer declaration without pNext",
                ));
            }
            let gpu = required(device, "device")?;
            with_runtime(&gpu.state, |r| {
                ensure_backend(r, wgpu::Backend::Vulkan)?;
                if desc.sharing_mode == vk::SharingMode::CONCURRENT {
                    let families = checked_slice(
                        desc.p_queue_family_indices,
                        desc.queue_family_index_count as usize,
                    )?;
                    #[cfg(any(target_os = "windows", target_os = "linux"))]
                    if !families.contains(
                        &r.context
                            .device
                            .as_hal::<wgpu::hal::api::Vulkan>()
                            .unwrap()
                            .queue_family_index(),
                    ) {
                        return Err(invalid("concurrent buffer omits imported queue family"));
                    }
                } else if desc.sharing_mode != vk::SharingMode::EXCLUSIVE {
                    return Err(invalid("invalid sharing mode"));
                }
                Ok(())
            })?;
            import_buffer(
                gpu,
                buffer.as_raw() as *mut c_void,
                desc.size,
                result,
                wgpu::Backend::Vulkan,
            )
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn oidnw_gpu_sync_import_vulkan(
    device: *mut GpuDevice,
    semaphore: vk::Semaphore,
    create: *const vk::SemaphoreCreateInfo<'_>,
    result: *mut *mut GpuSync,
    error: *mut c_char,
    capacity: usize,
) -> i32 {
    unsafe {
        guard(error, capacity, || {
            output(result)?;
            let desc = required(create, "semaphore create info")?;
            let kind = required(
                desc.p_next.cast::<vk::SemaphoreTypeCreateInfo>(),
                "semaphore type info",
            )?;
            if desc.s_type != vk::StructureType::SEMAPHORE_CREATE_INFO
                || !desc.flags.is_empty()
                || kind.s_type != vk::StructureType::SEMAPHORE_TYPE_CREATE_INFO
                || !kind.p_next.is_null()
                || kind.semaphore_type != vk::SemaphoreType::TIMELINE
            {
                return Err(invalid(
                    "expected a timeline semaphore declaration without other pNext structs",
                ));
            }
            let gpu = required(device, "device")?;
            with_runtime(&gpu.state, |r| ensure_backend(r, wgpu::Backend::Vulkan))?;
            import_sync(
                gpu,
                semaphore.as_raw() as *mut c_void,
                result,
                wgpu::Backend::Vulkan,
            )
        })
    }
}
