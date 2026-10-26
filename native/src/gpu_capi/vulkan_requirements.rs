use super::*;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct VulkanRequirementsDesc {
    pub next: *const std::ffi::c_void,
    pub flags: u32,
    pub get_instance_proc_addr: Option<vk::PFN_vkGetInstanceProcAddr>,
    pub instance: vk::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub instance_create_info: *const vk::InstanceCreateInfo<'static>,
}

pub struct VulkanRequirements {
    core: Box<vk::PhysicalDeviceFeatures>,
    names: Vec<*const c_char>,
    chain: *const c_void,
    enabled_features: u32,
    #[cfg(any(target_os = "windows", target_os = "linux"))]
    _features: Box<wgpu::hal::vulkan::PhysicalDeviceFeatures>,
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn oidnw_gpu_vulkan_requirements_desc_default(
    desc: *mut VulkanRequirementsDesc,
) {
    if !desc.is_null() && desc.is_aligned() {
        unsafe { desc.write(std::mem::zeroed()) };
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn oidnw_gpu_vulkan_instance_extensions(
    get_instance_proc_addr: Option<vk::PFN_vkGetInstanceProcAddr>,
    api_version: u32,
    names: *mut *const c_char,
    count: *mut usize,
    error: *mut c_char,
    capacity: usize,
) -> i32 {
    unsafe {
        guard(error, capacity, || {
            check_pointer(count, "extension count")?;
            let available = if names.is_null() { 0 } else { count.read() };
            count.write(0);
            let get_instance_proc_addr = get_instance_proc_addr
                .ok_or_else(|| invalid("vkGetInstanceProcAddr is required"))?;
            if api_version < vk::API_VERSION_1_2 {
                return Err(invalid("Vulkan 1.2 or newer is required"));
            }
            let extensions = instance_extensions(get_instance_proc_addr, api_version)?;
            count.write(extensions.len());
            if !names.is_null() {
                check_pointer(names, "extension names")?;
                if available < extensions.len() {
                    return Err(invalid("extension name array is too small"));
                }
                for (i, name) in extensions.iter().enumerate() {
                    names.add(i).write(name.as_ptr());
                }
            }
            Ok(())
        })
    }
}

#[cfg(any(target_os = "windows", target_os = "linux"))]
unsafe fn instance_extensions(
    get_instance_proc_addr: vk::PFN_vkGetInstanceProcAddr,
    api_version: u32,
) -> Outcome<Vec<&'static std::ffi::CStr>> {
    let entry = unsafe {
        ash::Entry::from_static_fn(ash::StaticFn {
            get_instance_proc_addr,
        })
    };
    wgpu::hal::vulkan::Instance::desired_extensions(
        &entry,
        api_version,
        wgpu::InstanceFlags::empty(),
    )
    .map_err(|e| gpu(e.to_string()))
}
#[cfg(not(any(target_os = "windows", target_os = "linux")))]
unsafe fn instance_extensions(
    _: vk::PFN_vkGetInstanceProcAddr,
    _: u32,
) -> Outcome<Vec<&'static std::ffi::CStr>> {
    Err(Failure(
        OIDNW_ERROR_UNAVAILABLE,
        "Vulkan interop requires Windows or Linux".into(),
    ))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn oidnw_gpu_vulkan_requirements_create(
    desc: *const VulkanRequirementsDesc,
    result: *mut *mut VulkanRequirements,
    error: *mut c_char,
    capacity: usize,
) -> i32 {
    unsafe {
        guard(error, capacity, || {
            output(result)?;
            let desc = descriptor(desc)?;
            if desc.flags & !7 != 0
                || desc.get_instance_proc_addr.is_none()
                || desc.instance == vk::Instance::null()
                || desc.physical_device == vk::PhysicalDevice::null()
            {
                return Err(invalid("invalid Vulkan requirements description"));
            }
            let requirements = query(&desc)?;
            result.write(Box::into_raw(Box::new(requirements)));
            Ok(())
        })
    }
}

#[cfg(any(target_os = "windows", target_os = "linux"))]
unsafe fn query(desc: &VulkanRequirementsDesc) -> Outcome<VulkanRequirements> {
    unsafe {
        let (adapter, _) = host_adapter(
            desc.get_instance_proc_addr.unwrap(),
            desc.instance,
            desc.physical_device,
            desc.instance_create_info,
        )?;
        let requested = requested_features(adapter.features(), desc.flags);
        let hal = adapter
            .as_hal::<wgpu::hal::api::Vulkan>()
            .ok_or_else(|| gpu("not Vulkan"))?;
        let extensions = hal.required_device_extensions(requested);
        let mut features = Box::new(hal.physical_device_features(&extensions, requested));
        let create = features.add_to_device_create(vk::DeviceCreateInfo::default());
        let snapshot = features::snapshot(&create).map_err(|e| invalid(e.to_string()))?;
        if snapshot
            .get("PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES")
            .and_then(|v| v.first())
            != Some(&1)
        {
            return Err(Failure(
                OIDNW_ERROR_UNAVAILABLE,
                "HAL requirements do not enable timelineSemaphore".into(),
            ));
        }
        // The chain points inside this Box; its contents must not move.
        let chain = create.p_next;
        let core = Box::new(features.get_core());
        Ok(VulkanRequirements {
            core,
            chain,
            names: extensions.iter().map(|n| n.as_ptr()).collect(),
            enabled_features: feature_bits(requested),
            _features: features,
        })
    }
}
#[cfg(not(any(target_os = "windows", target_os = "linux")))]
unsafe fn query(_: &VulkanRequirementsDesc) -> Outcome<VulkanRequirements> {
    Err(Failure(
        OIDNW_ERROR_UNAVAILABLE,
        "Vulkan interop requires Windows or Linux".into(),
    ))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn oidnw_gpu_vulkan_requirements_get(
    requirements: *const VulkanRequirements,
    create_info: *mut vk::DeviceCreateInfo<'_>,
    enabled_features: *mut u32,
    error: *mut c_char,
    capacity: usize,
) -> i32 {
    unsafe {
        guard(error, capacity, || {
            if !create_info.is_null() && create_info.is_aligned() {
                create_info.write(std::mem::zeroed());
            }
            if !enabled_features.is_null() && enabled_features.is_aligned() {
                enabled_features.write(0);
            }
            check_pointer(create_info, "device create info output")?;
            let requirements = required(requirements, "requirements")?;
            if !enabled_features.is_null() {
                check_pointer(enabled_features, "features output")?;
            }
            let mut create = vk::DeviceCreateInfo::default();
            create.p_enabled_features = &*requirements.core;
            create.p_next = requirements.chain;
            create.enabled_extension_count = requirements.names.len() as u32;
            create.pp_enabled_extension_names = if requirements.names.is_empty() {
                std::ptr::null()
            } else {
                requirements.names.as_ptr()
            };
            create_info.write(create);
            if !enabled_features.is_null() {
                enabled_features.write(requirements.enabled_features);
            }
            Ok(())
        })
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oidnw_gpu_vulkan_requirements_destroy(
    requirements: *mut VulkanRequirements,
) {
    if !requirements.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| unsafe {
            drop(Box::from_raw(requirements));
        }));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn declaration_pointers_survive_owner_moves() {
        let requirements = VulkanRequirements {
            core: Box::new(vk::PhysicalDeviceFeatures::default().robust_buffer_access(true)),
            names: vec![c"VK_KHR_timeline_semaphore".as_ptr()],
            chain: std::ptr::null(),
            enabled_features: OIDNW_FEATURE_SHADER_F16,
            #[cfg(any(target_os = "windows", target_os = "linux"))]
            _features: Box::default(),
        };
        unsafe {
            let mut before = std::mem::MaybeUninit::<vk::DeviceCreateInfo>::uninit();
            let mut enabled = 0;
            assert_eq!(
                oidnw_gpu_vulkan_requirements_get(
                    &requirements,
                    before.as_mut_ptr(),
                    &mut enabled,
                    std::ptr::null_mut(),
                    0
                ),
                OIDNW_OK
            );
            let before = before.assume_init();
            let moved = Box::new(requirements);
            let mut after = vk::DeviceCreateInfo::default();
            assert_eq!(
                oidnw_gpu_vulkan_requirements_get(
                    &*moved,
                    &mut after,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    0
                ),
                OIDNW_OK
            );
            assert_eq!(before.p_enabled_features, after.p_enabled_features);
            assert_eq!(
                before.pp_enabled_extension_names,
                after.pp_enabled_extension_names
            );
            assert_eq!((*after.p_enabled_features).robust_buffer_access, vk::TRUE);
            assert_eq!(after.enabled_extension_count, 1);
            assert_eq!(
                std::ffi::CStr::from_ptr(*after.pp_enabled_extension_names),
                c"VK_KHR_timeline_semaphore"
            );
            assert_eq!(enabled, OIDNW_FEATURE_SHADER_F16);
        }
    }
}
