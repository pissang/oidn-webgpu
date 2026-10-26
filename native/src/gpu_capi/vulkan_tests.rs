use super::*;
use ash::vk;
static VALIDATION_ERRORS: AtomicUsize = AtomicUsize::new(0);
unsafe extern "system" fn validation_message(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _: *mut c_void,
) -> vk::Bool32 {
    let text = unsafe { std::ffi::CStr::from_ptr((*data).p_message) }.to_string_lossy();
    let known = unsafe { (*data).message_id_number } == 0xb210f7c2_u32 as i32
        && text.contains("Workgroup")
        && text.contains("ArrayStride");
    if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) && !known {
        VALIDATION_ERRORS.fetch_add(1, Ordering::Relaxed);
    }
    eprintln!("Vulkan state test (known Naga={known}): {text}");
    vk::FALSE
}

#[test]
#[ignore = "requires Vulkan GPU and Khronos validation; controlled encoding failure and quarantine"]
fn vulkan_backpressure_and_failed_batch_preserve_external_completion() {
    unsafe {
        let entry = ash::Entry::load().unwrap();
        VALIDATION_ERRORS.store(0, Ordering::Relaxed);
        let mut count = 0;
        assert_eq!(
            oidnw_gpu_vulkan_instance_extensions(
                Some(entry.static_fn().get_instance_proc_addr),
                vk::API_VERSION_1_3,
                std::ptr::null_mut(),
                &mut count,
                std::ptr::null_mut(),
                0
            ),
            OIDNW_OK
        );
        let mut names = vec![std::ptr::null(); count];
        if count != 0 {
            let mut short = count - 1;
            assert_eq!(
                oidnw_gpu_vulkan_instance_extensions(
                    Some(entry.static_fn().get_instance_proc_addr),
                    vk::API_VERSION_1_3,
                    names.as_mut_ptr(),
                    &mut short,
                    std::ptr::null_mut(),
                    0
                ),
                OIDNW_ERROR_INVALID_ARGUMENT
            );
            assert!(names.iter().all(|n| n.is_null()));
            assert_eq!(short, count);
        }
        assert_eq!(
            oidnw_gpu_vulkan_instance_extensions(
                Some(entry.static_fn().get_instance_proc_addr),
                vk::API_VERSION_1_3,
                names.as_mut_ptr(),
                &mut count,
                std::ptr::null_mut(),
                0
            ),
            OIDNW_OK
        );
        if !names.contains(&c"VK_EXT_debug_utils".as_ptr()) {
            names.push(c"VK_EXT_debug_utils".as_ptr());
        }
        let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);
        let layers = [c"VK_LAYER_KHRONOS_validation".as_ptr()];
        let instance_create = vk::InstanceCreateInfo::default()
            .application_info(&app)
            .enabled_extension_names(&names)
            .enabled_layer_names(&layers);
        let instance = entry.create_instance(&instance_create, None).unwrap();
        let debug = ash::ext::debug_utils::Instance::new(&entry, &instance);
        let messenger = debug
            .create_debug_utils_messenger(
                &vk::DebugUtilsMessengerCreateInfoEXT::default()
                    .message_severity(
                        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                            | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING,
                    )
                    .message_type(
                        vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                            | vk::DebugUtilsMessageTypeFlagsEXT::GENERAL,
                    )
                    .pfn_user_callback(Some(validation_message)),
                None,
            )
            .unwrap();
        let physical = instance
            .enumerate_physical_devices()
            .unwrap()
            .into_iter()
            .find(|&p| {
                std::ffi::CStr::from_ptr(
                    instance
                        .get_physical_device_properties(p)
                        .device_name
                        .as_ptr(),
                )
                .to_string_lossy()
                .contains("RTX 4060")
            })
            .expect("RTX 4060 required");
        let family = instance
            .get_physical_device_queue_family_properties(physical)
            .iter()
            .position(|q| {
                q.queue_flags
                    .contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
            })
            .unwrap() as u32;
        let query = VulkanRequirementsDesc {
            next: std::ptr::null(),
            flags: 0,
            get_instance_proc_addr: Some(entry.static_fn().get_instance_proc_addr),
            instance: instance.handle(),
            physical_device: physical,
            instance_create_info: (&instance_create as *const vk::InstanceCreateInfo).cast(),
        };
        let mut requirements = std::ptr::null_mut();
        let mut error = [0i8; 4096];
        assert_eq!(
            oidnw_gpu_vulkan_requirements_create(
                &query,
                &mut requirements,
                error.as_mut_ptr(),
                error.len()
            ),
            OIDNW_OK,
            "{}",
            std::ffi::CStr::from_ptr(error.as_ptr()).to_string_lossy()
        );
        let mut create = vk::DeviceCreateInfo::default();
        let mut enabled_features = 0;
        assert_eq!(
            oidnw_gpu_vulkan_requirements_get(
                requirements,
                &mut create,
                &mut enabled_features,
                error.as_mut_ptr(),
                error.len()
            ),
            OIDNW_OK
        );
        let priorities = [1.];
        let queues = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(family)
            .queue_priorities(&priorities)];
        create.queue_create_info_count = queues.len() as u32;
        create.p_queue_create_infos = queues.as_ptr().cast();
        let raw = instance.create_device(physical, &create, None).unwrap();
        struct Lock {
            held: std::sync::atomic::AtomicBool,
            calls: AtomicUsize,
        }
        unsafe extern "C" fn lock(p: *mut c_void) {
            let l = unsafe { &*p.cast::<Lock>() };
            assert!(!l.held.swap(true, Ordering::Acquire));
            l.calls.fetch_add(1, Ordering::Relaxed);
        }
        unsafe extern "C" fn unlock(p: *mut c_void) {
            assert!(
                unsafe { &*p.cast::<Lock>() }
                    .held
                    .swap(false, Ordering::Release)
            );
        }
        let mut access = Lock {
            held: std::sync::atomic::AtomicBool::new(false),
            calls: AtomicUsize::new(0),
        };
        let desc = VulkanDeviceDesc {
            next: std::ptr::null(),
            flags: 0,
            get_instance_proc_addr: Some(entry.static_fn().get_instance_proc_addr),
            instance: instance.handle(),
            physical_device: physical,
            device: raw.handle(),
            queue: raw.get_device_queue(family, 0),
            queue_family_index: family,
            queue_index: 0,
            instance_create_info: (&instance_create as *const vk::InstanceCreateInfo).cast(),
            device_create_info: (&create as *const vk::DeviceCreateInfo).cast(),
            lock: Some(lock),
            unlock: Some(unlock),
            user_data: (&mut access as *mut Lock).cast(),
            max_pending_submissions: 1,
        };
        let mut error = [0i8; 4096];
        let mut device = std::ptr::null_mut();
        assert_eq!(
            oidnw_gpu_device_import_vulkan(&desc, &mut device, error.as_mut_ptr(), error.len()),
            0,
            "{}",
            std::ffi::CStr::from_ptr(error.as_ptr()).to_string_lossy()
        );
        let props = instance.get_physical_device_memory_properties(physical);
        let mut allocations = Vec::new();
        let mut imported = Vec::new();
        for _ in 0..4 {
            let create = vk::BufferCreateInfo::default()
                .size(4096)
                .usage(vk::BufferUsageFlags::STORAGE_BUFFER);
            let buffer = raw.create_buffer(&create, None).unwrap();
            let req = raw.get_buffer_memory_requirements(buffer);
            let index = props
                .memory_types_as_slice()
                .iter()
                .enumerate()
                .position(|(i, p)| {
                    req.memory_type_bits & (1 << i) != 0
                        && p.property_flags
                            .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
                })
                .unwrap();
            let memory = raw
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(req.size)
                        .memory_type_index(index as u32),
                    None,
                )
                .unwrap();
            raw.bind_buffer_memory(buffer, memory, 0).unwrap();
            let mut handle = std::ptr::null_mut();
            assert_eq!(
                oidnw_gpu_buffer_import_vulkan(
                    device,
                    buffer,
                    &create,
                    &mut handle,
                    error.as_mut_ptr(),
                    error.len()
                ),
                0
            );
            let mut duplicate = std::ptr::null_mut();
            assert_eq!(
                oidnw_gpu_buffer_import_vulkan(
                    device,
                    buffer,
                    &create,
                    &mut duplicate,
                    error.as_mut_ptr(),
                    error.len()
                ),
                OIDNW_ERROR_INVALID_ARGUMENT
            );
            imported.push(handle);
            allocations.push((buffer, memory));
        }
        let mut kind =
            vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE);
        let create = vk::SemaphoreCreateInfo::default().push_next(&mut kind);
        let ready = raw.create_semaphore(&create, None).unwrap();
        let done = raw.create_semaphore(&create, None).unwrap();
        let mut ready_handle = std::ptr::null_mut();
        let mut done_handle = std::ptr::null_mut();
        assert_eq!(
            oidnw_gpu_sync_import_vulkan(
                device,
                ready,
                &create,
                &mut ready_handle,
                error.as_mut_ptr(),
                error.len()
            ),
            0
        );
        assert_eq!(
            oidnw_gpu_sync_import_vulkan(
                device,
                done,
                &create,
                &mut done_handle,
                error.as_mut_ptr(),
                error.len()
            ),
            0
        );
        super::tests::exercise_failure_contract(
            device,
            imported,
            ready_handle,
            done_handle,
            OIDNW_FILTER_RUNTIME_CHECKS,
            || raw.get_semaphore_counter_value(done).unwrap(),
            |v| {
                raw.signal_semaphore(&vk::SemaphoreSignalInfo::default().semaphore(ready).value(v))
                    .unwrap()
            },
        );
        assert!(!access.held.load(Ordering::Acquire));
        assert!(access.calls.load(Ordering::Relaxed) > 8);
        raw.device_wait_idle().unwrap();
        for (buffer, memory) in allocations {
            raw.destroy_buffer(buffer, None);
            raw.free_memory(memory, None);
        }
        raw.destroy_semaphore(ready, None);
        raw.destroy_semaphore(done, None);
        oidnw_gpu_vulkan_requirements_destroy(requirements);
        raw.destroy_device(None);
        debug.destroy_debug_utils_messenger(messenger, None);
        instance.destroy_instance(None);
        assert_eq!(VALIDATION_ERRORS.load(Ordering::Relaxed), 0);
    }
}
